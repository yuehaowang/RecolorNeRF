import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
from tqdm import trange, tqdm

from data import dataset_dict
from models import MODEL_ZOO
from models.loss import TVLoss, PaletteBoundLoss
from engine.eval import evaluation, evaluation_path
from utils.recon import convert_sdf_samples_to_ply
from utils.render import chunkify_render, N_to_reso, cal_n_samples
from utils.fs import seek_checkpoint
from utils.color import sort_palette
from utils.palette_utils.Additive_mixing_layers_extraction import DCPPointTriangle, Hull_Simplification_determined_version


class SimpleSampler:
    def __init__(self, train_dataset, batch):
        self.all_rays = train_dataset.all_rays
        self.all_rgbs = train_dataset.all_rgbs
        self.total = self.all_rays.shape[0]
        self.curr = self.total
        self.ids = None
        self.batch = batch

    def apply_filter(self, func, *args, **kwargs):
        self.all_rays, self.all_rgbs = func(self.all_rays, self.all_rgbs, *args, **kwargs)
        self.total = self.all_rays.shape[0]
        self.curr = self.total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]

    def getbatch(self, device):
        ids = self.nextids()
        return self.all_rays[ids].to(device), self.all_rgbs[ids].to(device)


class Trainer:
    def __init__(self, args, run_dir, ckpt_dir, tb_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = chunkify_render

        self.args = args
        self.optimizer = None
        self.summary_writer = None
        self.trainingSampler = None

        self.run_dir = run_dir
        self.ckpt_dir = ckpt_dir
        self.tb_dir = tb_dir

        # init dataset
        dataset = dataset_dict[args.dataset_name]
        self.train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
        self.test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)

        # init parameters
        self.aabb = self.train_dataset.scene_bbox.to(self.device)
        self.reso_cur = N_to_reso(args.N_voxel_init, self.aabb)
        self.nSamples = min(args.nSamples, cal_n_samples(self.reso_cur, args.step_ratio))
        self.palette_prior, self.plt_bds_convhull_vtx = self.build_palette(args.palette_path)
        
        np.save(os.path.join(run_dir, 'palette_prior.npy'), self.palette_prior.numpy())

        print("[trainer init] aabb", self.aabb.tolist())
        print("[trainer init] num of render samples", self.nSamples)
        print("[trainer init] palette shape", self.palette_prior.shape)

    def build_palette(self, filepath, is_sort_palette=True):
        rgbs = self.train_dataset.all_rgbs
        if self.train_dataset.white_bg:
            fg = torch.lt(rgbs, 1.).any(dim=-1)
            rgbs = rgbs[fg]

        rgbs = rgbs.to(device='cpu', dtype=torch.double).numpy()
        hull = ConvexHull(rgbs)
        hull_vertices = hull.points[hull.vertices]
        if filepath:
            palette = np.load(filepath)
        else:
            simplify = True
            error_thres = 2. / 256.
            palette = Hull_Simplification_determined_version(rgbs, "", error_thres=error_thres) if simplify else hull_vertices
            if is_sort_palette:
                palette = sort_palette(rgbs, palette)
        palette = torch.from_numpy(palette).float()
        
        return palette, hull_vertices

    def build_network(self):
        args = self.args

        ckpt_path = seek_checkpoint(args, self.ckpt_dir)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            kwargs = ckpt['kwargs']
            kwargs.update({'device': self.device})
            tensorf = MODEL_ZOO[args.model_name](**kwargs)
            tensorf.load(ckpt)
            del ckpt
        else:
            n_lamb_sigma = args.n_lamb_sigma
            n_lamb_sh = args.n_lamb_sh
            near_far = self.train_dataset.near_far
            palette = self.palette_prior

            tensorf = MODEL_ZOO[args.model_name](
                self.aabb, self.reso_cur, self.device,
                density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                app_dim=args.data_dim_color, near_far=near_far,
                shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                density_shift=args.density_shift, distance_scale=args.distance_scale,
                pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct,
                palette=palette, learn_palette=args.learn_palette, palette_init=args.palette_init,
                soft_l0_sharpness=args.soft_l0_sharpness)

        return tensorf

    def train(self):
        args = self.args
        white_bg = self.train_dataset.white_bg

        # create model
        tensorf = self.build_network()
        tensorf.train()

        # create optimizer
        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        
        # loss function and regularization
        self.tvreg = TVLoss()
        self.plt_bd_reg = PaletteBoundLoss(self.plt_bds_convhull_vtx)

        self.Ortho_reg_weight = args.Ortho_weight
        print("[trainer train] initial Ortho_reg_weight", self.Ortho_reg_weight)
        self.L1_reg_weight = args.L1_weight_inital
        print("[trainer train] initial L1_reg_weight", self.L1_reg_weight)
        self.TV_weight_density = args.TV_weight_density
        self.TV_weight_app = args.TV_weight_app
        print(f"[trainer train] initial TV_weight density: {self.TV_weight_density} appearance: {self.TV_weight_app}")
        self.Plt_bd_weight = args.Plt_bd_weight
        print("[trainer train] initial Plt_bd_weight", self.Plt_bd_weight)
        self.Plt_opaque_conv_weight = args.Plt_opaque_conv_weight
        print("[trainer train] initial Plt_opaque_conv_weight", self.Plt_opaque_conv_weight)
        self.Plt_opaque_sps_weight = args.Plt_opaque_sps_weight
        print("[trainer train] initial Plt_opaque_sps_weight", self.Plt_opaque_sps_weight)

        if args.lr_decay_iters > 0:
            self.lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
        else:
            args.lr_decay_iters = args.n_iters
            self.lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
        print("[trainer train] lr decay", self.lr_factor, args.lr_decay_target_ratio, args.lr_decay_iters)

        # list of progressive voxel numbers (linear in logrithmic space)
        self.N_voxel_list = torch.round(torch.exp(torch.linspace(
            np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(args.upsamp_list) + 1))).long().tolist()[1:]

        # recorder
        PSNRs, PSNRs_test = [], [0]
        self.summary_writer = SummaryWriter(log_dir=self.tb_dir)

        # data sampler
        self.trainingSampler = SimpleSampler(self.train_dataset, args.batch_size)
        if not args.ndc_ray:
            self.trainingSampler.apply_filter(tensorf.filtering_rays, bbox_only=True)

        # start training
        print(f'=== training ======> {args.expname}')
        
        torch.cuda.empty_cache()
        
        pbar = trange(args.n_iters, miniters=args.progress_refresh_every, file=sys.stdout, position=0, leave=True)
        for iteration in pbar:
            ###### Core optimization ######
            batch_train = self.trainingSampler.getbatch(device=self.device)
            loss_dict = self.train_one_batch(tensorf, iteration, *batch_train)
            
            ###### Logging ######
            total_loss = loss_dict['total_loss']
            self.summary_writer.add_scalar('train/total_loss', total_loss, global_step=iteration)

            img_loss = loss_dict['img_loss']
            PSNRs.append(-10.0 * np.log(img_loss) / np.log(10.0))
            self.summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
            self.summary_writer.add_scalar('train/mse', img_loss, global_step=iteration)

            if 'ortho_reg_loss' in loss_dict:
                ortho_reg_loss = loss_dict['ortho_reg_loss']
                self.summary_writer.add_scalar('train/reg_ortho', ortho_reg_loss, global_step=iteration)
            if 'L1_reg_loss' in loss_dict:
                L1_reg_loss = loss_dict['L1_reg_loss']
                self.summary_writer.add_scalar('train/reg_L1', L1_reg_loss, global_step=iteration)

            if 'tv_loss_den' in loss_dict:
                tv_loss_den = loss_dict['tv_loss_den']
                self.summary_writer.add_scalar('train/reg_tv_density', tv_loss_den, global_step=iteration)
            if 'tv_loss_app' in loss_dict:
                tv_loss_app = loss_dict['tv_loss_app']
                self.summary_writer.add_scalar('train/reg_tv_app', tv_loss_app, global_step=iteration)

            if 'plt_bd_loss' in loss_dict:
                plt_bd_loss = loss_dict['plt_bd_loss']
                self.summary_writer.add_scalar('train/reg_plt_bd', plt_bd_loss, global_step=iteration)
            if 'opq_conv_loss' in loss_dict:
                opq_conv_loss = loss_dict['opq_conv_loss']
                self.summary_writer.add_scalar('train/reg_plt_opq_conv', opq_conv_loss, global_step=iteration)
            if 'opq_sps_loss' in loss_dict:
                opq_sps_loss = loss_dict['opq_sps_loss']
                self.summary_writer.add_scalar('train/reg_plt_opq_sps', opq_sps_loss, global_step=iteration)

            # Print the current values of the losses.
            if iteration % args.progress_refresh_every == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d}:'
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                    + f' mse = {img_loss:.6f}'
                )
                PSNRs = []

            # Evaluation on testset
            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
                try:
                    print(f'== evaluation ======> {args.N_vis} views')
                    savePath = Path(self.run_dir, f'testset_vis_{iteration:06d}')
                    PSNRs_test = evaluation(self.test_dataset, tensorf, args, self.renderer, os.fspath(savePath),
                                            N_vis=args.N_vis, N_samples=self.nSamples, white_bg=white_bg, ndc_ray=args.ndc_ray,
                                            compute_extra_metrics=False, save_gt=True)
                    self.summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
                    print(f'=== continue training ======>')
                except Exception as e:
                    print(f'Evaluation failed: {e}')

            ###### Progressive updating grid resolution ######
            self.update_grid_resolution(tensorf, iteration)
        print('training finished!')

        tensorf.save(f'{self.ckpt_dir}/{args.expname}_last.th')
        self.render_test(tensorf)
        print('evaluation finished!')

    def train_one_batch(self, tensorf, iteration, rays_train, rgb_train):
        args = self.args
        white_bg = self.train_dataset.white_bg
        ndc_ray = args.ndc_ray
        
        loss_dict = {}

        # render training rays
        res = self.renderer(
            rays_train, tensorf, chunk=args.batch_size, N_samples=self.nSamples,
            white_bg=white_bg, ndc_ray=ndc_ray, device=self.device, is_train=True,
            ret_sparsity_norm_map=True, ret_convexity_residual_map=True, ret_rgb0_map=True, ret_opaque_map=True)

        # Loss
        img_loss = torch.mean((res['rgb_map'] - rgb_train) ** 2)
        loss_dict['img_loss'] = img_loss.clone().detach().item()

        total_loss = img_loss

        if 'rgb0_map' in res:
            img_loss_0 = torch.mean((res['rgb0_map'] - rgb_train) ** 2)
            total_loss = total_loss + img_loss_0
        
        # Regularization
        if self.Ortho_reg_weight > 0:
            loss_reg_ortho = tensorf.vector_comp_diffs()
            total_loss += self.Ortho_reg_weight * loss_reg_ortho
            loss_dict['ortho_reg_loss'] = loss_reg_ortho.clone().detach().item()
        if self.L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += self.L1_reg_weight * loss_reg_L1
            loss_dict['L1_reg_loss'] = loss_reg_L1.clone().detach().item()

        if self.TV_weight_density > 0:
            # self.TV_weight_density *= self.lr_factor
            loss_tv = tensorf.TV_loss_density(self.tvreg)
            total_loss = total_loss + loss_tv * self.TV_weight_density
            loss_dict['tv_loss_den'] = loss_tv.clone().detach().item()
        if self.TV_weight_app > 0:
            # self.TV_weight_app *= self.lr_factor
            loss_tv = tensorf.TV_loss_app(self.tvreg)
            total_loss = total_loss + loss_tv * self.TV_weight_app
            loss_dict['tv_loss_app'] = loss_tv.clone().detach().item()

        if self.Plt_bd_weight > 0:
            loss_plt_bd = self.plt_bd_reg(tensorf.renderModule.palette.get_palette_array())
            total_loss = total_loss + loss_plt_bd * self.Plt_bd_weight
            loss_dict['plt_bd_loss'] = loss_plt_bd.clone().detach().item()
        if self.Plt_opaque_conv_weight > 0 and 'convexity_residual_map' in res:
            loss_opq_conv = torch.mean(res['convexity_residual_map'])
            total_loss = total_loss + loss_opq_conv * self.Plt_opaque_conv_weight
            loss_dict['opq_conv_loss'] = loss_opq_conv.clone().detach().item()
        if self.Plt_opaque_sps_weight > 0 and 'sparsity_norm_map' in res:
            loss_opq_sps = torch.mean(res['sparsity_norm_map'])
            total_loss = total_loss + loss_opq_sps * self.Plt_opaque_sps_weight
            loss_dict['opq_sps_loss'] = loss_opq_sps.clone().detach().item()

        loss_dict['total_loss'] = total_loss.clone().detach().item()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # LR shrinkage
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * self.lr_factor

        return loss_dict

    def update_grid_resolution(self, tensorf, iteration):
        args = self.args
        # init resolution
        upsamp_list = args.upsamp_list
        update_AlphaMask_list = args.update_AlphaMask_list

        if iteration in update_AlphaMask_list:
            # if self.reso_cur[0] * self.reso_cur[1] * self.reso_cur[2] < 256 ** 3:
            # update mask volume resolution
            reso_mask = self.reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                self.L1_reg_weight = args.L1_weight_rest
                print("[update_grid_resolution] set L1_reg_weight to", self.L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                self.trainingSampler.apply_filter(tensorf.filtering_rays)

        if iteration in upsamp_list:
            n_voxels = self.N_voxel_list.pop(0)
            self.reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            self.nSamples = min(args.nSamples, cal_n_samples(self.reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(self.reso_cur)

            if args.lr_upsample_reset:
                print("[update_grid_resolution] reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)

            grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    @torch.no_grad()
    def export_mesh(self):
        args = self.args
        ckpt = torch.load(args.ckpt, map_location=self.device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': self.device})
        tensorf = MODEL_ZOO[args.model_name](**kwargs)
        tensorf.load(ckpt)

        alpha, _ = tensorf.getDenseAlpha()
        convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)

    @torch.no_grad()
    def render_test(self, tensorf):
        args = self.args
        white_bg = self.test_dataset.white_bg
        ndc_ray = args.ndc_ray

        tensorf.eval()

        logfolder = Path(self.run_dir)

        PSNRs_test = None
        if args.render_train:
            print(f'=== render train ======> {args.expname}')
            filePath = logfolder / 'render_train'
            PSNRs_test = evaluation(self.train_dataset, tensorf, args, self.renderer, os.fspath(filePath),
                                    N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=self.device)
            print(f'mean psnr: {np.mean(PSNRs_test)}')

        if args.render_test:
            print(f'=== render test ======> {args.expname}')
            filePath = logfolder / 'render_test'
            PSNRs_test = evaluation(self.test_dataset, tensorf, args, self.renderer, os.fspath(filePath),
                                    N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=self.device)
            print(f'mean psnr: {np.mean(PSNRs_test)}')

        if args.render_path:
            filePath = logfolder / 'render_path'
            c2ws = self.test_dataset.render_path
            print('=== render path ======>', c2ws.shape)
            evaluation_path(self.test_dataset, tensorf, c2ws, self.renderer, os.fspath(filePath),
                            N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, save_video=True, device=self.device)

        return PSNRs_test
