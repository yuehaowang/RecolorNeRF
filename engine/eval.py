import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from einops import rearrange
from tqdm.auto import tqdm, trange

from utils.ray import get_rays, ndc_rays_blender
from utils.metrics import rgb_ssim, rgb_lpips
from utils.vis import visualize_depth_numpy, visualize_palette_components_numpy


@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, N_samples=-1, white_bg=False, ndc_ray=False,
                compute_extra_metrics=True, save_gt=False, save_video=False, palette=None, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    plt_decomp_maps = []
    ssims, l_alex, l_vgg = [], [], []

    if savePath is not None:
        os.makedirs(savePath, exist_ok=True)

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

    if palette is None and hasattr(tensorf, 'get_palette_array'):
        palette = tensorf.get_palette_array().cpu()

    test_rays = test_dataset.all_rays[0::img_eval_interval]
    pbar = trange(len(test_rays), file=sys.stdout, position=0, leave=True)
    for idx in pbar:
        samples = test_rays[idx]

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        # plt_map, _, depth_map, _, _
        res = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, ndc_ray=ndc_ray, white_bg=white_bg, device=device,
                       ret_opaque_map=True, palette=palette)
        # rgb_map = plt_map[..., :3].clamp(0.0, 1.0)
        rgb_map = res['rgb_map']
        depth_map = res['depth_map']
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

            if save_gt:
                gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
                imageio.imwrite(os.path.join(savePath, f'gt_{idx:03d}.png'), gt_rgb)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        depth_maps.append(depth_map)

        is_vis_plt = (palette is not None) and ('opaque_map' in res)
        if is_vis_plt:
            opaque = rearrange(res['opaque_map'], '(h w) c-> h w c', h=H, w=W).cpu()
            plt_decomp = visualize_palette_components_numpy(opaque.numpy(), palette.numpy())
            plt_decomp = (plt_decomp * 255).astype('uint8')
            plt_decomp_maps.append(plt_decomp)
        
        if savePath is not None:
            imageio.imwrite(os.path.join(savePath, f'rgb_{idx:03d}.png'), rgb_map)
            imageio.imwrite(os.path.join(savePath, f'depth_{idx:03d}.png'), depth_map)
            if is_vis_plt:
                imageio.imwrite(os.path.join(savePath, f'plt_decomp_{idx:03d}.png'), plt_decomp)

    if save_video and savePath is not None:
        fps = min(len(rgb_maps) / 5, 30)
        imageio.mimwrite(os.path.join(savePath, 'video_rgb.mp4'), np.stack(rgb_maps), fps=fps, quality=10)
        imageio.mimwrite(os.path.join(savePath, 'video_depth.mp4'), np.stack(depth_maps), fps=fps, quality=10)
        if len(plt_decomp_maps) > 0:
            imageio.mimwrite(os.path.join(savePath, 'video_palette_decomp.mp4'), np.stack(plt_decomp_maps), fps=fps, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        metrics = [psnr]
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            metrics += [psnr, ssim, l_a, l_v]
        
        np.savetxt(f'{savePath}/mean.txt', np.asarray(metrics))

    return PSNRs


@torch.no_grad()
def evaluation_path(test_dataset, tensorf, c2ws, renderer, savePath=None, N_samples=-1,
                    white_bg=False, ndc_ray=False, save_video=False, palette=None, device='cuda'):
    rgb_maps, depth_maps = [], []
    plt_decomp_maps = []

    if savePath is not None:
        os.makedirs(savePath, exist_ok=True)

    near_far = test_dataset.near_far

    if palette is None and hasattr(tensorf, 'get_palette_array'):
        palette = tensorf.get_palette_array().cpu()

    pbar = trange(len(c2ws), file=sys.stdout)
    for idx in pbar:
        c2w = c2ws[idx]

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        res = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, ndc_ray=ndc_ray, white_bg=white_bg, device=device,
                       ret_opaque_map=True, palette=palette)
        # rgb_map = rend_map[..., :3].clamp(0.0, 1.0)
        rgb_map = res['rgb_map']
        depth_map = res['depth_map']

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        depth_maps.append(depth_map)

        is_vis_plt = (palette is not None) and ('opaque_map' in res)
        if is_vis_plt:
            opaque = rearrange(res['opaque_map'], '(h w) c-> h w c', h=H, w=W).cpu()
            plt_decomp = visualize_palette_components_numpy(opaque.numpy(), palette.numpy())
            plt_decomp = (plt_decomp * 255).astype('uint8')
            plt_decomp_maps.append(plt_decomp)

        if savePath is not None:
            imageio.imwrite(os.path.join(savePath, f'rgb_{idx:03d}.png'), rgb_map)
            imageio.imwrite(os.path.join(savePath, f'depth_{idx:03d}.png'), depth_map)
            if is_vis_plt:
                imageio.imwrite(os.path.join(savePath, f'plt_decomp_{idx:03d}.png'), plt_decomp)

    if save_video and savePath is not None:
        fps = min(len(rgb_maps) / 5, 30)
        imageio.mimwrite(os.path.join(savePath, 'video_rgb.mp4'), np.stack(rgb_maps), fps=fps, quality=8)
        imageio.mimwrite(os.path.join(savePath, 'video_depth.mp4'), np.stack(depth_maps), fps=fps, quality=8)
        if len(plt_decomp_maps) > 0:
            imageio.mimwrite(os.path.join(savePath, 'video_palette_decomp.mp4'), np.stack(plt_decomp_maps), fps=fps, quality=8)
