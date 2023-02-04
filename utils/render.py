import numpy as np
import torch


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


def chunkify_render(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda', **kwargs):
    # rend_maps, alphas, depth_maps, weights, render_bufs = [], [], [], [], []
    ret = {}
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        res_dict = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples, **kwargs)

        for k in res_dict:
            if not k in ret:
                ret[k] = []
            ret[k].append(res_dict[k]) 
        # rend_maps.append(rend_map)
        # depth_maps.append(depth_map)
        # if 'acc_map' in extras:
        #     alphas.append(extras['acc_map'])
        # if 'weight' in extras:
        #     weights.append(extras['weight'])
        # if 'render_buf' in extras:
        #     render_bufs.append(extras['render_buf'])

    # rend_maps = torch.cat(rend_maps)
    # depth_maps = torch.cat(depth_maps)

    for k in ret:
        ret[k] = torch.cat(ret[k])

    # if len(alphas) > 0:
    #     alphas = torch.cat(alphas)
    # else:
    #     alphas = None

    # if len(weights) > 0:
    #     weights = torch.cat(weights)
    # else:
    #     weights = None

    # if len(render_bufs) > 0:
    #     render_bufs = torch.cat(render_bufs)
    # else:
    #     render_bufs = None
    
    return ret