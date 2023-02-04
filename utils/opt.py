import configargparse

from data import dataset_dict
from models import MODEL_ZOO


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log', help='directory to store ckpts and logs')
    parser.add_argument('--model_name', type=str, default='PaletteTensorVM', choices=MODEL_ZOO.keys())
    parser.add_argument("--ckpt", type=str, default=None, help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_reload", type=int, default=0)
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # dataset options
    parser.add_argument("--datadir", type=str, required=True, help='input data directory')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)
    parser.add_argument('--dataset_name', type=str, default='blender', choices=dataset_dict.keys())

    # training options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02, help='learning rate')
    parser.add_argument("--lr_basis", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help='number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1, help='reset lr to inital after upsampling')
    # progressive upsampling and masking
    parser.add_argument('--N_voxel_init', type=int, default=100 ** 3)
    parser.add_argument('--N_voxel_final', type=int, default=300 ** 3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0, help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0.0, help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0, help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0, help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help='loss weight')
    parser.add_argument("--Plt_bd_weight", type=float, default=0.0, help='loss weight')
    parser.add_argument("--Plt_opaque_conv_weight", type=float, default=0.0, help='loss weight')
    parser.add_argument("--Plt_opaque_sps_weight", type=float, default=0.0, help='loss weight')
    parser.add_argument('--soft_l0_sharpness', type=float, default=24., help='sharpness of soft L0')

    # tensorf model
    ## volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)
    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001, help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25, help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
    ## network decoder
    parser.add_argument("--shadingMode", type=str, default="PLT_Direct", help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6, help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6, help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6, help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128, help='hidden feature channel in MLP')

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true", help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=int(1e6), help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio', type=float, default=0.5)
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5, help='N images to visualize during training')
    parser.add_argument("--vis_every", type=int, default=10000, help='frequency of visualize the image during training')
    parser.add_argument("--progress_refresh_every", type=int, default=10, help='how many iterations to show psnrs or iters')

    # palette options
    parser.add_argument('--palette_path', type=str, default='', help='palette path')
    parser.add_argument('--learn_palette', action='store_true', help='learnable palette')
    parser.add_argument('--palette_init', type=str, default='userinput', help='initialization of palette')

    return parser


# Compare args1 with args2
def compare_args(args1, args2, keys=[]):
    if len(keys) == 0:
        keys = vars(args2).keys()

    mismatched_keys = []

    for k in keys:
        if not hasattr(args1, k) or getattr(args1, k) != getattr(args2, k):
            mismatched_keys.append(k)
            
    return len(mismatched_keys) == 0, mismatched_keys
