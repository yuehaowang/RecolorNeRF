import os
import shutil
from utils.opt import compare_args


def setup_wd(parser, args):
    # Create log dir and copy the config file
    run_dir = os.path.join(args.basedir, args.expname)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    tb_dir = os.path.join(run_dir, 'tensorboard')

    # Save/reload config
    if not os.path.exists(run_dir):
        # if not args.eval:
        print('>>> Starting a new working directory...')

        os.makedirs(run_dir)
        os.makedirs(ckpt_dir)
        os.makedirs(tb_dir)

        # Dump training configuration
        config_path = os.path.join(run_dir, 'args.txt')
        parser.write_config_file(args, [config_path])
        # Backup the default config file for checking
        shutil.copy(args.config, os.path.join(run_dir, 'config.txt'))
    else:
        print('>>> The specified working directory exists. Checking previous progress...')
        config_path = os.path.join(run_dir, 'args.txt')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_file, remainings = parser.parse_known_args(args=[], config_file_contents=f.read())
                if len(remainings) > 0:
                    print('These saved options are unrecognizable any more:', ' '.join(remainings))
                if args.no_reload:
                    print('Warning: Reloading is disabled.')
                # # Check hyper-parameters
                # argval_same, mismatched_keys = compare_args(args, config_file)
                # if not argval_same:
                #     raise RuntimeError('These options are inconsistent to the saved checkpoints:', ', '.join(mismatched_keys))

    return run_dir, ckpt_dir, tb_dir

def seek_checkpoint(args, ckpt_dir):
    ckpt_path = args.ckpt
    if not ckpt_path and not args.no_reload:
        # chronological order
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.th')]
        if len(ckpt_files) > 0:
            sort_fn = lambda x: os.path.splitext(x)[0]
            ckpt_files = sorted(ckpt_files, key=sort_fn)
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    return ckpt_path

def mk_suffix(s):
    return f'_{s}' if s else ''
def mk_prefix(s):
    return f'{s}_' if s else ''