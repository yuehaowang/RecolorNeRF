import numpy as np
from einops import rearrange
from matplotlib.colors import rgb2hex as mpl_rgb2hex


def sort_palette(rgbs, palette_rgb):
    dist = rearrange(rgbs, 'N C -> N 1 C') - rearrange(palette_rgb, 'P C -> 1 P C')
    dist = np.linalg.norm(dist, axis=-1)
    dist = np.argmin(dist, axis=-1)

    dist = np.argsort(np.bincount(dist))

    # bg = np.ones(3) if dataset.white_bg else np.zeros(3)
    # palette_rgb = [tuple(a.tolist()) for a in palette_rgb[dist.cpu().numpy()] if not np.allclose(a, bg)]
    # palette_rgb.append(tuple(bg.tolist()))
    palette_rgb = [tuple(a) for a in palette_rgb[dist].tolist()]
    return np.array(palette_rgb)


def hex2rgb(hex):
    if hex.startswith('#'):
        hex = hex[1:]
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)
    return tuple(rgb)

rgb2hex = mpl_rgb2hex
