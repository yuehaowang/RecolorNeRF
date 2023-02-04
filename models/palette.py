import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transforms import axisangle_to_R


class FreeformPalette(nn.Module):
    def __init__(self, num, is_train=True, palette_init_scheme='userinput', random_std=1., initial_palette=None):
        super(FreeformPalette, self).__init__()

        if is_train:
            if palette_init_scheme == 'userinput' and initial_palette is not None:
                random_palette = torch.normal(mean=torch.zeros(num, 3), std=torch.ones(num, 3) * random_std).to(initial_palette.device)
                initial_palette = initial_palette + random_palette
            elif palette_init_scheme == 'std_normal':
                initial_palette = torch.normal(mean=torch.zeros(num, 3), std=torch.ones(num, 3) * random_std)
            elif palette_init_scheme == 'recentered_normal':
                initial_palette = torch.normal(mean=torch.ones(num, 3) * 0.5, std=torch.ones(num, 3) * random_std)
            else:
                raise NotImplementedError

            self.palette = torch.nn.Parameter(torch.as_tensor(initial_palette, dtype=torch.float32))
        else:
            self.register_buffer('palette', torch.as_tensor(initial_palette, dtype=torch.float32), persistent=False)
    
    def get_palette_array(self):
        return self.palette

    def forward(self, weights):
        plt_cart = self.get_palette_array()
        return weights @ plt_cart

    def __rmatmul__(self, w):
        return self(w)

