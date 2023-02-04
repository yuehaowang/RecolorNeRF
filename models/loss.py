import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import knn_points
from scipy.spatial import ConvexHull, Delaunay
from operator import itemgetter
from utils.palette_utils.Additive_mixing_layers_extraction import DCPPointTriangle


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class PaletteBoundLoss(nn.Module):
    def __init__(self, cvh_vtx):
        super(PaletteBoundLoss, self).__init__()

        self.hull = ConvexHull(cvh_vtx)
        self.de = Delaunay(cvh_vtx)
        self.hull_vertices = torch.from_numpy(cvh_vtx)

    def forward(self, inp_points, w_in=1e-3, w_out=1.):
        points = inp_points.detach().to('cpu', dtype=torch.double).numpy()
        simplex = self.de.find_simplex(points, tol=1e-8)
        loss = knn_points(inp_points[simplex >= 0].unsqueeze(0).double(), self.hull_vertices[None].to(inp_points.device, dtype=torch.double),
                          K=1, return_sorted=False).dists
        loss = w_in / n_in * loss.sum() if (n_in := loss.nelement()) else loss.sum()
        ind, = np.nonzero(simplex < 0)
        # loss = torch.Tensor([0.]).to(inp_points.device)
        points = [min((DCPPointTriangle(pts, self.hull.points[j]) for j in self.hull.simplices),
                      key=itemgetter('distance'))['closest'] for pts in points[ind]]
        if points:
            points = torch.asarray(points, device=inp_points.device, dtype=inp_points.dtype)
            loss = loss + w_out * F.mse_loss(inp_points[ind], points, reduction='none').sum(dim=-1).max()

            assert torch.isfinite(loss)
        return loss


def soft_L0_norm(x, scale=12.):
    '''
    use sigmoid to approximate L0 norm
    @scale: control the sharpness, should be larger than 12.
    '''
    return torch.sigmoid(scale * x - 6.)
