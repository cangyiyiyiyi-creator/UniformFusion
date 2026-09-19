import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
# [ADDED] optional MinkowskiEngine import; 2D training does not need it
try:
    from MinkowskiEngine import SparseTensor  # type: ignore
    ME_AVAILABLE = True
except Exception:
    class SparseTensor:  # simple placeholder so type checks pass; a friendly error is raised when it is really used
        pass
    ME_AVAILABLE = False


# ======================
# [ADDED] small helper that raises a friendlier error when MinkowskiEngine is missing
# ======================
def _require_me():
    if not ME_AVAILABLE:
        raise ImportError(
            "MinkowskiEngine is not installed. It is only needed for sparse/3D settings;"
            "avoid instantiating Minkowski* layers in 2D tasks, or install MinkowskiEngine following the official documentation."
        )

class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors. """
    def __init__(self, dim):  
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        _require_me()  # [ADDED]
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
                self.gamma * (x.F * Nx) + self.beta + x.F,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors. """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        _require_me()  # [ADDED]
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob

        # [CHANGED] sample the mask directly on the device with torch.rand (instead of numpy.random)
        mask_list = []
        for coords in x.decomposed_coordinates:
            # one Bernoulli sample per sub-block (keep or drop)
            keep = (torch.rand((), device=x.device) > self.drop_prob)
            m = torch.ones(len(coords), 1, device=x.device) if keep else torch.zeros(len(coords), 1, device=x.device)
            mask_list.append(m)
        mask = torch.cat(mask_list, dim=0)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)

        return SparseTensor(
                x.F * mask,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors. """
    def __init__(self, normalized_shape, eps=1e-6):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, input):
        _require_me()  # [ADDED]
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)
            
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
