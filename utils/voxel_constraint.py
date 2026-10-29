"""Fixed sparse occupancy with implicit Chebyshev (26-neighbour) dilation.

Only sorted int64 keys of the original occupied cells are stored. Querying
neighbour cells is exactly equivalent to materializing dilation, without its
(2*d+1)**3 storage multiplier. No dense world-sized grid is allocated.
"""
import itertools
import math
import numpy as np
import torch


class SparseVoxelGrid:
    def __init__(self, points, voxel_size, dilation):
        if not math.isfinite(voxel_size) or voxel_size <= 0 or dilation < 0:
            raise ValueError("voxel_size must be positive and dilation non-negative")
        xyz = points.detach().cpu().double().numpy()
        if xyz.shape[0] == 0 or not np.isfinite(xyz).all():
            raise ValueError("Voxel initialization requires nonempty finite points")
        cells = np.floor(xyz / voxel_size)
        if np.abs(cells).max() >= 2**62:
            raise ValueError("Voxel coordinates exceed int64 range; increase voxel_size")
        cells = cells.astype(np.int64)
        lower = cells.min(axis=0)
        shape = [int(x) + 1 for x in cells.max(axis=0) - lower]
        if math.prod(shape) >= 2**63:
            raise ValueError("Voxel key range exceeds int64; increase voxel_size")
        strides = np.array([shape[1]*shape[2], shape[2], 1], dtype=np.int64)
        self.voxel_size, self.dilation = voxel_size, dilation
        self.lower = torch.from_numpy(lower).to(points.device)
        self.shape = torch.tensor(shape, device=points.device)
        self.strides = torch.from_numpy(strides).to(points.device)
        self.keys = torch.from_numpy(np.unique(((cells - lower)*strides).sum(axis=1))).to(points.device)

    def state_dict(self):
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in vars(self).items()}

    @classmethod
    def from_state_dict(cls, state, device):
        grid = cls.__new__(cls)
        for k, v in state.items():
            setattr(grid, k, v.to(device) if torch.is_tensor(v) else v)
        return grid

    @torch.no_grad()
    def contains(self, points, chunk_size=32768):
        result = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        d = self.dilation
        for start in range(0, len(points), chunk_size):
            xyz = points[start:start + chunk_size]
            coords = torch.floor(xyz.double() / self.voxel_size) - self.lower
            valid = torch.isfinite(coords).all(dim=1) & ((coords >= -d) & (coords < self.shape + d)).all(dim=1)
            ids = valid.nonzero(as_tuple=True)[0]
            cells = coords[ids].long()
            hit = torch.zeros(len(ids), dtype=torch.bool, device=points.device)
            for offset in itertools.product(range(-d, d+1), repeat=3):
                neighbour = cells + cells.new_tensor(offset)
                bounded = ((neighbour >= 0) & (neighbour < self.shape)).all(dim=1)
                # Clamp before encoding so out-of-bounds values cannot overflow keys.
                neighbour = torch.minimum(neighbour.clamp_min(0), self.shape - 1)
                keys = (neighbour * self.strides).sum(dim=1)
                pos = torch.searchsorted(self.keys, keys).clamp_max(len(self.keys)-1)
                hit |= bounded & (self.keys[pos] == keys)
            result[start + ids] = hit
        return result
