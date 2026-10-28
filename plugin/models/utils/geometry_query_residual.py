"""A bounded, initially identical update read from prior polyline geometry.

The input boxes use MapTracker's normalized ego coordinates: x increases right
and y increases upward. The native decoder reverses y before sampling BEV
(MapTransformer.py), so its corresponding grid is (2*x - 1, 1 - 2*y).
Only the new projection is trainable. No historical state or IDs are created.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class GeometryQueryResidual(nn.Module):
    def __init__(self, in_channels=256, embed_dims=512, max_delta=0.1):
        super().__init__()
        if not math.isfinite(max_delta) or not 0 <= max_delta <= 1:
            raise ValueError('max_delta must be finite and in [0, 1]')
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.max_delta = float(max_delta)
        # Linear's default initialization consumes random numbers even though
        # we overwrite it. Keep the baseline CPU RNG stream unchanged.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(in_channels, embed_dims, bias=False)
            nn.init.zeros_(self.projection.weight)

    def sample_bev_features(self, boxes, bev):
        """Return vertex-mean features and a per-curve support mask.

        The geometry is detached. Strict interior points are used because the
        parent clips transformed references to [0, 1]; a clipped border point
        cannot be distinguished from genuine boundary support. This also
        conservatively discards genuine vertices exactly on the ROI border.
        """
        if bev.ndim != 3 or bev.shape[0] != self.in_channels:
            raise ValueError('bev must have shape [in_channels, H, W]')
        if boxes.ndim != 2 or boxes.shape[1] == 0 or boxes.shape[1] % 2:
            raise ValueError('boxes must have shape [N, 2 * num_points]')
        if boxes.device != bev.device:
            raise ValueError('boxes and bev must be on the same device')
        if len(boxes) == 0:
            return bev.new_empty((0, self.in_channels)), torch.zeros(
                0, dtype=torch.bool, device=bev.device)

        points = boxes.detach().reshape(len(boxes), -1, 2)
        valid = torch.isfinite(points).all(-1)
        valid = valid & (points > 0).all(-1) & (points < 1).all(-1)
        # Never send NaN/Inf grids into grid_sample, including invalid vertices
        # in a curve that also has valid vertices.
        safe_points = torch.where(valid[..., None], points,
                                  torch.full_like(points, 0.5))
        grid = torch.stack((2 * safe_points[..., 0] - 1,
                            1 - 2 * safe_points[..., 1]), -1)
        grid = grid.to(dtype=bev.dtype)
        features = F.grid_sample(bev[None], grid[None], mode='bilinear',
                                 padding_mode='zeros', align_corners=False)[0]
        # [channels, curves, vertices] -> [curves, channels]
        weights = valid.to(features.dtype)[None]
        pooled = (features * weights).sum(-1).T
        pooled = pooled / valid.sum(-1).clamp_min(1).to(pooled.dtype)[:, None]
        return pooled, valid.any(-1)

    def forward(self, queries, boxes, bev):
        if queries.ndim != 2 or queries.shape[1] != self.embed_dims:
            raise ValueError('queries must have shape [N, embed_dims]')
        if len(queries) != len(boxes):
            raise ValueError('queries and boxes must have the same length')
        base = queries.detach()
        pooled, supported = self.sample_bev_features(boxes, bev)
        if not len(queries):
            return base
        pooled = F.layer_norm(pooled, (self.in_channels,))
        delta = self.projection(pooled).tanh()
        rms = base.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        delta = delta * self.max_delta * rms
        # The final where also guards future changes to the branch against a
        # spurious nonzero update when all vertices have been clipped out.
        return torch.where(supported[:, None], base + delta, base)
