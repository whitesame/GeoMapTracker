"""Per-instance current-feature attention with PETR-style inverse-camera 3D PE.
PV depth/range/order are candidate settings, not undisclosed author settings.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .history_geometry import metric_grid, project_points


class PriorAttention(nn.Module):
    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout)
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 2 * channels), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(2 * channels, channels))

    def forward(self, query, tokens):
        active = [i for i, t in enumerate(tokens) if len(t)]
        if not active:
            return query
        sequences = [tokens[i] for i in active]
        keys = pad_sequence(sequences)
        lens = torch.tensor([len(x) for x in sequences], device=query.device)
        mask = torch.arange(keys.shape[0], device=query.device)[None] >= lens[:, None]
        q = query[active][None]
        x = self.norm1(q + self.attn(q, keys, keys, key_padding_mask=mask, need_weights=False)[0])
        x = self.norm2(x + self.ffn(x))[0]
        result = query.clone()
        result[active] = x
        return result


class MapTrajectoryPrior(nn.Module):
    def __init__(self, roi=(60,30), embed_dims=512, in_channels=256, num_heads=8,
                 order=None, pv_levels=(0,), max_samples_per_instance=256,
                 ground_z=0., valid_threshold=.2, dropout=0., depth_num=32,
                 depth_start=1., depth_end=60., position_range=(-30,-15,-5,30,15,5),
                 use_category=True, mode=None):
        super().__init__()
        modes = {'disabled': (), 'ce': (), 'bev': ('bev',),
                 'pv': ('pv',), 'bev_pv': ('pv', 'bev')}
        if mode is not None:
            if mode not in modes:
                raise ValueError('Unknown MTP ablation mode: ' + str(mode))
            if order is None or mode == 'disabled':
                order = modes[mode]
            elif set(order) != set(modes[mode]):
                raise ValueError('MTP mode and explicit branch order disagree')
        elif order is None:
            order = ('pv', 'bev')
        if not set(order).issubset({'bev', 'pv'}) or len(set(order)) != len(order):
            raise ValueError('Each MTP branch may occur at most once')
        if depth_num <= 0 or depth_end <= depth_start or depth_start <= 0:
            raise ValueError('PV candidate depths require 0 < start < end')
        if max_samples_per_instance is not None and max_samples_per_instance <= 0:
            raise ValueError('max_samples_per_instance must be positive or None')
        self.disabled = mode == 'disabled'
        self.roi, self.order, self.pv_levels = roi, tuple(order), tuple(pv_levels)
        self.max_samples, self.z, self.threshold = max_samples_per_instance, ground_z, valid_threshold
        self.use_category = bool(use_category) and not self.disabled
        if self.use_category:
            self.category = nn.Embedding(3, embed_dims)
        if 'bev' in self.order:
            self.bev_proj = nn.Linear(in_channels, embed_dims)
            self.bev_pe = nn.Sequential(nn.Linear(2, embed_dims), nn.ReLU(), nn.Linear(embed_dims, embed_dims))
        if 'pv' in self.order:
            if not self.pv_levels:
                raise ValueError('PV MTP requires at least one feature level')
            self.pv_proj = nn.ModuleList([nn.Linear(in_channels, embed_dims) for _ in pv_levels])
            self.pv_pe = nn.Sequential(nn.Linear(depth_num * 3, embed_dims * 2), nn.ReLU(), nn.Linear(embed_dims * 2, embed_dims))
        self.branches = nn.ModuleDict({name: PriorAttention(embed_dims, num_heads, dropout) for name in order})
        self.register_buffer('depths', torch.linspace(depth_start, depth_end, depth_num), persistent=False)
        self.register_buffer('position_range', torch.tensor(position_range, dtype=torch.float32), persistent=False)

    def _limit(self, tensor):
        if self.max_samples and len(tensor) > self.max_samples:
            ids = torch.linspace(0, len(tensor)-1, self.max_samples, device=tensor.device).round().long()
            return tensor[ids]
        return tensor

    def _position3d(self, uv, inverse):
        # PETR principle: backproject a sampled image ray at multiple depths,
        # normalize in ego-space, inverse sigmoid, then a learned projection.
        d = self.depths
        xy = uv[:, None] * d[None, :, None]
        rays = torch.cat((xy, d[None, :, None].expand(len(uv), -1, 1),
                          torch.ones_like(xy[..., :1])), -1)
        xyz = torch.einsum('pij,pdj->pdi', inverse, rays)[..., :3]
        bounds = self.position_range
        xyz = ((xyz - bounds[:3]) / (bounds[3:] - bounds[:3])).clamp(1e-5, 1-1e-5)
        return self.pv_pe(torch.logit(xyz).flatten(1))

    def forward(self, queries, state, bev, pv_levels, meta):
        # M1 has no query change, no detach, and no additional parameters.
        if self.disabled or len(queries) == 0:
            return queries
        assert len(queries) == len(state.maps)
        query = queries.detach()
        if self.use_category:
            query = query + self.category(state.labels.long())
        if not self.order:
            return query
        xy_grid = metric_grid(state.maps.shape[-2:], self.roi, queries.device)
        points = []
        for mask in state.maps.detach()[:, 0]:
            rc = self._limit(torch.nonzero(mask > self.threshold))
            xy = xy_grid[rc[:,0], rc[:,1]]
            points.append(torch.cat((xy, xy.new_full((len(xy), 1), self.z)), -1))
        for branch in self.order:
            tokens = []
            for xyz in points:
                if not len(xyz):
                    tokens.append(query.new_empty((0, query.shape[-1])))
                    continue
                if branch == 'bev':
                    grid = torch.stack((2*xyz[:,0]/self.roi[0], -2*xyz[:,1]/self.roi[1]), -1)
                    feat = F.grid_sample(bev[None], grid[None,None], align_corners=False)[0,:,0].T
                    tokens.append(self.bev_proj(feat) + self.bev_pe(grid))
                else:
                    grid, valid, uv, matrices = project_points(xyz, meta, query.device)
                    ids = self._limit(torch.nonzero(valid))
                    if not len(ids):
                        tokens.append(query.new_empty((0, query.shape[-1])))
                        continue
                    cam, pt = ids[:,0], ids[:,1]
                    inverse = torch.linalg.inv(matrices)[cam]
                    pe = self._position3d(uv[cam,pt], inverse)
                    level_tokens = []
                    for proj, level in zip(self.pv_proj, self.pv_levels):
                        sampled = F.grid_sample(pv_levels[level], grid[:,None], align_corners=False)[:,:,0]
                        feat = sampled[cam,:,pt]
                        level_tokens.append(proj(feat) + pe)
                    tokens.append(self._limit(torch.cat(level_tokens)))
            query = self.branches[branch](query, tokens)
        return query
