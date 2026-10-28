"""Candidate full-grid RMU; parameters are explicit engineering assumptions."""
import torch


class ReliableMapUpdate:
    def __init__(self, a=1., b=1., process_noise=.05, unknown=10., eps=1e-6,
                 u_min=1e-4, u_max=100., r_min=1e-4, r_max=100., mode='rmu', decay=.8):
        assert mode in ('rmu', 'overwrite', 'decay')
        self.a, self.b, self.q, self.unknown = a, b, process_noise, unknown
        self.eps, self.u_min, self.u_max = eps, u_min, u_max
        self.r_min, self.r_max, self.mode, self.decay = r_min, r_max, mode, decay
        self.last_gain = None
        self.last_diagnostics = None
        if not 0 <= decay <= 1 or process_noise < 0:
            raise ValueError('decay must be in [0,1] and process_noise nonnegative')
        if min(a, b, eps, u_min, r_min) <= 0 or u_max < u_min or r_max < r_min:
            raise ValueError('Invalid full-grid uncertainty parameters')

    @torch.no_grad()
    def initialize(self, raster, scores):
        self.last_gain = None
        self.last_diagnostics = None
        if raster.ndim != 4 or raster.shape[1] != 1 or scores.shape != (len(raster),):
            raise ValueError('Expected instance maps [N,1,H,W] and scores [N]')
        if not torch.isfinite(raster).all() or not torch.isfinite(scores).all():
            raise ValueError('History observations must be finite')
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError('History scores must be prediction probabilities')
        score = scores.float()[:, None, None, None]
        variance = (self.a / (score + self.eps)).clamp(self.u_min, self.u_max)
        uncertainty = torch.where(raster > 0, variance, torch.full_like(raster, self.unknown))
        return raster.float() * score, uncertainty

    @torch.no_grad()
    def update(self, maps, uncertainty, raster, scores):
        obs, initialized_u = self.initialize(raster, scores)
        variance = (self.b / (scores.float()[:, None, None, None] + self.eps)).clamp(self.r_min, self.r_max)
        gain = uncertainty.float() / (uncertainty.float() + variance).clamp_min(self.eps)
        self.last_gain = gain.detach()
        self.last_diagnostics = {
            'mode': self.mode, 'scope': 'full_grid',
            'uncertainty_before': uncertainty.detach(),
            'observation_variance': variance.detach(),
            'kalman_gain': gain.detach(),
        }
        if self.mode == 'overwrite':
            updated, updated_u = obs, initialized_u
            self.last_diagnostics['applied_gain'] = torch.ones_like(gain)
        elif self.mode == 'decay':
            updated, updated_u = self.decay * maps + (1 - self.decay) * obs, uncertainty
            self.last_diagnostics['applied_gain'] = torch.full_like(gain, 1 - self.decay)
        else:
            updated = (1 - gain) * maps.float() + gain * obs
            updated_u = ((1 - gain) * uncertainty).clamp(self.u_min, self.u_max)
            self.last_diagnostics['applied_gain'] = gain.detach()
        self.last_diagnostics['uncertainty_after'] = updated_u.detach()
        return updated.detach(), updated_u.detach()
