"""Guide-aligned explicit instance history on the native MapTracker loops.

History selection is observed at the existing native selection sites.  No GT
geometry or labels enter the explicit prior.  Native thresholds, BEV history,
vector memory and motion losses are independently configurable.
"""
import json
from pathlib import Path
import torch
from torch import nn

from .MapTracker import MapTracker
from .base_mapper import MAPPERS
from ..utils.history_geometry import pose_matrix
from ..utils.instance_history import update_selected
from ..utils.reliable_map_update import ReliableMapUpdate
from ..utils.map_trajectory_prior import MapTrajectoryPrior


class IdentityMotion(nn.Module):
    """The native motion call takes a feature and a pose condition."""

    def forward(self, features, pose_condition):
        return features


@MAPPERS.register_module()
class HisTrackMapMainline(MapTracker):
    def __init__(self, history_cfg=None, rmu_cfg=None, mtp_cfg=None,
                 prior_enabled=True, track_transform_source='native',
                 use_motion_mlp=True, transform_loss_enabled=True,
                 latent_memory_motion='native', tau_det=None, tau_track=.5,
                 retired_enabled=False, retired_max_queries=20, **kwargs):
        if track_transform_source not in ('native', 'state_pose'):
            raise ValueError('track_transform_source must be native or state_pose')
        if latent_memory_motion not in ('native', 'identity'):
            raise ValueError('latent_memory_motion must be native or identity')
        if not use_motion_mlp and transform_loss_enabled:
            raise ValueError('Disable transformation losses before removing MotionMLP')
        if not use_motion_mlp and kwargs.get('use_memory') and latent_memory_motion != 'identity':
            raise ValueError('Latent memory without MotionMLP requires explicit identity propagation')
        if retired_enabled and not prior_enabled:
            raise ValueError('Retired priors require the MTP branch')
        super().__init__(**kwargs)
        self.prior_enabled = bool(prior_enabled)
        self.track_transform_source = track_transform_source
        self.use_motion_mlp = bool(use_motion_mlp)
        self.transform_loss_enabled = bool(transform_loss_enabled)
        self.latent_memory_motion = latent_memory_motion
        self.tau_det, self.tau_track = tau_det, float(tau_track)
        self.retired_enabled = bool(retired_enabled)
        self.retired_max_queries = int(retired_max_queries)
        self.history_cfg = dict(map_hw=(self.bev_h, self.bev_w), line_width_px=1,
                                polygon_fill=False, valid_threshold=.2)
        self.history_cfg.update(history_cfg or {})
        self.rmu = ReliableMapUpdate(**(rmu_cfg or {}))
        options = dict(roi=self.roi_size, valid_threshold=self.history_cfg['valid_threshold'])
        options.update(mtp_cfg or {})
        if not self.prior_enabled:
            options['mode'] = 'disabled'
        # New modules must not change the baseline's augmentation/random state.
        with torch.random.fork_rng(devices=[]):
            self.mtp = MapTrajectoryPrior(**options)
        self.identity_motion = IdentityMotion()
        if not self.use_motion_mlp:
            self.query_propagate = IdentityMotion()
        self.head.temporal_selection_observer = self._on_inference_selection
        if self.retired_enabled:
            self.head.detection_prior_injector = self._inject_retired_priors
        self.reset_explicit_history(0)

    def reset_explicit_history(self, batch_size):
        self.explicit_history = [None] * int(batch_size)
        self.retired_history = [None] * int(batch_size)
        self.explicit_next_ids = [0] * int(batch_size)
        self._history_transforms = {}
        self._history_source_poses = {}
        self._clip_diagnostics = []
        self._retired_injected = {}
        self._frame_metas = None
        self._frame_bev = None
        self._frame_pv = None

    def forward_train(self, *args, **kwargs):
        # The original method owns its complete temporal loop and loss sum.
        self.reset_explicit_history(0)
        try:
            result = super().forward_train(*args, **kwargs)
            if self._clip_diagnostics:
                loss, log_vars, num_samples = result
                for key in ('instances', 'support_pixels_mean', 'empty_support_fraction',
                            'uncertainty_mean', 'gain_mean', 'gain_min', 'gain_max',
                            'applied_gain_mean', 'applied_gain_min', 'applied_gain_max'):
                    values = [row[key] for row in self._clip_diagnostics if row[key] is not None]
                    if values:
                        log_vars['history_' + key] = sum(values) / len(values)
                result = loss, log_vars, num_samples
            return result
        finally:
            # Avoid retaining image graphs or states across independent clips.
            self.reset_explicit_history(0)

    def forward_test(self, *args, **kwargs):
        seq_info = kwargs.get('seq_info')
        if seq_info is None and len(args) > 3:
            seq_info = args[3]
        if seq_info is not None and int(seq_info[0][1]) == 0:
            self.reset_explicit_history(1)
        try:
            return super().forward_test(*args, **kwargs)
        finally:
            self._frame_bev = self._frame_pv = self._frame_metas = None
            self._history_transforms = {}

    def _on_frame_features(self, bev, pv_levels, metas):
        if not self.explicit_history:
            self.reset_explicit_history(len(metas))
        if len(metas) != len(self.explicit_history):
            raise AssertionError('Explicit history batch size changed inside a clip/scene')
        self._frame_bev, self._frame_pv, self._frame_metas = bev, pv_levels, metas
        self._history_transforms = {}
        self._history_source_poses = {}
        self._retired_injected = {}

    def _memory_motion_module(self):
        if self.latent_memory_motion == 'identity':
            return self.identity_motion
        return self.query_propagate

    def _adapt_track_transform(self, batch_index, meta, transform, nominal_transform, target):
        state = self.explicit_history[batch_index]
        if state is None:
            raise AssertionError('Native tracks have no corresponding explicit state')
        if state.scene != meta['scene_name']:
            raise AssertionError('Explicit history crossed a scene boundary')
        if len(state.ids) != len(target['track_query_hs_embeds']):
            raise AssertionError('Track queries and explicit states have different lengths')
        current_pose = pose_matrix(meta, state.maps.device)
        state_nominal = torch.linalg.solve(current_pose.double(), state.pose.double())
        self._history_source_poses[batch_index] = state.pose.detach().cpu().tolist()
        # Keep the native single random perturbation.  The source frame may
        # differ from the last selected BEV memory at inference time.
        noise_delta = transform.double() @ torch.linalg.inv(nominal_transform.double())
        state_transform = noise_delta @ state_nominal
        self._history_transforms[batch_index] = state_transform
        if self.track_transform_source == 'native':
            return transform
        # Preserve the native math exactly when both sources already coincide.
        if torch.equal(state_nominal, nominal_transform.double()):
            return transform
        return state_transform

    def _on_temporal_transform(self, batch_index, meta, transform, target):
        state = self.explicit_history[batch_index]
        history_transform = self._history_transforms[batch_index]
        if self.track_transform_source == 'state_pose':
            history_transform = transform
        self.explicit_history[batch_index] = state.align(
            meta, self.roi_size, self.rmu, prev2curr=history_transform)
        retired = self.retired_history[batch_index]
        if self.retired_enabled and retired is not None:
            # Retired and active states were committed together at the previous
            # frame, so this reuses exactly the active branch's single noise.
            if retired.frame != state.frame or not torch.equal(retired.pose, state.pose):
                raise AssertionError('Retired prior is not from the preceding committed frame')
            self.retired_history[batch_index] = retired.align(
                meta, self.roi_size, self.rmu, prev2curr=history_transform)

    def _inject_retired_priors(self, queries, references):
        """Consume one-frame priors in fixed detection slots; IDs stay native."""
        if not self.retired_enabled:
            return queries, references
        if queries.shape[1] != self.head.num_queries:
            raise AssertionError('Retired priors must not change the detection-query count')
        result, ref_result = queries.clone(), references.clone()
        for b_i, retired in enumerate(self.retired_history):
            self.retired_history[b_i] = None  # consume once, including empty/invalid priors
            self._retired_injected[b_i] = 0
            if retired is None or not len(retired.ids):
                continue
            if retired.frame != int(self._frame_metas[b_i]['local_idx']):
                raise AssertionError('Retired prior was not aligned to the current frame')
            active = self.explicit_history[b_i]
            threshold = self.history_cfg['valid_threshold']
            masks = retired.maps[:, 0] > threshold
            valid = masks.flatten(1).any(1) & (retired.scores >= self.history_cfg.get('retired_min_score', 0.))
            # Duplicate suppression only compares the same predicted class.
            if active is not None and len(active.ids):
                active_masks = active.maps[:, 0] > threshold
                for i in range(len(retired.ids)):
                    same = active.labels == retired.labels[i]
                    if bool(same.any()):
                        overlap = (active_masks[same] & masks[i]).flatten(1).sum(1).float()
                        union = (active_masks[same] | masks[i]).flatten(1).sum(1).clamp_min(1).float()
                        if bool((overlap / union >= self.history_cfg.get('retired_dedup_iou', .5)).any()):
                            valid[i] = False
            candidates = torch.where(valid)[0]
            if not len(candidates):
                continue
            ordered = torch.argsort(retired.scores[candidates], descending=True, stable=True)
            indices = candidates[ordered[:min(self.retired_max_queries, self.head.num_queries)]]
            prior = retired.select(indices)
            base = queries[b_i, :len(indices)]
            enhanced = self.mtp(base, prior, self._frame_bev[b_i],
                                [level[b_i] for level in self._frame_pv], self._frame_metas[b_i])
            # MTP detaches its history input.  Restore the native detection
            # embedding's identity gradient while keeping MTP parameters live.
            result[b_i, :len(indices)] = base + enhanced - base.detach()
            ref_result[b_i, :len(indices)] = prior.boxes.clamp(0., 1.).reshape(
                len(indices), self.head.num_points, 2)
            self._retired_injected[b_i] = len(indices)
        return result, ref_result

    def temporal_propagate(self, curr_bev_feats, img_metas,
                           all_history_curr2prev, all_history_prev2curr,
                           use_memory, track_query_info=None, timestep=None,
                           get_trans_loss=False):
        result = super().temporal_propagate(
            curr_bev_feats, img_metas, all_history_curr2prev, all_history_prev2curr,
            use_memory, track_query_info, timestep=timestep,
            get_trans_loss=get_trans_loss and self.transform_loss_enabled)
        if self.prior_enabled and track_query_info is not None and not self.skip_vector_head:
            for b_i, target in enumerate(track_query_info):
                state = self.explicit_history[b_i]
                if state is None:
                    raise AssertionError('MTP was called without selected prediction history')
                target['track_query_hs_embeds'] = self.mtp(
                    target['track_query_hs_embeds'], state,
                    curr_bev_feats[b_i], [level[b_i] for level in self._frame_pv],
                    img_metas[b_i])
        # Empty dict prevents the original loop from accumulating or logging
        # removed f_trans/b_trans paths; no auxiliary head was called above.
        if get_trans_loss and not self.transform_loss_enabled:
            return {}
        return result

    @torch.no_grad()
    def _on_training_selection(self, batch_index, target, predictions, indices, meta):
        self.rmu.last_gain = None
        state, retired, next_id = update_selected(
            self.explicit_history[batch_index], indices,
            predictions['lines'][batch_index], predictions['scores'][batch_index],
            meta, self.explicit_next_ids[batch_index], self.history_cfg,
            self.roi_size, self.rmu)
        if not torch.equal(state.labels, target['track_query_labels']):
            raise AssertionError('CE prediction labels diverged from native selection')
        if not torch.equal(state.scores, target['track_query_scores']):
            raise AssertionError('Prediction confidence diverged from native selection')
        self.explicit_history[batch_index] = state
        self.retired_history[batch_index] = retired
        self.explicit_next_ids[batch_index] = next_id
        self._record_history_diagnostics(batch_index, state, meta)

    @torch.no_grad()
    def _on_inference_selection(self, predictions, prop_info):
        if len(self.explicit_history) != 1 or self._frame_metas is None:
            raise AssertionError('Inference selection is outside its native frame')
        self.rmu.last_gain = None
        state, retired, next_id = update_selected(
            self.explicit_history[0], prop_info['selected_indices'],
            predictions['lines'][0], predictions['scores'][0], self._frame_metas[0],
            self.explicit_next_ids[0], self.history_cfg, self.roi_size, self.rmu,
            global_ids=prop_info['global_ids'])
        if not torch.equal(state.ids.cpu(), prop_info['global_ids'].cpu()):
            raise AssertionError('Explicit history IDs diverged from native lifecycle')
        self.explicit_history[0], self.retired_history[0] = state, retired
        self.explicit_next_ids[0] = next_id
        self._record_history_diagnostics(0, state, self._frame_metas[0])

    @torch.no_grad()
    def _record_history_diagnostics(self, batch_index, state, meta):
        count = len(state.ids)
        support = (state.maps > self.history_cfg['valid_threshold']).flatten(1).sum(1)
        gain = self.rmu.last_gain
        gain_diagnostics = self.rmu.last_diagnostics or {}
        applied_gain = gain_diagnostics.get('applied_gain')
        row = dict(
            schema='histrackmap.history-frame.v1', training=bool(self.training),
            scene=state.scene, frame=int(state.frame), batch_index=int(batch_index),
            instances=count, ids=state.ids.detach().cpu().tolist(),
            update_mode=self.rmu.mode, transform_source=self.track_transform_source,
            gain_semantics='candidate_K_equals_applied_only_for_RMU',
            retired_injected=self._retired_injected.get(batch_index, 0),
            source_pose=self._history_source_poses.get(batch_index),
            current_pose=state.pose.detach().cpu().tolist(),
            support_pixels=support.detach().cpu().tolist(),
            support_pixels_mean=float(support.float().mean()) if count else 0.,
            empty_support_fraction=float((support == 0).float().mean()) if count else 0.,
            uncertainty_mean=float(state.uncertainty.mean()) if state.uncertainty.numel() else None,
            uncertainty_min=float(state.uncertainty.min()) if state.uncertainty.numel() else None,
            uncertainty_max=float(state.uncertainty.max()) if state.uncertainty.numel() else None,
            gain_mean=float(gain.mean()) if gain is not None and gain.numel() else None,
            gain_min=float(gain.min()) if gain is not None and gain.numel() else None,
            gain_max=float(gain.max()) if gain is not None and gain.numel() else None,
            applied_gain_mean=float(applied_gain.mean()) if applied_gain is not None and applied_gain.numel() else None,
            applied_gain_min=float(applied_gain.min()) if applied_gain is not None and applied_gain.numel() else None,
            applied_gain_max=float(applied_gain.max()) if applied_gain is not None and applied_gain.numel() else None)
        if self.training:
            self._clip_diagnostics.append(row)
        self.latest_history_diagnostics = row
        destination = self.history_cfg.get('diagnostics_path')
        if destination:
            # DDP ranks have independent states and must never interleave JSON.
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            path = Path(destination)
            path = path.with_name(path.stem + '.rank' + str(rank) + path.suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def _inference_detection_threshold(self, first_frame):
        if self.tau_det is None:
            return super()._inference_detection_threshold(first_frame)
        return float(self.tau_det)

    def _inference_track_threshold(self):
        return self.tau_track
