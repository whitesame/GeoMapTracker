"""MapTracker plus one explicit-geometry BEV query residual.

All sequence loops, native vector memory, MotionMLP, transformation losses,
pose perturbations, track identities and thresholds remain inherited.
"""

from .MapTracker import MapTracker
from .base_mapper import MAPPERS
from ..utils.geometry_query_residual import GeometryQueryResidual


@MAPPERS.register_module()
class HisTrackMapIncremental(MapTracker):
    def __init__(self, geometry_prior_cfg=None, **kwargs):
        super().__init__(**kwargs)
        options = dict(in_channels=256, embed_dims=self.head.embed_dims,
                       max_delta=0.1)
        options.update(geometry_prior_cfg or {})
        self.geometry_prior = GeometryQueryResidual(**options)

    def temporal_propagate(self, curr_bev_feats, img_metas,
                           all_history_curr2prev, all_history_prev2curr,
                           use_memory, track_query_info=None, timestep=None,
                           get_trans_loss=False):
        # Complete the baseline path first. Its auxiliary loss tensors retain
        # their original graph; replace only the detached decoder input query.
        result = super().temporal_propagate(
            curr_bev_feats, img_metas, all_history_curr2prev,
            all_history_prev2curr, use_memory,
            track_query_info=track_query_info, timestep=timestep,
            get_trans_loss=get_trans_loss)
        if track_query_info is not None and not self.skip_vector_head:
            for batch_index, target in enumerate(track_query_info):
                target['track_query_hs_embeds'] = self.geometry_prior(
                    target['track_query_hs_embeds'],
                    target['trans_track_query_boxes'],
                    curr_bev_feats[batch_index])
        return result
