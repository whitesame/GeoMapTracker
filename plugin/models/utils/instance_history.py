from dataclasses import dataclass
import torch
from .history_geometry import warp_history, transform_boxes, rasterize_instances, pose_matrix


@dataclass
class HistoryState:
    scene: str
    frame: int
    pose: torch.Tensor
    ids: torch.Tensor
    maps: torch.Tensor
    uncertainty: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor

    def select(self, indices):
        return HistoryState(self.scene, self.frame, self.pose, *[getattr(self, k)[indices].detach() for k in ('ids', 'maps', 'uncertainty', 'labels', 'scores', 'boxes')])

    @torch.no_grad()
    def align(self, meta, roi, rmu, prev2curr=None):
        assert self.scene == meta['scene_name'], 'History crossed scene boundary'
        pose = pose_matrix(meta, self.maps.device)
        frame = int(meta['local_idx'])
        assert frame >= self.frame, 'Non-monotonic history'
        maps = warp_history(self.maps, self.pose, pose, roi, prev2curr=prev2curr)
        uncertainty = warp_history(self.uncertainty, self.pose, pose, roi, rmu.unknown,
                                   prev2curr=prev2curr)
        if frame != self.frame:
            uncertainty = (uncertainty + rmu.q).clamp(rmu.u_min, rmu.u_max)
        boxes = transform_boxes(self.boxes, self.pose, pose, roi, prev2curr=prev2curr)
        return HistoryState(self.scene, frame, pose, self.ids, maps, uncertainty, self.labels, self.scores, boxes)


@torch.no_grad()
def update_selected(previous, indices, boxes, logits, meta, next_id, history_cfg, roi, rmu,
                    global_ids=None):
    """Same selected indices drive query order, parents, IDs and rasters, incl. FP."""
    indices = indices.to(boxes.device).long()
    if indices.ndim != 1 or len(indices.unique()) != len(indices):
        raise ValueError('History selection must be a unique one-dimensional index list')
    if len(indices) and (indices.min() < 0 or indices.max() >= len(boxes)):
        raise ValueError('History selection includes padding or unavailable predictions')
    if logits.ndim != 2 or len(logits) != len(boxes) or logits.shape[-1] != 3:
        raise ValueError('History expects raw three-class prediction logits')
    if previous is not None:
        if previous.scene != meta['scene_name']:
            raise ValueError('Reset history before changing scenes')
        if len(previous.ids.unique()) != len(previous.ids):
            raise ValueError('Previous native IDs are not unique')
    if global_ids is not None:
        global_ids = torch.as_tensor(global_ids, device=boxes.device, dtype=torch.long)
        if global_ids.shape != indices.shape or len(global_ids.unique()) != len(global_ids):
            raise ValueError('global_ids must be selected native IDs, in selection order')
    selected_boxes = boxes[indices].detach()
    # Match the native head: argmax raw logits first. Sigmoid can saturate two
    # unequal large logits to the same value and thereby change the class ID.
    scores, labels = logits[indices].detach().max(-1)
    scores = scores.sigmoid()
    raster = rasterize_instances(selected_boxes, labels, history_cfg['map_hw'], roi,
                                 history_cfg['line_width_px'], history_cfg['polygon_fill'])
    maps, uncertainty = rmu.initialize(raster, scores)
    ids = torch.empty(len(indices), dtype=torch.long, device=boxes.device)
    count = 0 if previous is None else len(previous.ids)
    inherited = indices < count
    if inherited.any():
        parents = indices[inherited]
        maps[inherited], uncertainty[inherited] = rmu.update(previous.maps[parents], previous.uncertainty[parents], raster[inherited], scores[inherited])
        ids[inherited] = previous.ids[parents]
    newborn = int((~inherited).sum())
    ids[~inherited] = torch.arange(next_id, next_id + newborn, device=boxes.device)
    if global_ids is not None:
        if inherited.any() and not torch.equal(global_ids[inherited], ids[inherited]):
            raise ValueError('Native query IDs and inherited history IDs are misaligned')
        ids = global_ids.detach().clone()
    retired = None
    if previous is not None:
        keep = torch.ones(count, dtype=torch.bool, device=boxes.device)
        keep[indices[inherited]] = False
        retired = previous.select(keep)
    state = HistoryState(meta['scene_name'], int(meta['local_idx']), pose_matrix(meta, boxes.device), ids,
                         maps.detach(), uncertainty.detach(), labels, scores, selected_boxes)
    next_id = max(next_id + newborn, int(ids.max()) + 1 if len(ids) else next_id)
    return state, retired, next_id
