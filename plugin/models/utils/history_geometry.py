"""BEV history: +y at top, pixel centers, align_corners=False."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pose_matrix(meta, device):
    pose = torch.eye(4, dtype=torch.float64, device=device)
    pose[:3, :3] = torch.as_tensor(meta['ego2global_rotation'], device=device)
    pose[:3, 3] = torch.as_tensor(meta['ego2global_translation'], device=device)
    return pose


def metric_grid(hw, roi, device):
    h, w = hw
    x = (torch.arange(w, device=device, dtype=torch.float32) + .5) * roi[0] / w - roi[0] / 2
    y = roi[1] / 2 - (torch.arange(h, device=device, dtype=torch.float32) + .5) * roi[1] / h
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), -1)


def _previous_to_current(source_pose, current_pose, device, prev2curr=None):
    """Use the native temporal transform verbatim when it is supplied.

    ``prev2curr`` maps previous ego coordinates into current ego coordinates.
    It may contain the native training pose noise.  This module never samples
    noise itself; raster maps, U and reference vectors must share one transform.
    """
    if prev2curr is None:
        transform = torch.linalg.solve(current_pose.double(), source_pose.double())
    else:
        transform = torch.as_tensor(prev2curr, device=device, dtype=torch.float64)
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError('prev2curr must be one finite [4,4] native transform')
    return transform.to(device=device)


def warp_history(tensor, source_pose, current_pose, roi, fill=0., prev2curr=None):
    if len(tensor) == 0:
        return tensor.clone()
    grid = metric_grid(tensor.shape[-2:], roi, tensor.device)
    xyz1 = torch.cat((grid, torch.zeros_like(grid[..., :1]), torch.ones_like(grid[..., :1])), -1)
    # grid_sample needs the pullback (current pixel -> previous coordinates).
    forward = _previous_to_current(source_pose, current_pose, tensor.device, prev2curr)
    transform = torch.linalg.inv(forward).float()
    source = xyz1 @ transform.T
    sample_grid = torch.stack((2 * source[..., 0] / roi[0], -2 * source[..., 1] / roi[1]), -1)
    sample_grid = sample_grid[None].expand(len(tensor), -1, -1, -1)
    return F.grid_sample(tensor.float() - fill, sample_grid, align_corners=False, padding_mode='zeros') + fill


def transform_boxes(boxes, source_pose, current_pose, roi, prev2curr=None):
    if len(boxes) == 0:
        return boxes.clone()
    points = boxes.detach().reshape(len(boxes), -1, 2).clone()
    scale = points.new_tensor(roi)
    points = (points - .5) * scale
    xyzh = torch.cat((points, torch.zeros_like(points[..., :1]), torch.ones_like(points[..., :1])), -1)
    transform = _previous_to_current(source_pose, current_pose, boxes.device, prev2curr).float()
    points = (xyzh @ transform.T)[..., :2] / scale + .5
    # Keep out-of-ROI coordinates out of the ROI. Clamping would create border
    # segments that were never predicted; rasterization already clips segments.
    return points.reshape_as(boxes)


@torch.no_grad()
def rasterize_instances(boxes, labels, hw, roi, line_width_px=1, polygon_fill=False):
    h, w = hw
    masks = np.zeros((len(boxes), 1, h, w), dtype=np.float32)
    coords = boxes.detach().float().reshape(len(boxes), -1, 2).cpu().numpy() if len(boxes) else []
    for i, (points, label) in enumerate(zip(coords, labels.detach().cpu().tolist())):
        if not np.isfinite(points).all():
            continue
        # Clip original segments via OpenCV; never clamp all vertices to ROI.
        pixel = np.stack((points[:, 0] * w - .5, (1 - points[:, 1]) * h - .5), -1)
        pixel = np.round(pixel).astype(np.int32)
        if label == 0 and polygon_fill:
            cv2.fillPoly(masks[i, 0], [pixel], 1.)
        else:
            cv2.polylines(masks[i, 0], [pixel], label == 0, 1., line_width_px)
    return torch.from_numpy(masks).to(boxes.device)


def project_points(points, meta, device):
    matrices = torch.as_tensor(np.asarray(meta['ego2img']), dtype=torch.float32, device=device)
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), -1)
    uvz = torch.einsum('cij,pj->cpi', matrices, homogeneous)
    depth = uvz[..., 2]
    uv = uvz[..., :2] / depth.clamp_min(1e-5)[..., None]
    padded = torch.tensor([s[:2] for s in meta['img_shape']], device=device, dtype=torch.float32)
    shapes = torch.tensor([s[:2] for s in meta.get('valid_img_shape', meta['img_shape'])], device=device, dtype=torch.float32)
    valid = torch.isfinite(uvz).all(-1) & (depth > 1e-5) & (uv[..., 0] >= 0) & (uv[..., 1] >= 0)
    valid &= (uv[..., 0] < shapes[:, None, 1]) & (uv[..., 1] < shapes[:, None, 0])
    return 2 * uv / padded[:, None, [1, 0]] - 1, valid, uv, matrices
