"""CPU geometry, state-order, RMU and real MTP gradient checks."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = '_mainline_history_utils'
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / 'plugin/models/utils')]
    sys.modules[PACKAGE] = package


def load(name):
    full = PACKAGE + '.' + name
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, ROOT / 'plugin/models/utils' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


geometry = load('history_geometry')
history = load('instance_history')
ReliableMapUpdate = load('reliable_map_update').ReliableMapUpdate
MapTrajectoryPrior = load('map_trajectory_prior').MapTrajectoryPrior


def meta(frame=0, scene='scene-a'):
    camera = torch.eye(4)
    camera[0, 2] = camera[1, 2] = 2
    return dict(scene_name=scene, local_idx=frame,
                ego2global_rotation=torch.eye(3), ego2global_translation=[0, 0, 0],
                ego2img=torch.stack((camera, camera)).numpy(),
                img_shape=[(4, 4, 3)] * 2, valid_img_shape=[(4, 4, 3)] * 2)


def state(n=3, hw=(4, 4)):
    maps = torch.zeros(n, 1, *hw)
    if n:
        maps[:, :, 1, 1] = .9
    return history.HistoryState('scene-a', 0, torch.eye(4, dtype=torch.float64),
                                torch.arange(n) + 10, maps, torch.ones_like(maps),
                                torch.arange(n) % 3, torch.full((n,), .9),
                                torch.tensor([[.25, .75, .75, .25]]).expand(n, -1).clone())


def prior(**kwargs):
    return MapTrajectoryPrior(roi=(4, 4), embed_dims=8, in_channels=3, num_heads=2,
                              depth_num=4, depth_start=1, depth_end=4,
                              ground_z=1, max_samples_per_instance=16, **kwargs)


def test_identity_warp_has_no_half_pixel_or_y_flip_error():
    raster = torch.arange(30.).reshape(1, 1, 5, 6)
    eye = torch.eye(4, dtype=torch.float64)
    torch.testing.assert_close(geometry.warp_history(raster, eye, eye, (6, 5)), raster)


def test_native_translation_is_shared_by_raster_uncertainty_and_boxes():
    old = state(1, (5, 5))
    old.maps.zero_(); old.maps[0, 0, 2, 2] = 1
    old.boxes = torch.tensor([[.5, .5]])
    transform = torch.eye(4, dtype=torch.float64); transform[0, 3] = 1
    # The metadata has no movement: the explicit native (possibly noisy) T
    # must still move raster and vector by one metre in exactly one direction.
    aligned = old.align(meta(1), (5, 5), ReliableMapUpdate(), prev2curr=transform)
    assert aligned.maps[0, 0, 2, 3] == 1
    torch.testing.assert_close(aligned.boxes, torch.tensor([[.7, .5]]))
    assert aligned.uncertainty[0, 0, 2, 0] >= 10
    assert torch.equal(aligned.pose, old.pose)


def test_native_ninety_degree_rotation_and_inverse_return_original():
    raster = torch.arange(25.).reshape(1, 1, 5, 5)
    transform = torch.eye(4, dtype=torch.float64)
    transform[:2, :2] = torch.tensor([[0., -1.], [1., 0.]])
    rotated = geometry.warp_history(raster, None, None, (5, 5), prev2curr=transform)
    torch.testing.assert_close(rotated, torch.rot90(raster, 1, (-2, -1)))
    back = geometry.warp_history(rotated, None, None, (5, 5), prev2curr=torch.linalg.inv(transform))
    torch.testing.assert_close(back, raster)


def test_true_pose_and_explicit_transform_are_equivalent():
    raster = torch.arange(25.).reshape(1, 1, 5, 5)
    source = torch.eye(4, dtype=torch.float64)
    current = source.clone(); current[1, 3] = 1
    inferred = geometry.warp_history(raster, source, current, (5, 5))
    explicit = geometry.warp_history(raster, None, None, (5, 5),
                                     prev2curr=torch.linalg.solve(current, source))
    torch.testing.assert_close(inferred, explicit)


def test_no_independent_pose_noise_or_rng_consumption():
    old = state(2)
    rng = torch.random.get_rng_state().clone()
    old.align(meta(1), (4, 4), ReliableMapUpdate(), prev2curr=torch.eye(4))
    assert torch.equal(rng, torch.random.get_rng_state())


def test_outside_geometry_is_not_clamped_into_false_border_support():
    boxes = torch.tensor([[.8, .5, .9, .5]])
    transform = torch.eye(4); transform[0, 3] = 10
    moved = geometry.transform_boxes(boxes, None, None, (4, 4), prev2curr=transform)
    assert (moved.reshape(-1, 2)[:, 0] > 1).all()
    raster = geometry.rasterize_instances(moved, torch.tensor([1]), (8, 8), (4, 4))
    assert raster.count_nonzero() == 0


@pytest.mark.parametrize('which', ['scene', 'time'])
def test_history_refuses_cross_scene_or_backward_frame(which):
    old = state(1)
    wrong = meta(1, 'other') if which == 'scene' else meta(-1)
    with pytest.raises(AssertionError):
        old.align(wrong, (4, 4), ReliableMapUpdate())


def test_selection_delete_middle_then_insert_false_positive_uses_prediction_labels():
    old = state()
    boxes = torch.tensor([[.2, .2, .3, .3], [.2, .7, .3, .8],
                          [.5, .3, .7, .4], [.6, .6, .8, .8]], requires_grad=True)
    logits = torch.tensor([[-3., 5., -2.], [-3., -2., 4.],
                           [6., -3., -4.], [-3., 7., -2.]], requires_grad=True)
    selected, retired, next_id = history.update_selected(
        old, torch.tensor([2, 0, 3]), boxes, logits, meta(1), 13,
        dict(map_hw=(8, 8), line_width_px=1, polygon_fill=False),
        (4, 4), ReliableMapUpdate(mode='overwrite'))
    assert selected.ids.tolist() == [12, 10, 13]
    assert selected.labels.tolist() == [0, 1, 1]
    assert retired.ids.tolist() == [11]
    assert next_id == 14
    torch.testing.assert_close(selected.boxes, boxes[[2, 0, 3]].detach())
    for field in ('boxes', 'maps', 'uncertainty', 'labels', 'scores'):
        assert not getattr(selected, field).requires_grad


def test_native_selected_global_ids_preserve_fp_identity():
    boxes = torch.full((4, 4), .5)
    logits = torch.zeros(4, 3)
    old = state()
    selected, _, next_id = history.update_selected(
        old, torch.tensor([2, 3]), boxes, logits, meta(1), 13,
        dict(map_hw=(4, 4), line_width_px=1, polygon_fill=False),
        (4, 4), ReliableMapUpdate(), global_ids=torch.tensor([12, 99]))
    assert selected.ids.tolist() == [12, 99] and next_id == 100
    with pytest.raises(ValueError, match='misaligned'):
        history.update_selected(old, torch.tensor([2]), boxes, logits, meta(1), 13,
                                dict(map_hw=(4, 4), line_width_px=1, polygon_fill=False),
                                (4, 4), ReliableMapUpdate(), global_ids=torch.tensor([11]))


def test_repeated_empty_frames_keep_valid_empty_state_without_phantom_ids():
    empty, retired, next_id = history.update_selected(
        None, torch.empty(0, dtype=torch.long), torch.empty(0, 4), torch.empty(0, 3),
        meta(), 0, dict(map_hw=(4, 4), line_width_px=1, polygon_fill=False),
        (4, 4), ReliableMapUpdate())
    assert retired is None and next_id == 0 and empty.maps.shape == (0, 1, 4, 4)
    again = empty.align(meta(1), (4, 4), ReliableMapUpdate(), prev2curr=torch.eye(4))
    assert again.ids.numel() == 0 and again.maps.numel() == 0


def test_predicted_category_matches_native_raw_logit_argmax_when_sigmoid_saturates():
    selected, _, _ = history.update_selected(
        None, torch.tensor([0]), torch.full((1, 4), .5), torch.tensor([[50., 60., 40.]]),
        meta(), 0, dict(map_hw=(4, 4), line_width_px=1, polygon_fill=False),
        (4, 4), ReliableMapUpdate())
    assert selected.labels.tolist() == [1]


def test_full_grid_rmu_matches_published_candidate_formula_even_off_support():
    updater = ReliableMapUpdate(a=1, b=1, process_noise=.05)
    maps = torch.tensor([[[[.9, .8]]]])
    uncertainty = torch.tensor([[[[1., 10.]]]])
    raster = torch.tensor([[[[1., 0.]]]])
    score = torch.tensor([.5])
    actual, after_u = updater.update(maps, uncertainty, raster, score)
    variance = 1 / (.5 + 1e-6)
    gain = uncertainty / (uncertainty + variance)
    torch.testing.assert_close(actual, (1 - gain) * maps + gain * raster * .5)
    torch.testing.assert_close(after_u, (1 - gain) * uncertainty)
    assert actual[0, 0, 0, 1] < maps[0, 0, 0, 1]  # full-grid is not masked
    assert updater.last_diagnostics['scope'] == 'full_grid'
    torch.testing.assert_close(updater.last_diagnostics['kalman_gain'], gain)


@pytest.mark.parametrize('mode', ['overwrite', 'decay', 'rmu'])
def test_update_modes_and_u_k_diagnostics_are_distinct(mode):
    updater = ReliableMapUpdate(mode=mode, decay=.8)
    maps = torch.full((1, 1, 2, 2), .8)
    uncertainty = torch.ones_like(maps)
    raster = torch.ones_like(maps)
    actual, actual_u = updater.update(maps, uncertainty, raster, torch.tensor([.5]))
    if mode == 'overwrite':
        torch.testing.assert_close(actual, torch.full_like(maps, .5))
    elif mode == 'decay':
        torch.testing.assert_close(actual, .8 * maps + .2 * .5)
        assert torch.equal(actual_u, uncertainty)
    else:
        assert ((actual > .5) & (actual < .8)).all()
    assert updater.last_diagnostics['mode'] == mode
    assert not updater.last_gain.requires_grad


def test_extreme_prediction_scores_keep_u_and_gain_finite():
    updater = ReliableMapUpdate()
    raster = torch.tensor([[[[1., 0.]]], [[[1., 0.]]]])
    maps, uncertainty = updater.initialize(raster, torch.tensor([0., 1.]))
    assert uncertainty[0, 0, 0, 0] == 100
    assert (uncertainty[:, 0, 0, 1] == 10).all()
    result, u = updater.update(maps, uncertainty, raster, torch.tensor([1., 0.]))
    assert torch.isfinite(result).all() and torch.isfinite(u).all()
    assert ((updater.last_gain >= 0) & (updater.last_gain <= 1)).all()


def test_disabled_mtp_is_exact_tensor_identity_without_parameters_or_detach():
    module = prior(mode='disabled')
    query = torch.randn(3, 8, requires_grad=True)
    assert module(query, None, None, None, None) is query
    assert not list(module.parameters())


def test_ce_only_uses_prediction_category_without_bev_pv_parameters():
    module = prior(mode='ce')
    query = torch.randn(3, 8, requires_grad=True)
    old = state()
    actual = module(query, old, None, None, None)
    torch.testing.assert_close(actual, query.detach() + module.category(old.labels))
    assert list(dict(module.named_parameters())) == ['category.weight']
    actual.sum().backward()
    assert query.grad is None and module.category.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('mode', ['bev', 'pv', 'bev_pv'])
def test_active_mtp_branches_have_finite_current_feature_and_parameter_gradients(mode):
    module = prior(mode=mode)
    old = state()
    old.maps[1] = 0  # one all-masked instance, with other active instances
    query = torch.randn(3, 8, requires_grad=True)
    bev = torch.randn(3, 4, 4, requires_grad=True)
    pv = torch.randn(2, 3, 4, 4, requires_grad=True)
    actual = module(query, old, bev, [pv], meta())
    assert torch.isfinite(actual).all()
    weights = torch.arange(1., 9.)[None]
    (actual * weights).sum().backward()
    assert query.grad is None
    assert module.category.weight.grad.abs().sum() > 0
    for name in module.order:
        projection = module.bev_proj if name == 'bev' else module.pv_proj[0]
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0
    if 'bev' in module.order:
        assert bev.grad.abs().sum() > 0
    if 'pv' in module.order:
        assert pv.grad.abs().sum() > 0 and module.pv_pe[0].weight.grad.abs().sum() > 0


def test_all_empty_support_safely_keeps_category_query_for_both_branches():
    module = prior(mode='bev_pv')
    old = state(); old.maps.zero_()
    query = torch.randn(3, 8)
    actual = module(query, old, torch.randn(3, 4, 4), [torch.randn(2, 3, 4, 4)], meta())
    torch.testing.assert_close(actual, query + module.category(old.labels))
    assert torch.isfinite(actual).all()


def test_pv_all_behind_camera_is_safe_and_has_no_feature_dependency():
    module = prior(mode='pv')
    bad = meta()
    bad['ego2img'][:, 2, 2] = -1
    old = state(); query = torch.randn(3, 8)
    pv = torch.randn(2, 3, 4, 4, requires_grad=True)
    actual = module(query, old, None, [pv], bad)
    torch.testing.assert_close(actual, query + module.category(old.labels))
    assert torch.isfinite(actual).all()


def test_pv_projection_uses_valid_shape_before_padding_and_positive_depth():
    camera = torch.eye(4)[None].numpy()
    metadata = dict(ego2img=camera, img_shape=[(8, 8, 3)], valid_img_shape=[(4, 4, 3)])
    xyz = torch.tensor([[2., 2., 1.], [6., 2., 1.], [1., 1., -1.]])
    grid, valid, uv, _ = geometry.project_points(xyz, metadata, torch.device('cpu'))
    assert valid.tolist() == [[True, False, False]]
    torch.testing.assert_close(grid[0, 0], torch.tensor([-.5, -.5]))
    torch.testing.assert_close(uv[0, 0], torch.tensor([2., 2.]))


def test_empty_queries_return_same_empty_tensor_for_every_mode():
    query = torch.empty(0, 8, requires_grad=True)
    for mode in ('disabled', 'ce', 'bev', 'pv', 'bev_pv'):
        assert prior(mode=mode)(query, state(0), None, None, None) is query


def test_two_branch_mode_preserves_explicit_reverse_order_ablation():
    assert prior(mode='bev_pv').order == ('pv', 'bev')
    assert prior(mode='bev_pv', order=('bev', 'pv')).order == ('bev', 'pv')
    with pytest.raises(ValueError, match='disagree'):
        prior(mode='bev', order=('pv',))
