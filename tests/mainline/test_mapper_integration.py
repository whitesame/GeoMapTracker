"""Exercise the real native mapper/head/motion CPU paths.

CUDA-only vector-bank I/O and image backbone execution are covered by the
queued two-GPU smoke, not simulated here. These tests build the actual model.
"""
import copy
import gc
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def factory():
    from mmcv import Config
    from mmdet3d.models import build_model
    import plugin  # noqa: F401
    config = Config.fromfile(str(ROOT / 'plugin/configs/maptracker/'
                                'nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py'))

    def build(native=False, mode='disabled', **changes):
        options = copy.deepcopy(config.model)
        # These tests compare controlled random initialization, without a download.
        options.backbone_cfg.img_backbone.pretrained = None
        if not native:
            options.type = 'HisTrackMapMainline'
            options.prior_enabled = mode != 'disabled'
            options.mtp_cfg = dict(mode=mode, depth_num=4, depth_start=1, depth_end=5,
                                   ground_z=1, max_samples_per_instance=8)
            options.rmu_cfg = dict(mode='overwrite')
        options.update(changes)
        torch.manual_seed(127)
        return build_model(options)
    return build


def metadata(frame=1, scene='cpu-scene'):
    camera = np.eye(4)
    camera[0, 2] = camera[1, 2] = 4
    return dict(scene_name=scene, local_idx=frame,
                ego2global_rotation=np.eye(3), ego2global_translation=np.zeros(3),
                ego2img=np.stack([camera] * 6), img_shape=[(8, 8, 3)] * 6,
                valid_img_shape=[(8, 8, 3)] * 6)


def install_history(model, boxes, frame=0, scene='cpu-scene'):
    from plugin.models.utils.instance_history import HistoryState
    n = len(boxes)
    masks = torch.zeros(n, 1, model.bev_h, model.bev_w)
    masks[:, :, model.bev_h // 2, model.bev_w // 2] = 1
    old = HistoryState(scene, frame, torch.eye(4, dtype=torch.float64),
                       torch.arange(n), masks, torch.ones_like(masks),
                       torch.arange(n) % 3, torch.full((n,), .9), boxes.detach())
    model.reset_explicit_history(1)
    model.explicit_history[0] = old
    model.explicit_next_ids[0] = n
    return old


def inputs():
    torch.manual_seed(17)
    bev = torch.randn(1, 256, 6, 8, requires_grad=True)
    pv = torch.randn(1, 6, 256, 4, 4, requires_grad=True)
    queries = torch.randn(2, 512)
    boxes = torch.rand(2, 40) * .6 + .2
    gt_lines = torch.rand(2, 40) * .6 + .2
    return bev, pv, queries, boxes, gt_lines


def tracks(queries, boxes, gt_lines):
    return [dict(track_query_hs_embeds=queries.clone().requires_grad_(),
                 track_query_boxes=boxes.clone(), track_query_gt_lines=gt_lines.clone(),
                 track_query_gt_labels=torch.tensor([0, 2]))]


def test_m1_native_parameters_rng_query_reference_losses_and_motion_gradient_are_exact(factory):
    baseline = factory(native=True)
    expected_init_rng = torch.random.get_rng_state().clone()
    candidate = factory()
    assert torch.equal(expected_init_rng, torch.random.get_rng_state())
    expected_state, actual_state = baseline.state_dict(), candidate.state_dict()
    assert set(expected_state) == set(actual_state)
    assert all(torch.equal(value, actual_state[key]) for key, value in expected_state.items())
    assert candidate.use_memory and type(candidate.memory_bank) is type(baseline.memory_bank)
    bev, pv, queries, boxes, gt_lines = inputs()
    install_history(candidate, boxes)
    metas = [metadata()]
    candidate._on_frame_features(bev, [pv], metas)
    poses = [torch.eye(4, dtype=torch.float64)[None]]

    def run(model):
        target = tracks(queries, boxes, gt_lines)
        torch.manual_seed(129); np.random.seed(129)
        losses = model.temporal_propagate(bev, metas, poses, poses, use_memory=False,
                                          track_query_info=target, timestep=1,
                                          get_trans_loss=True)
        state = torch.random.get_rng_state().clone()
        np_state = np.random.get_state()
        sum(losses.values()).backward()
        return target, losses, state, np_state, model.query_propagate.fc[0].weight.grad.clone()

    expected, expected_losses, expected_rng, expected_np, expected_grad = run(baseline)
    actual, actual_losses, actual_rng, actual_np, actual_grad = run(candidate)
    for key in ('track_query_hs_embeds', 'trans_track_query_boxes'):
        assert torch.equal(expected[0][key], actual[0][key])
    assert all(torch.equal(expected_losses[key], actual_losses[key]) for key in expected_losses)
    assert torch.equal(expected_grad, actual_grad) and torch.equal(expected_rng, actual_rng)
    np.testing.assert_array_equal(expected_np[1], actual_np[1])
    assert expected_np[2:] == actual_np[2:]
    assert torch.isfinite(candidate.explicit_history[0].maps).all()
    del baseline, candidate
    gc.collect()


def test_real_native_training_selection_preserves_matched_first_and_false_positive_order(factory):
    model = factory()
    model.track_fp_aug = False
    model.tracked_query_length = {}
    model.reset_explicit_history(1)
    count = model.head.num_queries
    boxes = torch.rand(count, 40) * .6 + .2
    queries = torch.randn(1, count, 512)
    logits = torch.full((1, count, 3), -20.)
    logits[0, 2, 0], logits[0, 17, 2], logits[0, 44, 1] = 2.2, .8, 1.4
    previous = dict(lines=[boxes], hs_embeds=queries, scores=logits)
    gt = [dict(lines=[boxes[:2]], labels=[torch.tensor([0, 2])])]
    previous_gt = dict(lines=[boxes], labels=[torch.zeros(count, dtype=torch.long)])
    result = model.prepare_track_queries_and_targets(
        gt, [torch.tensor([17, 2])], [torch.tensor([0, 1])], [torch.zeros(2)],
        previous_gt, previous, [torch.tensor([0, 1])], [torch.tensor([0, 1])],
        [metadata(0)], use_memory=False, timestep=0)
    order = torch.tensor([17, 2, 44])
    state = model.explicit_history[0]
    assert torch.equal(result[0]['track_query_hs_embeds'], queries[0, order])
    assert torch.equal(state.boxes, boxes[order])
    assert state.labels.tolist() == [2, 0, 1]  # prediction labels, not GT
    assert state.ids.tolist() == [0, 1, 2] and len(state.ids.unique()) == 3
    assert result[0]['track_queries_fal_pos_mask'][:3].tolist() == [False, False, True]
    assert all(not getattr(state, key).requires_grad for key in ('maps', 'uncertainty', 'boxes'))


def test_real_head_inference_deletion_newborn_order_and_native_ids(factory):
    model = factory()
    model.eval()
    model.reset_explicit_history(1)
    bev, pv, _, _, _ = inputs()
    count = model.head.num_queries
    boxes = torch.rand(count, 40) * .6 + .2
    queries = torch.randn(1, count, 512)
    logits = torch.full((1, count, 3), -20.)
    logits[0, 2, 0], logits[0, 17, 2], logits[0, 44, 1] = 2.2, .8, 1.4
    model._on_frame_features(bev, [pv], [metadata(0)])
    with torch.no_grad():
        model.head.prepare_temporal_propagation(
            dict(lines=[boxes], hs_embeds=queries, scores=logits),
            'cpu-scene', 0, memory_bank=None, thr_track=.5, thr_det=.6)
    assert model.explicit_history[0].ids.tolist() == [0, 1, 2]
    assert torch.equal(model.explicit_history[0].boxes, boxes[torch.tensor([2, 17, 44])])
    query_info = model.head.get_track_info('cpu-scene', 1)
    poses = [torch.eye(4, dtype=torch.float64)[None]]
    model._on_frame_features(bev, [pv], [metadata(1)])
    model.temporal_propagate(bev, [metadata(1)], poses, poses, False, query_info)
    next_boxes = torch.rand(count + 3, 40) * .6 + .2
    next_logits = torch.full((1, count + 3, 3), -20.)
    next_logits[0, 0, 0] = 2; next_logits[0, 2, 1] = 2; next_logits[0, 5, 2] = 2
    with torch.no_grad():
        model.head.prepare_temporal_propagation(
            dict(lines=[next_boxes], hs_embeds=torch.randn(1, count + 3, 512), scores=next_logits),
            'cpu-scene', 1, memory_bank=None, thr_track=.5, thr_det=.6)
    assert model.explicit_history[0].ids.tolist() == [0, 2, 3]
    assert model.retired_history[0].ids.tolist() == [1]
    assert torch.equal(model.explicit_history[0].ids.cpu(), model.head.prop_info['global_ids'])


def test_forward_test_scene_start_clears_explicit_state_and_native_head_restarts_ids(factory, monkeypatch):
    model = factory()
    install_history(model, torch.rand(2, 40))
    observed = {}

    def native_forward(self, *args, **kwargs):
        observed['state'] = list(self.explicit_history)
        observed['retired'] = list(self.retired_history)
        return ['native-result']

    monkeypatch.setattr(type(model).__mro__[1], 'forward_test', native_forward)
    assert model.forward_test(seq_info=[('other-scene', 0)]) == ['native-result']
    assert observed == dict(state=[None], retired=[None])
    assert model.explicit_next_ids == [0]
    assert model._frame_bev is None and model._frame_pv is None


def test_two_sample_native_selection_keeps_padding_out_of_independent_scene_states(factory):
    model = factory()
    model.track_fp_aug = False
    model.tracked_query_length = {}
    model.reset_explicit_history(2)
    count = model.head.num_queries
    boxes = [torch.rand(count, 40) * .6 + .2 for _ in range(2)]
    queries = torch.randn(2, count, 512)
    logits = torch.full((2, count, 3), -20.)
    logits[0, 2, 0], logits[0, 17, 2], logits[0, 44, 1] = 2.2, .8, 1.4
    logits[1, 5, 1] = 3
    previous = dict(lines=boxes, hs_embeds=queries, scores=logits)
    gt = [dict(lines=[boxes[0][:2], boxes[1][:1]],
               labels=[torch.tensor([0, 2]), torch.tensor([1])])]
    previous_gt = dict(lines=boxes, labels=[torch.zeros(count, dtype=torch.long)] * 2)
    result = model.prepare_track_queries_and_targets(
        gt, [torch.tensor([17, 2]), torch.tensor([5])],
        [torch.tensor([0, 1]), torch.tensor([0])], [torch.zeros(2), torch.zeros(1)],
        previous_gt, previous, [torch.tensor([0, 1]), torch.tensor([0])],
        [torch.tensor([0, 1]), torch.tensor([0])],
        [metadata(0), metadata(0, 'other-scene')], use_memory=False, timestep=0)
    assert [len(value.ids) for value in model.explicit_history] == [3, 1]
    assert [value.scene for value in model.explicit_history] == ['cpu-scene', 'other-scene']
    assert model.explicit_history[0].labels.tolist() == [2, 0, 1]
    assert model.explicit_history[1].labels.tolist() == [1]
    assert result[1]['pad_hs_embeds'].shape == (2, 512)
    assert result[1]['query_padding_mask'].sum() == 2
    assert model.explicit_history[1].maps.shape[0] == 1


def test_training_clip_wrapper_resets_state_before_and_after_parent_call(factory, monkeypatch):
    model = factory()
    install_history(model, torch.rand(2, 40))
    observed = {}

    def native_forward(self, *args, **kwargs):
        observed['before'] = list(self.explicit_history)
        self.reset_explicit_history(1)
        self.explicit_history[0] = 'temporary-state'
        return torch.tensor(1.), {}, 1

    monkeypatch.setattr(type(model).__mro__[1], 'forward_train', native_forward)
    result = model.forward_train()
    assert observed['before'] == [] and result[2] == 1
    assert model.explicit_history == [] and model.retired_history == []


def test_state_pose_variant_draws_one_native_noise_and_reuses_it_for_raster_and_references(factory, monkeypatch):
    from plugin.models.utils.history_geometry import warp_history, transform_boxes
    from scipy.spatial.transform import Rotation
    model = factory(track_transform_source='state_pose')
    bev, pv, queries, boxes, gt_lines = inputs()
    old = install_history(model, boxes)
    metas = [metadata()]; metas[0]['ego2global_translation'] = np.array([2., 0, 0])
    model._on_frame_features(bev, [pv], metas)
    stale_native = torch.eye(4, dtype=torch.float64); stale_native[0, 3] = -8
    poses = [stale_native[None]]
    calls = []

    def noisy(rot, translation):
        calls.append(translation.clone())
        return Rotation.from_euler('z', .05) * rot, translation + torch.tensor([.3, .2, 0])

    monkeypatch.setattr(model, 'add_noise_to_pose', noisy)
    target = tracks(queries, boxes, gt_lines)
    model.temporal_propagate(bev, metas, poses, poses, False, target)
    assert len(calls) == 1
    native_noisy = stale_native.clone()
    native_noisy[:3, :3] = torch.from_numpy(Rotation.from_euler('z', .05).as_matrix())
    native_noisy[:3, 3] += torch.tensor([.3, .2, 0])
    state_nominal = torch.eye(4, dtype=torch.float64); state_nominal[0, 3] = -2
    expected_t = native_noisy @ torch.linalg.inv(stale_native) @ state_nominal
    torch.testing.assert_close(model._history_transforms[0], expected_t)
    torch.testing.assert_close(model.explicit_history[0].maps,
                               warp_history(old.maps, None, None, model.roi_size, prev2curr=expected_t))
    expected_boxes = transform_boxes(boxes, None, None, model.roi_size, prev2curr=expected_t).clamp(0, 1)
    torch.testing.assert_close(target[0]['trans_track_query_boxes'], expected_boxes)


@pytest.mark.parametrize('mode', ['ce', 'bev', 'pv', 'bev_pv'])
def test_real_mapper_stage2_freezes_backbone_but_mtp_and_active_features_have_gradients(factory, mode):
    model = factory(mode=mode, freeze_bev=True)
    model.train()
    assert all(not value.requires_grad for value in model.backbone.parameters())
    assert all(not value.requires_grad for value in model.seg_decoder.parameters())
    assert all(value.requires_grad for value in model.mtp.parameters())
    bev, pv, queries, boxes, gt_lines = inputs()
    install_history(model, boxes)
    metas = [metadata()]
    model._on_frame_features(bev, [pv], metas)
    poses = [torch.eye(4, dtype=torch.float64)[None]]
    target = tracks(queries, boxes, gt_lines)
    # Disable pose perturbation in this gradient check to keep synthetic rays
    # inside all six image planes. The dedicated noisy test covers perturbation.
    model.add_noise_to_pose = lambda rotation, translation: (rotation, translation)
    model.temporal_propagate(bev, metas, poses, poses, False, target)
    weights = torch.arange(1., 513.)[None]
    (target[0]['track_query_hs_embeds'] * weights).sum().backward()
    assert model.mtp.category.weight.grad.abs().sum() > 0
    assert model.query_propagate.fc[0].weight.grad is None  # native detach kept
    if 'bev' in model.mtp.order:
        assert model.mtp.bev_proj.weight.grad.abs().sum() > 0
        assert bev.grad.abs().sum() > 0
    if 'pv' in model.mtp.order:
        assert model.mtp.pv_proj[0].weight.grad.abs().sum() > 0
        assert model.mtp.pv_pe[0].weight.grad.abs().sum() > 0
        assert pv.grad.abs().sum() > 0


@pytest.mark.parametrize('no_motion', [False, True])
def test_disabled_transform_loss_has_no_auxiliary_head_calls_and_memory_switches_are_explicit(factory, monkeypatch, no_motion):
    options = dict(transform_loss_enabled=False)
    if no_motion:
        options.update(use_motion_mlp=False, latent_memory_motion='identity')
    model = factory(use_memory=False, **options)
    assert not hasattr(model, 'memory_bank')
    bev, pv, queries, boxes, gt_lines = inputs()
    install_history(model, boxes)
    metas = [metadata()]
    model._on_frame_features(bev, [pv], metas)

    def forbidden(*args, **kwargs):
        pytest.fail('Removed transformation loss still calls the auxiliary head')

    monkeypatch.setattr(model.head.reg_branches[-1], 'forward', forbidden)
    monkeypatch.setattr(model.head.cls_branches[-1], 'forward', forbidden)
    target = tracks(queries, boxes, gt_lines)
    poses = [torch.eye(4, dtype=torch.float64)[None]]
    losses = model.temporal_propagate(bev, metas, poses, poses, False, target,
                                      get_trans_loss=True)
    assert losses == {}
    if no_motion:
        assert not list(model.query_propagate.parameters())
        assert model._memory_motion_module() is model.identity_motion
        assert torch.equal(target[0]['track_query_hs_embeds'], queries)
