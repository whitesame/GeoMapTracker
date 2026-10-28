"""Check M1 bookkeeping and native parameter/gradient traversal on real models.

CPU checks deliberately do not claim deterministic CUDA/DDP training.  The
queue also compares 603-frame B0/M1 inference from one trained checkpoint.
"""
import gc
import random

import numpy as np
import torch

from test_mapper_integration import factory, inputs, install_history, metadata, tracks


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def assert_rng_unchanged(before):
    python_before, numpy_before, torch_before = before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())


def test_m1_real_callbacks_do_not_consume_python_numpy_or_torch_rng(factory):
    model = factory()
    bev, pv, queries, boxes, _ = inputs()
    logits = torch.tensor([[[2., -2., -3.], [-3., 2., -2.]]])
    selection = torch.tensor([1, 0])
    scores, labels = logits[0, selection].max(-1)
    target = dict(track_query_hs_embeds=queries[selection].clone(),
                  track_query_labels=labels, track_query_scores=scores.sigmoid())
    predictions = dict(lines=[boxes], scores=logits)
    initial_meta, next_meta = metadata(0), metadata(1)
    transform = torch.eye(4, dtype=torch.float64)
    random.seed(911)
    np.random.seed(911)
    torch.manual_seed(911)
    before = rng_snapshot()

    # These are the actual new training callbacks, including rasterization,
    # overwrite update, pose alignment and diagnostics with nonempty history.
    model._on_frame_features(bev, [pv], [initial_meta])
    model._on_training_selection(0, target, predictions, selection, initial_meta)
    model._on_frame_features(bev, [pv], [next_meta])
    returned = model._adapt_track_transform(0, next_meta, transform, transform, target)
    assert returned is transform
    model._on_temporal_transform(0, next_meta, transform, target)
    model._on_training_selection(0, target, predictions, selection, next_meta)
    assert len(model.explicit_history[0].ids) == 2
    assert_rng_unchanged(before)

    # Exercise the native inference head's real observer dispatch as well.
    model.eval()
    model.reset_explicit_history(1)
    before = rng_snapshot()
    model._on_frame_features(bev, [pv], [initial_meta])
    with torch.no_grad():
        model.head.prepare_temporal_propagation(
            dict(lines=[boxes], hs_embeds=queries[None], scores=logits),
            'cpu-scene', 0, memory_bank=None, thr_track=.5, thr_det=.6)
    assert len(model.explicit_history[0].ids) == 2
    assert_rng_unchanged(before)
    assert not torch.cuda.is_initialized()
    del model
    gc.collect()


def test_m1_native_parameter_order_and_all_shared_temporal_gradients_are_exact(factory):
    baseline, candidate = factory(native=True), factory()
    baseline_parameters = list(baseline.named_parameters())
    candidate_parameters = list(candidate.named_parameters())
    assert [(name, tuple(value.shape), value.requires_grad)
            for name, value in baseline_parameters] == [
                (name, tuple(value.shape), value.requires_grad)
                for name, value in candidate_parameters]
    assert all(torch.equal(left, right) for (_, left), (_, right)
               in zip(baseline_parameters, candidate_parameters))
    bev, pv, queries, boxes, gt_lines = inputs()
    install_history(candidate, boxes)
    metas = [metadata()]
    candidate._on_frame_features(bev, [pv], metas)
    poses = [torch.eye(4, dtype=torch.float64)[None]]

    def backward(model):
        target = tracks(queries, boxes, gt_lines)
        random.seed(129)
        np.random.seed(129)
        torch.manual_seed(129)
        losses = model.temporal_propagate(
            bev, metas, poses, poses, use_memory=False, track_query_info=target,
            timestep=1, get_trans_loss=True)
        sum(losses.values()).backward()
        return [(name, None if value.grad is None else value.grad.detach().clone())
                for name, value in model.named_parameters()]

    expected, actual = backward(baseline), backward(candidate)
    expected_active = [name for name, gradient in expected if gradient is not None]
    actual_active = [name for name, gradient in actual if gradient is not None]
    assert expected_active == actual_active
    assert any(name.startswith('query_propagate.') for name in expected_active)
    assert any(name.startswith('head.cls_branches.') for name in expected_active)
    assert any(name.startswith('head.reg_branches.') for name in expected_active)
    for (expected_name, expected_gradient), (actual_name, actual_gradient) in zip(expected, actual):
        assert expected_name == actual_name
        if expected_gradient is None:
            assert actual_gradient is None
        else:
            assert torch.equal(expected_gradient, actual_gradient), expected_name
    assert not torch.cuda.is_initialized()
    del baseline, candidate
    gc.collect()
