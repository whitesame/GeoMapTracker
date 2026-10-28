"""Regression against the actual MMCV writer, not a fabricated torch format."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from plugin.models.utils.checkpoint_compat import (
    canonical_state_dict, load_checkpoint_strict, load_state_dict_strict)


class BufferPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('depths', torch.linspace(1., 60., 32), persistent=False)
        self.register_buffer('position_range', torch.tensor([-30., -15., -5., 30., 15., 5.]),
                             persistent=False)


class BufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.norm = nn.BatchNorm1d(2)
        self.mtp = BufferPrior()


@pytest.fixture
def written(tmp_path):
    from mmcv.runner import save_checkpoint
    model = BufferModel()
    path = tmp_path / 'real_mmcv_checkpoint.pth'
    save_checkpoint(model, str(path), meta=dict(iter=19))
    serialized = torch.load(str(path), map_location='cpu')['state_dict']
    # This assertion exercises the real behavior that broke the previous queue.
    assert set(serialized) - set(model.state_dict()) == {
        'mtp.depths', 'mtp.position_range'}
    return model, path, serialized


def test_real_mmcv_writer_roundtrip_removes_only_exact_declared_buffers(written):
    model, path, serialized = written
    target = BufferModel()
    canonical, evidence = load_state_dict_strict(target, serialized)
    assert set(canonical) == set(model.state_dict())
    assert evidence['exact_nonpersistent_buffers_removed'] == [
        'mtp.depths', 'mtp.position_range']
    assert all(torch.equal(value, target.state_dict()[key])
               for key, value in model.state_dict().items())
    loaded = load_checkpoint_strict(target, path)
    assert loaded['meta']['iter'] == 19


@pytest.mark.parametrize('mutation', ['value', 'shape', 'dtype'])
@pytest.mark.parametrize('key', ['mtp.depths', 'mtp.position_range'])
def test_redundant_buffer_tampering_is_rejected(written, key, mutation):
    model, _, serialized = written
    serialized = copy.deepcopy(serialized)
    if mutation == 'value':
        serialized[key][0] += .001
    elif mutation == 'shape':
        serialized[key] = serialized[key][None]
    else:
        serialized[key] = serialized[key].double()
    with pytest.raises(RuntimeError, match='Nonpersistent buffer'):
        load_state_dict_strict(model, serialized)


def test_arbitrary_mtp_key_is_not_silently_discarded(written):
    model, _, serialized = written
    serialized['mtp.surprise'] = torch.ones(1)
    before = model.linear.weight.detach().clone()
    with pytest.raises(RuntimeError, match='unexpected=.*mtp.surprise'):
        load_state_dict_strict(model, serialized)
    assert torch.equal(before, model.linear.weight)


def test_missing_learned_parameter_is_rejected(written):
    model, _, serialized = written
    del serialized['linear.weight']
    with pytest.raises(RuntimeError, match='missing=.*linear.weight'):
        load_state_dict_strict(model, serialized)


@pytest.mark.parametrize('mutation', ['shape', 'dtype', 'nan'])
def test_persistent_parameter_shape_dtype_and_finiteness_stay_strict(written, mutation):
    model, _, serialized = written
    if mutation == 'shape':
        serialized['linear.weight'] = serialized['linear.weight'][None]
    elif mutation == 'dtype':
        serialized['linear.weight'] = serialized['linear.weight'].double()
    else:
        serialized['linear.weight'][0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='Persistent state'):
        load_state_dict_strict(model, serialized)


def test_learning_updates_to_persistent_weights_are_preserved(written):
    model, _, serialized = written
    serialized['linear.weight'] = serialized['linear.weight'] + .25
    canonical, _ = load_state_dict_strict(model, serialized)
    assert torch.equal(model.linear.weight, canonical['linear.weight'])


def test_canonical_torch_checkpoint_needs_no_compatibility_exception(written):
    model, _, _ = written
    canonical, evidence = canonical_state_dict(model, model.state_dict())
    assert evidence['exact_nonpersistent_buffers_removed'] == []
    assert set(canonical) == set(model.state_dict())


def test_inference_loader_rejects_unknown_checkpoint_tensor(written):
    model, path, serialized = written
    serialized['unknown.weight'] = torch.ones(1)
    torch.save(dict(state_dict=serialized), str(path))
    with pytest.raises(RuntimeError, match='unknown.weight'):
        load_checkpoint_strict(model, path)


def test_actual_native724_vs_mainline726_mmcv_checkpoints_roundtrip(tmp_path):
    """Reproduce the exact native/M1 tensor counts from the failed queue."""
    from mmcv import Config
    from mmcv.runner import save_checkpoint
    from mmdet3d.models import build_model
    root = Path(__file__).resolve().parents[2]
    config = Config.fromfile(str(root / 'plugin/configs/maptracker/'
                                'nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py'))
    config.model.backbone_cfg.img_backbone.pretrained = None
    outputs = {}
    for name in ('native', 'identity'):
        options = copy.deepcopy(config.model)
        if name == 'identity':
            options.update(type='HisTrackMapMainline', prior_enabled=False,
                           track_transform_source='native',
                           mtp_cfg=dict(mode='disabled'))
        torch.manual_seed(0)
        model = build_model(options)
        assert len(model.state_dict()) == 724
        path = tmp_path / (name + '.pth')
        save_checkpoint(model, str(path), meta=dict(iter=19))
        checkpoint = torch.load(str(path), map_location='cpu')
        assert len(checkpoint['state_dict']) == (724 if name == 'native' else 726)
        canonical, evidence = load_state_dict_strict(model, checkpoint['state_dict'])
        assert evidence['persistent_state_tensors'] == 724
        assert evidence['exact_nonpersistent_buffers_removed'] == (
            [] if name == 'native' else ['mtp.depths', 'mtp.position_range'])
        outputs[name] = canonical
        del model
    assert set(outputs['native']) == set(outputs['identity'])
    assert all(torch.equal(value, outputs['identity'][key])
               for key, value in outputs['native'].items())
