"""Strict checkpoint loading across the MMCV 1.7 / torch buffer mismatch.

MMCV's checkpoint writer includes buffers which torch declares nonpersistent.
Such entries may be discarded only when the target model declares the same
buffer and its serialized shape, dtype and value exactly match that model.
Every persistent tensor is still required and loaded with ``strict=True``.
"""
from collections import OrderedDict
from collections.abc import Mapping

import torch


def canonical_state_dict(model, serialized):
    """Return strict persistent state plus evidence for redundant buffer keys."""
    if not isinstance(serialized, Mapping):
        raise TypeError('Checkpoint state_dict must be a mapping')
    expected = model.state_dict()
    nonpersistent = {}
    for prefix, module in model.named_modules():
        for name in module._non_persistent_buffers_set:
            value = module._buffers.get(name)
            if value is not None:
                key = prefix + '.' + name if prefix else name
                nonpersistent[key] = value
    missing = set(expected) - set(serialized)
    extras = set(serialized) - set(expected)
    unknown = extras - set(nonpersistent)
    if missing or unknown:
        raise RuntimeError('Strict checkpoint keys differ: missing=%s unexpected=%s' %
                           (sorted(missing), sorted(unknown)))
    for key in sorted(extras):
        actual, declared = serialized[key], nonpersistent[key]
        if not isinstance(actual, torch.Tensor):
            raise TypeError('Nonpersistent buffer is not a tensor: ' + key)
        if actual.shape != declared.shape or actual.dtype != declared.dtype:
            raise RuntimeError('Nonpersistent buffer shape/dtype mismatch: ' + key)
        if not torch.equal(actual.detach().cpu(), declared.detach().cpu()):
            raise RuntimeError('Nonpersistent buffer value mismatch: ' + key)
    canonical = OrderedDict()
    for key, target in expected.items():
        value = serialized[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError('Persistent state is not a tensor: ' + key)
        if value.shape != target.shape or value.dtype != target.dtype:
            raise RuntimeError('Persistent state shape/dtype mismatch: ' + key)
        if not torch.isfinite(value).all():
            raise RuntimeError('Persistent state is not finite: ' + key)
        canonical[key] = value
    if hasattr(serialized, '_metadata'):
        canonical._metadata = serialized._metadata
    return canonical, dict(
        serialized_state_tensors=len(serialized),
        persistent_state_tensors=len(canonical),
        exact_nonpersistent_buffers_removed=sorted(extras),
        persistent_keys_shapes_dtypes_and_finiteness_verified=True,
        redundant_buffers_checked='model declaration, exact shape, dtype and value')


def load_state_dict_strict(model, serialized):
    """Validate all tensors before mutating the model, then load strictly."""
    canonical, evidence = canonical_state_dict(model, serialized)
    model.load_state_dict(canonical, strict=True)
    return canonical, evidence


def load_checkpoint_strict(model, filename, map_location='cpu'):
    """Load a real MMCV checkpoint without its permissive state-dict loader."""
    from mmcv.runner.checkpoint import _load_checkpoint
    checkpoint = _load_checkpoint(str(filename), map_location=map_location)
    if not isinstance(checkpoint, Mapping) or 'state_dict' not in checkpoint:
        raise RuntimeError('Expected a checkpoint with an explicit state_dict')
    _, evidence = load_state_dict_strict(model, checkpoint['state_dict'])
    print('Strict checkpoint compatibility:', evidence)
    return checkpoint
