"""Exercise real bank updates through saturation, deletion and slot reordering.

The checked tensor rejects an overlapping source/destination assignment even
when a CPU happens to copy it in a safe order. CUDA blocks have no such order.
"""
from collections import defaultdict, deque

import pytest
import torch

from plugin.models.mapers.vector_memory import VectorInstanceMemory


class CheckedStorageTensor(torch.Tensor):
    def __setitem__(self, index, value):
        if isinstance(value, torch.Tensor):
            destination = super().__getitem__(index)
            assert not torch._C._overlaps(destination, value), (
                'Memory-bank assignment must snapshot overlapping source storage')
        return super().__setitem__(index, value)


@pytest.mark.parametrize('bank_size', [3, 20])
def test_real_update_memory_matches_independent_time_series_for_multiple_batches(bank_size):
    batches, channels, number_instances, queries = 2, 8, 8, 16
    bank = VectorInstanceMemory(dim_in=channels, number_ins=number_instances,
        bank_size=bank_size, mem_len=min(4, bank_size), mem_select_dist_ranges=[1, 5, 10, 15])
    # init_memory is CUDA-specific; install the same storage/metadata on CPU so
    # the actual update_memory body runs unchanged, including its CPU counters.
    shape = (bank_size, batches, bank.max_number_ins)
    bank.mem_bank = torch.zeros(*shape, channels).as_subclass(CheckedStorageTensor)
    bank.mem_bank_seq_id = torch.zeros(*shape, dtype=torch.long).as_subclass(CheckedStorageTensor)
    bank.mem_bank_rot = torch.zeros(bank_size, batches, 3, 3)
    bank.mem_bank_trans = torch.zeros(bank_size, batches, 3)
    bank.curr_rot = torch.eye(3).repeat(batches, 1, 1)
    bank.curr_trans = torch.zeros(batches, 3)
    bank.instance2mem = [{} for _ in range(batches)]
    bank.num_ins = [0 for _ in range(batches)]
    bank.mem_entry_lengths = [torch.zeros(bank.max_number_ins, dtype=torch.long)
                              for _ in range(batches)]
    bank.active_mem_ids = [None for _ in range(batches)]
    bank.valid_track_idx = [None for _ in range(batches)]

    active = [[] for _ in range(batches)]
    next_ids = [0] * batches
    embeddings = [defaultdict(lambda: deque(maxlen=bank_size)) for _ in range(batches)]
    sequence_ids = [defaultdict(lambda: deque(maxlen=bank_size)) for _ in range(batches)]
    total_lengths = [defaultdict(int) for _ in range(batches)]
    poses = [deque(maxlen=bank_size) for _ in range(batches)]

    for frame in range(bank_size + 6):
        bank.curr_t = frame
        features = torch.arange(batches * queries * channels, dtype=torch.float32).reshape(
            batches, queries, channels) + 10000 * frame
        predictions = dict(hs_embeds=features, scores=torch.zeros(batches, queries, 3))
        for batch in range(batches):
            old_active = active[batch]
            if frame == 0:
                selected = [1, 3, 5]
            elif frame == 1:
                selected = [2, 0, 1]  # keep the same instances in another query order
            elif frame == 2:
                selected = [0, 2] if batch == 0 else [1, 2]  # independent retirements
            elif frame == 4:
                selected = list(range(len(old_active))) + [len(old_active) + 3]
            elif frame == bank_size + 2:
                selected = list(reversed(range(len(old_active))))
            else:
                selected = list(range(len(old_active)))

            expected_active = []
            for query in selected:
                if frame and query < len(old_active):
                    memory_id = old_active[query]
                else:
                    memory_id = next_ids[batch]
                    next_ids[batch] += 1
                expected_active.append(memory_id)
                embeddings[batch][memory_id].append(features[batch, query].clone())
                sequence_ids[batch][memory_id].append(frame)
                total_lengths[batch][memory_id] += 1

            bank.curr_trans[batch] = torch.tensor([100 * frame + batch, -frame, batch])
            poses[batch].append(bank.curr_trans[batch].clone())
            bank.update_memory(batch, frame == 0, torch.tensor(selected, dtype=torch.long),
                predictions, num_tracks=len(old_active), seq_idx=frame, timestep=frame)
            active[batch] = expected_active

        expected_bank = torch.zeros(*shape, channels)
        expected_sequences = torch.zeros(*shape, dtype=torch.long)
        expected_translations = torch.zeros(bank_size, batches, 3)
        expected_rotations = torch.zeros(bank_size, batches, 3, 3)
        for batch in range(batches):
            for memory_id, history in embeddings[batch].items():
                expected_bank[:len(history), batch, memory_id] = torch.stack(list(history))
                expected_sequences[:len(history), batch, memory_id] = torch.tensor(
                    list(sequence_ids[batch][memory_id]))
            expected_translations[:len(poses[batch]), batch] = torch.stack(list(poses[batch]))
            expected_rotations[:len(poses[batch]), batch] = torch.eye(3).repeat(len(poses[batch]), 1, 1)
            expected_lengths = torch.zeros(bank.max_number_ins, dtype=torch.long)
            for memory_id, length in total_lengths[batch].items():
                expected_lengths[memory_id] = length
            assert torch.equal(bank.mem_entry_lengths[batch], expected_lengths)
            assert bank.instance2mem[batch] == dict(enumerate(active[batch]))
            assert bank.num_ins[batch] == next_ids[batch]
            assert torch.equal(bank.active_mem_ids[batch], torch.tensor(active[batch]))
            assert torch.equal(bank.valid_track_idx[batch], torch.arange(len(active[batch])))
        assert torch.equal(bank.mem_bank, expected_bank)
        assert torch.equal(bank.mem_bank_seq_id, expected_sequences)
        assert torch.equal(bank.mem_bank_trans, expected_translations)
        assert torch.equal(bank.mem_bank_rot, expected_rotations)
    assert max(total_lengths[0].values()) > bank_size
    assert max(total_lengths[1].values()) > bank_size
    assert not torch.cuda.is_initialized()
