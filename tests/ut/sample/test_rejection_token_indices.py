from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend.sample.rejection_sampler import (
    _make_rejection_token_indices,
    rejection_random_sample_pytorch,
)


class RejectTinyBroadcastAdd(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.add.Tensor and len(args) >= 2:
            lhs, rhs = args[:2]
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                if lhs.shape == (1, 1) and rhs.ndim == 2:
                    raise AssertionError("Redundant single-request broadcast Add")
            if isinstance(lhs, torch.Tensor) and lhs.numel() == 1 and isinstance(rhs, int):
                raise AssertionError("Static bound must be computed before H2D")
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("counts", [(15,), (1,), (0,), (15, 7), (0, 3, 0, 15)])
@pytest.mark.parametrize("width", [1, 3, 7, 15, 31])
def test_index_grid_matches_cumulative_offsets_without_changing_rng(counts, width):
    cumulative = torch.tensor(counts, dtype=torch.long).cumsum(0)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), cumulative[:-1]))
    positions = torch.arange(width, dtype=torch.long)[None, :]
    expected = starts[:, None] + positions
    rng_before = torch.random.get_rng_state()
    with RejectTinyBroadcastAdd():
        actual = _make_rejection_token_indices(starts, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    if len(counts) == 1:
        assert actual is positions


def without_pinning(factory):
    def create(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return factory(*args, **kwargs)

    return create


@pytest.mark.parametrize("count", [1, 7, 15])
@pytest.mark.parametrize("rejection", [0, 7, 14, 15])
def test_single_request_stochastic_output_keeps_accepted_prefix_and_bonus(count, rejection):
    width = 15
    output = torch.full((1, width + 1), -1, dtype=torch.int32)
    draft_ids = torch.ones(count, dtype=torch.int32)
    draft_probs = torch.zeros((count, 4), dtype=torch.float32)
    draft_probs[:, 1] = 1.0
    target_probs = draft_probs.clone()
    if rejection < count:
        target_probs[rejection] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    rng_before = torch.random.get_rng_state()
    with ExitStack() as stack:
        for name in ("tensor", "arange", "ones", "full"):
            stack.enter_context(patch.object(torch, name, without_pinning(getattr(torch, name))))
        stack.enter_context(RejectTinyBroadcastAdd())
        rejection_random_sample_pytorch(
            output, torch.tensor([count]), draft_ids, draft_probs, target_probs,
            torch.tensor([[3]], dtype=torch.int32), torch.full((count,), 2, dtype=torch.int32),
            torch.full((count,), 0.5), torch.tensor([False]), width, 4,
        )
    expected = torch.full_like(output, -1)
    accepted = min(count, rejection)
    expected[0, :accepted] = 1
    expected[0, accepted] = 2 if rejection < count else 3
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
