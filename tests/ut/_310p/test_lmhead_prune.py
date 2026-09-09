from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend._310p.lmhead_prune import maybe_prune_lm_head


class _FakeQuantMethod:
    def apply(self, lm_head, hidden_states):
        rows = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
        values = torch.arange(rows * 2, dtype=hidden_states.dtype, device=hidden_states.device)
        return values.reshape(*hidden_states.shape[:-1], 2)


def _make_pack():
    return {
        "mode": "int8",
        "weight": torch.ones((2, 4), dtype=torch.int8),
        "deq_scale": torch.ones(2, dtype=torch.int64),
        "quant_bias": torch.zeros(2, dtype=torch.int32),
        "inv_map": torch.tensor([0, 2, 1], dtype=torch.int64),
        "orig_vocab": 3,
    }


def test_pruned_lm_head_uses_safe_load_and_head_device():
    lm_head = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros((3, 4), dtype=torch.int8)),
        deq_scale=SimpleNamespace(data=torch.zeros(3, dtype=torch.int64)),
        quant_bias=SimpleNamespace(data=torch.zeros(3, dtype=torch.int32)),
        quant_method=_FakeQuantMethod(),
    )
    original_calls = []

    def original_compute_logits(hidden_states):
        original_calls.append(hidden_states)
        return lm_head.quant_method.apply(lm_head, hidden_states)

    model = SimpleNamespace(
        compute_logits=original_compute_logits,
        lm_head=lm_head,
        logits_processor=SimpleNamespace(org_vocab_size=3, scale=1.0, soft_cap=None),
    )
    pack = _make_pack()

    with (
        patch.dict("os.environ", {"VLLM_LMHEAD_PRUNE_PACK": "/tmp/prune-pack.pt"}),
        patch("vllm_ascend._310p.lmhead_prune.torch.load", return_value=pack) as load_pack,
        patch("vllm_ascend._310p.lmhead_prune.maybe_trans_nz", side_effect=lambda value: value),
    ):
        maybe_prune_lm_head(model)

    load_pack.assert_called_once_with("/tmp/prune-pack.pt", map_location="cpu", weights_only=True)
    assert lm_head.weight.data.device.type == "cpu"
    assert lm_head.weight.data.shape == (4, 2)

    hidden_f32 = torch.zeros((1, 4), dtype=torch.float32)
    hidden_f16 = torch.zeros((2, 4), dtype=torch.float16)
    output_f32 = model.compute_logits(hidden_f32)
    output_f16 = model.compute_logits(hidden_f16)
    assert output_f32.shape == (1, 3)
    assert output_f16.shape == (2, 3)
    assert output_f32.dtype == torch.float32
    assert output_f16.dtype == torch.float16
    assert output_f32[0, 0] == 0
    assert output_f32[0, 1] == torch.finfo(torch.float32).min
    assert output_f32[0, 2] == 1
    assert original_calls[0] is hidden_f32
    assert original_calls[1] is hidden_f16


def test_pruned_lm_head_preserves_step_aware_compute_logits():
    lm_head = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros((3, 4), dtype=torch.int8)),
        deq_scale=SimpleNamespace(data=torch.zeros(3, dtype=torch.int64)),
        quant_bias=SimpleNamespace(data=torch.zeros(3, dtype=torch.int32)),
        quant_method=_FakeQuantMethod(),
    )
    selected_heads = []
    per_step_head_logits = ([0, 1], [10, 11, 12])

    def original_compute_logits(hidden_states, spec_step_idx=0):
        head_idx = spec_step_idx % len(per_step_head_logits)
        selected_heads.append(head_idx)
        rows = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
        values = torch.tensor(
            per_step_head_logits[head_idx],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return values.expand(rows, -1).reshape(*hidden_states.shape[:-1], values.shape[-1])

    model = SimpleNamespace(
        compute_logits=original_compute_logits,
        lm_head=lm_head,
        logits_processor=SimpleNamespace(org_vocab_size=3, scale=1.0, soft_cap=None),
    )

    with (
        patch.dict("os.environ", {"VLLM_LMHEAD_PRUNE_PACK": "/tmp/prune-pack.pt"}),
        patch("vllm_ascend._310p.lmhead_prune.torch.load", return_value=_make_pack()),
        patch("vllm_ascend._310p.lmhead_prune.maybe_trans_nz", side_effect=lambda value: value),
    ):
        maybe_prune_lm_head(model)

    hidden_states = torch.zeros((1, 4))
    output_pruned = model.compute_logits(hidden_states, spec_step_idx=0)
    output_full = model.compute_logits(hidden_states, spec_step_idx=3)

    assert selected_heads == [0, 1]
    assert torch.equal(output_pruned[:, [0, 2]], torch.tensor([[0.0, 1.0]]))
    assert output_pruned[0, 1] == torch.finfo(output_pruned.dtype).min
    assert torch.equal(output_full, torch.tensor([[10.0, 11.0, 12.0]]))


def test_pruned_lm_head_preserves_none_logits_on_non_last_rank():
    lm_head = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros((3, 4), dtype=torch.int8)),
        deq_scale=SimpleNamespace(data=torch.zeros(3, dtype=torch.int64)),
        quant_bias=SimpleNamespace(data=torch.zeros(3, dtype=torch.int32)),
        quant_method=_FakeQuantMethod(),
    )
    calls = []

    def original_compute_logits(hidden_states):
        calls.append(hidden_states)
        return None

    model = SimpleNamespace(
        compute_logits=original_compute_logits,
        lm_head=lm_head,
        logits_processor=SimpleNamespace(org_vocab_size=3, scale=1.0, soft_cap=None),
    )

    with (
        patch.dict("os.environ", {"VLLM_LMHEAD_PRUNE_PACK": "/tmp/prune-pack.pt"}),
        patch("vllm_ascend._310p.lmhead_prune.torch.load", return_value=_make_pack()),
        patch("vllm_ascend._310p.lmhead_prune.maybe_trans_nz", side_effect=lambda value: value),
    ):
        maybe_prune_lm_head(model)

    hidden_states = torch.zeros((1, 4))
    assert model.compute_logits(hidden_states) is None
    assert calls[0] is hidden_states


def test_pruned_lm_head_rejects_tensor_parallel_head_before_mutation():
    original_weight = torch.zeros((3, 4), dtype=torch.int8)
    lm_head = SimpleNamespace(
        weight=SimpleNamespace(data=original_weight),
        deq_scale=SimpleNamespace(data=torch.zeros(3, dtype=torch.int64)),
        quant_bias=SimpleNamespace(data=torch.zeros(3, dtype=torch.int32)),
        quant_method=_FakeQuantMethod(),
        tp_size=2,
    )

    def original_compute_logits(hidden_states):
        return lm_head.quant_method.apply(lm_head, hidden_states)

    model = SimpleNamespace(
        compute_logits=original_compute_logits,
        lm_head=lm_head,
        logits_processor=SimpleNamespace(org_vocab_size=3, scale=1.0, soft_cap=None),
    )

    with (
        patch.dict("os.environ", {"VLLM_LMHEAD_PRUNE_PACK": "/tmp/prune-pack.pt"}),
        patch("vllm_ascend._310p.lmhead_prune.torch.load", return_value=_make_pack()),
        patch("vllm_ascend._310p.lmhead_prune.maybe_trans_nz") as trans_nz,
        pytest.raises(NotImplementedError, match="supports only tensor-parallel size 1"),
    ):
        maybe_prune_lm_head(model)

    assert lm_head.weight.data is original_weight
    assert model.compute_logits is original_compute_logits
    trans_nz.assert_not_called()
