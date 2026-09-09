from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend._310p.lmhead_prune import maybe_prune_lm_head


class _FakeQuantMethod:
    def apply(self, lm_head, hidden_states):
        rows = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
        values = torch.arange(rows * 2, dtype=hidden_states.dtype, device=hidden_states.device)
        return values.reshape(*hidden_states.shape[:-1], 2)


def test_pruned_lm_head_uses_safe_load_and_head_device():
    lm_head = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros((3, 4), dtype=torch.int8)),
        deq_scale=SimpleNamespace(data=torch.zeros(3, dtype=torch.int64)),
        quant_bias=SimpleNamespace(data=torch.zeros(3, dtype=torch.int32)),
        quant_method=_FakeQuantMethod(),
    )
    model = SimpleNamespace(
        compute_logits=lambda hidden_states: hidden_states,
        lm_head=lm_head,
        logits_processor=SimpleNamespace(org_vocab_size=3, scale=1.0, soft_cap=None),
    )
    pack = {
        "mode": "int8",
        "weight": torch.ones((2, 4), dtype=torch.int8),
        "deq_scale": torch.ones(2, dtype=torch.int64),
        "quant_bias": torch.zeros(2, dtype=torch.int32),
        "inv_map": torch.tensor([0, 2, 1], dtype=torch.int64),
        "orig_vocab": 3,
    }

    with (
        patch.dict("os.environ", {"VLLM_LMHEAD_PRUNE_PACK": "/tmp/prune-pack.pt"}),
        patch("vllm_ascend._310p.lmhead_prune.torch.load", return_value=pack) as load_pack,
        patch("vllm_ascend._310p.lmhead_prune.maybe_trans_nz", side_effect=lambda value: value),
    ):
        maybe_prune_lm_head(model)

    load_pack.assert_called_once_with("/tmp/prune-pack.pt", map_location="cpu", weights_only=True)
    assert lm_head.weight.data.device.type == "cpu"
    assert lm_head.weight.data.shape == (4, 2)

    output_f32 = model.compute_logits(torch.zeros((1, 4), dtype=torch.float32))
    output_f16 = model.compute_logits(torch.zeros((2, 4), dtype=torch.float16))
    assert output_f32.shape == (1, 3)
    assert output_f16.shape == (2, 3)
    assert output_f32.dtype == torch.float32
    assert output_f16.dtype == torch.float16
    assert output_f32[0, 0] == 0
    assert output_f32[0, 1] == torch.finfo(torch.float32).min
    assert output_f32[0, 2] == 1
