"""Share identical static input quantization for the two 310P GDN projections.

Only loaded, equal quantization parameters authorize sharing. There is no
activation cache, and the original hidden-state dtype remains available to
the attention output allocation. No attention kernel or weight layout changes.
"""

import torch
from einops import rearrange
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.utils.torch_utils import _encode_layer_name

from vllm_ascend import envs
from vllm_ascend.utils import vllm_version_is

_ORIGINAL_FORWARD_CUDA = QwenGatedDeltaNetAttention.forward_cuda
_INPUT_WIDTH = 4096
_QUANT_FIELDS = ("aclnn_input_scale", "aclnn_input_scale_reciprocal", "aclnn_input_offset")


def _projection_is_supported(linear):
    scheme = getattr(getattr(linear, "quant_method", None), "quant_method", None)
    return (
        type(linear).__name__ in ("MergedColumnParallelLinear", "AscendMergedColumnParallelLinear")
        and type(scheme).__name__ == "AscendW8A8LinearMethod310"
        and getattr(linear, "custom_op", None) is None
        and getattr(linear, "tp_size", None) == 1
        and getattr(linear, "bias", None) is None
        and getattr(linear, "params_dtype", None) == torch.float16
    )


def prepare_shared_gdn_quant(model):
    """Called after all quant methods finish loading/reloading, not in forward."""
    for _, module in model.named_modules():
        if not isinstance(module, QwenGatedDeltaNetAttention):
            continue
        module._shared_input_quant_verified = False
        if not envs.VLLM_ASCEND_GDN_SHARED_INPUT_QUANT:
            continue
        qkvz, ba = getattr(module, "in_proj_qkvz", None), getattr(module, "in_proj_ba", None)
        if (getattr(module, "gqa_interleaved_layout", True) or module.tp_size != 1
                or not _projection_is_supported(qkvz) or not _projection_is_supported(ba)):
            continue
        valid = True
        for name in _QUANT_FIELDS:
            first, second = getattr(qkvz, name, None), getattr(ba, name, None)
            if (first is None or second is None or first.shape != (_INPUT_WIDTH,)
                    or first.shape != second.shape or first.dtype != torch.float16
                    or first.dtype != second.dtype or first.device.type != "npu"
                    or first.device != second.device or not torch.equal(first, second)):
                valid = False
                break
        module._shared_input_quant_verified = valid


def shared_quant_forward_cuda(self, hidden_states, output):
    if not getattr(self, "_shared_input_quant_verified", False):
        return _ORIGINAL_FORWARD_CUDA(self, hidden_states, output)
    num_tokens = hidden_states.size(0)
    projection_input = hidden_states
    if hidden_states.dtype != torch.int8:
        projection_input = torch.ops.vllm.quantize(
            hidden_states, self.in_proj_qkvz.aclnn_input_scale,
            self.in_proj_qkvz.aclnn_input_scale_reciprocal,
            self.in_proj_qkvz.aclnn_input_offset)
    mixed_qkvz, _ = self.in_proj_qkvz(projection_input)
    ba, _ = self.in_proj_ba(projection_input)

    if self.gqa_interleaved_layout:
        query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
        query, key, value = map(lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)
    else:
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, a = self.split_ba(ba)
        b = b.contiguous()
        a = a.contiguous()

    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype, device=hidden_states.device)
    torch.ops.vllm.qwen_gdn_attention_core(
        mixed_qkv, b, a, core_attn_out, layer_name=_encode_layer_name(self.prefix))
    self._output_projection(core_attn_out, z, output, num_tokens)


if envs.VLLM_ASCEND_GDN_SHARED_INPUT_QUANT:
    if not vllm_version_is("0.24.0"):
        raise RuntimeError("Shared GDN quantization requires the verified vLLM 0.24 forward API")
    QwenGatedDeltaNetAttention.forward_cuda = shared_quant_forward_cuda
