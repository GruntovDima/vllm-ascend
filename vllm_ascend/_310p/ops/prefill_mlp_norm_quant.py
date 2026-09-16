"""Opt-in existing-CANN fusion before the Qwen3.5 static-W8A8 MLP.

Not bitwise equivalent to split FP16 norm/quant. Keep decode and all other
consumers on the original path; in particular attention must not see INT8
hidden_states when it allocates its output using hidden_states.dtype.
"""

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_ascend import envs

PREFILL_MIN_TOKENS = 128
HIDDEN_SIZE = 4096


def prepare_mlp_norm_quant_params(layer):
    """Called by the 310P static scheme after loading; never cache activations."""
    if (
        not envs.VLLM_ASCEND_PREFILL_MLP_NORM_QUANT
        or not getattr(layer, "prefix", "").endswith(".mlp.gate_up_proj")
        or layer.params_dtype != torch.float16
        or layer.aclnn_input_scale_reciprocal.shape != (HIDDEN_SIZE,)
        or layer.input_offset.dtype != torch.int8
    ):
        return
    # Preserve the model's FP16 reciprocal rounding, then promote for this API.
    layer.register_buffer("_prefill_norm_quant_scale", layer.aclnn_input_scale_reciprocal.float(), persistent=False)
    layer.register_buffer("_prefill_norm_quant_offset", layer.aclnn_input_offset.to(torch.int32), persistent=False)


def maybe_fused_mlp_norm_quant(decoder, x, residual):
    if not envs.VLLM_ASCEND_PREFILL_MLP_NORM_QUANT or residual is None:
        return None
    norm = decoder.post_attention_layernorm
    mlp = decoder.mlp
    linear = getattr(mlp, "gate_up_proj", None)
    scale = getattr(linear, "_prefill_norm_quant_scale", None)
    offset = getattr(linear, "_prefill_norm_quant_offset", None)
    scheme = getattr(getattr(linear, "quant_method", None), "quant_method", None)
    if (
        type(norm).__name__ not in ("GemmaRMSNorm", "AscendGemmaRMSNorm310")
        or type(mlp).__name__ != "Qwen2MoeMLP"
        or getattr(mlp, "expert_gate", None) is not None
        or type(linear).__name__ not in ("MergedColumnParallelLinear", "AscendMergedColumnParallelLinear")
        or getattr(linear, "custom_op", None) is not None
        or getattr(linear, "tp_size", 1) != 1
        or type(scheme).__name__ != "AscendW8A8LinearMethod310"
        or getattr(norm, "bias", None) is not None
        or getattr(linear, "bias", None) is not None
        or scale is None
        or offset is None
        or get_tensor_model_parallel_world_size() != 1
        or x.device.type != "npu"
        or x.ndim != 2
        or x.shape[0] < PREFILL_MIN_TOKENS
        or x.shape[1] != HIDDEN_SIZE
        or x.shape != residual.shape
        or norm.weight.shape != (HIDDEN_SIZE,)
        or x.dtype != torch.float16
        or residual.dtype != x.dtype
        or norm.weight.dtype != x.dtype
        or residual.device != x.device
        or norm.weight.device != x.device
        or scale.device != x.device
        or offset.device != x.device
        or scale.shape != (HIDDEN_SIZE,)
        or offset.shape != (HIDDEN_SIZE,)
        or scale.dtype != torch.float32
        or offset.dtype != torch.int32
        or not is_forward_context_available()
        or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.NONE
    ):
        return None
    quantized, _, summed = torch_npu.npu_add_rms_norm_quant(
        x, residual, 1.0 + norm.weight, scale, offset,
        epsilon=norm.variance_epsilon, div_mode=False,
    )
    return quantized, summed
