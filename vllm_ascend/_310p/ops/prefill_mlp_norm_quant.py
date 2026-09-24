"""Opt-in existing-CANN fusion before a compatible static-W8A8 MLP.

Not bitwise equivalent to split FP16 norm/quant. Keep decode and all other
consumers on the original path; in particular attention must not see INT8
hidden_states when it allocates its output using hidden_states.dtype.
"""

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend import envs

PREFILL_MIN_TOKENS = 128


def _mlp_norm_quant(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor,
    reciprocal: torch.Tensor, offset: torch.Tensor, fused_scale: torch.Tensor,
    fused_offset: torch.Tensor, epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This dispatch must remain opaque to Dynamo: one compiled graph covers
    # both prefill and decode. Never specialize on the warmup context/length.
    if (x.shape[0] >= PREFILL_MIN_TOKENS and is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE):
        quantized, _, summed = torch_npu.npu_add_rms_norm_quant(
            x, residual, gamma, fused_scale, fused_offset, epsilon=epsilon, div_mode=False)
        return quantized, summed
    # Exact original FP16 Gemma add + RMSNorm + static quant sequence. The
    # following W8A8 gate/up projection skips its own quantization for INT8.
    summed = x + residual
    normalized = torch_npu.npu_rms_norm(summed, gamma, epsilon)[0]
    quantized = torch_npu.npu_quantize(normalized, reciprocal, offset, torch.qint8, -1, False)
    return quantized, summed


def _mlp_norm_quant_fake(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor,
    reciprocal: torch.Tensor, offset: torch.Tensor, fused_scale: torch.Tensor,
    fused_offset: torch.Tensor, epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=torch.int8), torch.empty_like(residual)


def ensure_mlp_norm_quant_registered():
    if not hasattr(torch.ops.vllm, "prefill_mlp_norm_quant"):
        direct_register_custom_op(
            op_name="prefill_mlp_norm_quant", op_func=_mlp_norm_quant,
            mutates_args=[], fake_impl=_mlp_norm_quant_fake)


def prepare_mlp_norm_quant_params(layer):
    """Called by the 310P static scheme after loading; never cache activations."""
    reciprocal = getattr(layer, "aclnn_input_scale_reciprocal", None)
    input_offset = getattr(layer, "input_offset", None)
    aclnn_offset = getattr(layer, "aclnn_input_offset", None)
    if (
        not envs.VLLM_ASCEND_PREFILL_MLP_NORM_QUANT
        or getattr(layer, "params_dtype", None) != torch.float16
        or reciprocal is None
        or reciprocal.ndim != 1
        or reciprocal.shape[0] == 0
        or input_offset is None
        or input_offset.dtype != torch.int8
        or aclnn_offset is None
        or aclnn_offset.shape != reciprocal.shape
    ):
        return
    ensure_mlp_norm_quant_registered()
    # Preserve the model's FP16 reciprocal rounding, then promote for this API.
    layer.register_buffer("_prefill_norm_quant_scale", reciprocal.float(), persistent=False)
    layer.register_buffer("_prefill_norm_quant_offset", aclnn_offset.to(torch.int32), persistent=False)


def maybe_fused_mlp_norm_quant(decoder, x, residual):
    if not envs.VLLM_ASCEND_PREFILL_MLP_NORM_QUANT or residual is None:
        return None
    norm = decoder.post_attention_layernorm
    mlp = decoder.mlp
    linear = getattr(mlp, "gate_up_proj", None)
    scale = getattr(linear, "_prefill_norm_quant_scale", None)
    offset = getattr(linear, "_prefill_norm_quant_offset", None)
    scheme = getattr(getattr(linear, "quant_method", None), "quant_method", None)
    norm_weight = getattr(norm, "weight", None)
    epsilon = getattr(norm, "variance_epsilon", None)
    hidden_size = x.shape[1] if x.ndim == 2 else None
    if (
        linear is None
        or getattr(mlp, "expert_gate", None) is not None
        or getattr(linear, "custom_op", None) is not None
        or getattr(linear, "tp_size", 1) != 1
        or not getattr(scheme, "accepts_prequantized_input", False)
        or getattr(norm, "bias", None) is not None
        or getattr(linear, "bias", None) is not None
        or norm_weight is None
        or epsilon is None
        or scale is None
        or offset is None
        or get_tensor_model_parallel_world_size() != 1
        or x.device.type != "npu"
        or x.ndim != 2
        or x.shape != residual.shape
        or norm_weight.shape != (hidden_size,)
        or x.dtype != torch.float16
        or residual.dtype != x.dtype
        or norm_weight.dtype != x.dtype
        or residual.device != x.device
        or norm_weight.device != x.device
        or scale.device != x.device
        or offset.device != x.device
        or scale.shape != (hidden_size,)
        or offset.shape != (hidden_size,)
        or scale.dtype != torch.float32
        or offset.dtype != torch.int32
    ):
        return None
    return torch.ops.vllm.prefill_mlp_norm_quant(
        x, residual, 1.0 + norm_weight, linear.aclnn_input_scale_reciprocal,
        linear.aclnn_input_offset, scale, offset, epsilon,
    )
