"""Opt-in scalar parameter broadcast for existing CANN static quantization.

Only the parameter representation changes. Keep the original FP16 reciprocal
rounding, multiply mode and integer offset; do not recompute a reciprocal from
a Python float. Widths are restricted to the measured beneficial shapes.
"""

import torch

from vllm_ascend import envs

SCALAR_QUANT_WIDTHS = (4096, 8192)


def prepare_scalar_quant_params(layer):
    """Post-load preparation only; reset the guard on every weight reload."""
    layer._scalar_quant_ready = False
    if not envs.VLLM_ASCEND_STATIC_SCALAR_QUANT:
        return
    params = (layer.aclnn_input_scale, layer.aclnn_input_scale_reciprocal, layer.aclnn_input_offset)
    width = params[0].numel()
    if width not in SCALAR_QUANT_WIDTHS:
        return
    for param in params:
        if (param.dtype != torch.float16 or param.shape != (width,)
                or param.device != params[0].device or not param.is_contiguous()):
            return
        # Device-to-host boolean checks are deliberate here, before compilation
        # and measurement. Forward contains no item()/equal()/synchronization.
        if not bool(torch.isfinite(param).all()) or not torch.equal(param, param[:1].expand_as(param)):
            return
    if not bool((params[0] > 0).all()) or not bool((params[1] > 0).all()):
        return
    for name, param in zip(("scale", "reciprocal", "offset"), params):
        layer.register_buffer(f"_scalar_quant_{name}", param[:1].detach().clone(), persistent=False)
    layer._scalar_quant_ready = True


def static_quant_params(layer, x):
    if (getattr(layer, "_scalar_quant_ready", False) and x.dtype == torch.float16
            and x.ndim == 2 and x.shape[-1] == layer.aclnn_input_scale.numel()):
        return layer._scalar_quant_scale, layer._scalar_quant_reciprocal, layer._scalar_quant_offset
    return layer.aclnn_input_scale, layer.aclnn_input_scale_reciprocal, layer.aclnn_input_offset
