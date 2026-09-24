"""Experimental final-token row selection, without touching attention/cache.

Only the real uncompiled BS1 DFlash prefill route may discard rows. Compiled
verification, prompt logprobs, auxiliary final-layer consumers and warmup keep
the original full-row path. Unused output rows are zero-filled, not undefined.
"""

import weakref

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_ascend import envs

PREFILL_MIN_TOKENS = 128


def prepare_last_prefill_mlp(model):
    if not envs.VLLM_ASCEND_LAST_PREFILL_MLP:
        return
    if envs.VLLM_ASCEND_LAST_PREFILL_NORMS and (
        envs.VLLM_ASCEND_PREFILL_MLP_NORM_QUANT or envs.VLLM_ASCEND_GEMMA_PREFILL_ADD_RMS_NORM
    ):
        raise ValueError("Selected prefill norms require the original split normalization, not experimental fusion")
    for backbone in model.modules():
        layers = getattr(backbone, "layers", None)
        config = getattr(backbone, "config", None)
        count = getattr(config, "num_hidden_layers", None)
        if layers is None or not isinstance(count, int) or count <= 0:
            continue
        if getattr(backbone, "start_layer", None) != 0 or getattr(backbone, "end_layer", None) != count:
            continue
        try:
            decoder = layers[count - 1]
        except (IndexError, KeyError, TypeError):
            continue
        if getattr(decoder, "layer_type", None) != "full_attention" or getattr(decoder, "layer_scale", None):
            continue
        if any(getattr(decoder, name, None) is None for name in ("mlp", "self_attn", "post_attention_layernorm")):
            continue
        # A weak reference avoids adding a recursive module/state_dict edge.
        decoder._last_prefill_mlp_backbone = weakref.ref(backbone)
        decoder.last_prefill_mlp_calls = 0
        decoder.last_prefill_mlp_skipped_rows = 0
        if envs.VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT:
            decoder.self_attn._last_prefill_mlp_backbone = weakref.ref(backbone)
            decoder.self_attn.last_prefill_attn_output_calls = 0
            decoder.self_attn.last_prefill_attn_output_skipped_rows = 0
        if envs.VLLM_ASCEND_LAST_PREFILL_NORMS:
            for role, norm in (("post_attention", decoder.post_attention_layernorm),
                               ("final", getattr(backbone, "norm", None))):
                if (norm is None or getattr(norm, "weight", None) is None
                        or getattr(norm, "variance_epsilon", None) is None):
                    continue
                norm._last_prefill_mlp_backbone = weakref.ref(backbone)
                norm.last_prefill_norm_role = role
                norm.last_prefill_norm_calls = 0
                norm.last_prefill_norm_skipped_rows = 0


def last_prefill_mlp_eligible(context, rows, aux_layers, final_capture_index):
    """Host-only runtime contract, deliberately false for absent evidence."""
    return (
        rows >= PREFILL_MIN_TOKENS
        and getattr(context, "last_prefill_mlp_allowed", False)
        and getattr(context, "skip_compiled", False)
        and not getattr(context, "in_profile_run", True)
        and not getattr(context, "capturing", True)
        and not getattr(context, "is_draft_model", True)
        and getattr(getattr(context, "cudagraph_runtime_mode", None), "name", None) == "NONE"
        and bool(aux_layers)
        and final_capture_index not in aux_layers
    )


def selected_mlp_output(mlp, x):
    last = mlp(x[-1:])
    output = torch.zeros_like(x)
    output[-1:] = last
    return output


def last_prefill_backbone(module, x):
    # FULL-profile Dynamo tracing must retain the unmodified MLP graph. Real
    # NONE prefill already bypasses this graph in the DFlash FDO dispatcher.
    if torch.compiler.is_compiling() or not envs.VLLM_ASCEND_LAST_PREFILL_MLP:
        return None
    ref = getattr(module, "_last_prefill_mlp_backbone", None)
    if ref is None or not is_forward_context_available():
        return None
    backbone = ref()
    if backbone is None or x.ndim != 2 or x.dtype != torch.float16 or x.device.type != "npu":
        return None
    context = get_forward_context()
    if not last_prefill_mlp_eligible(
        context, x.shape[0], backbone.aux_hidden_state_layers, backbone.end_layer
    ):
        return None
    # Reject padded shapes: the runner samples the last *actual* row.
    if x.shape[0] != getattr(context, "last_prefill_mlp_rows", None):
        return None
    return backbone


def maybe_last_prefill_mlp(decoder, x):
    if last_prefill_backbone(decoder, x) is None:
        return None
    output = selected_mlp_output(decoder.mlp, x)
    decoder.last_prefill_mlp_calls += 1
    decoder.last_prefill_mlp_skipped_rows += x.shape[0] - 1
    return output


def selected_attention_output(attention, x, gate):
    """Keep full QKV/attention/cache work; shorten only the pure output tail."""
    last = x[-1:]
    if gate is not None:
        last = last * torch.sigmoid(gate[-1:])
    last, _ = attention.o_proj(last)
    output = last.new_zeros((x.shape[0], last.shape[-1]))
    output[-1:] = last
    return output


def maybe_last_prefill_attention_output(attention, x, gate):
    if torch.compiler.is_compiling() or not envs.VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT:
        return None
    if gate is not None and (gate.shape != x.shape or gate.dtype != x.dtype or gate.device != x.device):
        return None
    if last_prefill_backbone(attention, x) is None:
        return None
    output = selected_attention_output(attention, x, gate)
    attention.last_prefill_attn_output_calls += 1
    attention.last_prefill_attn_output_skipped_rows += x.shape[0] - 1
    return output


def selected_norm_output(norm, norm_fn, x, residual):
    last, last_residual = norm_fn(x[-1:], residual[-1:], norm.weight, norm.variance_epsilon)
    output = torch.zeros_like(x)
    output_residual = torch.zeros_like(residual)
    output[-1:] = last
    output_residual[-1:] = last_residual
    return output, output_residual


def maybe_last_prefill_norm(norm, norm_fn, x, residual):
    if torch.compiler.is_compiling() or not envs.VLLM_ASCEND_LAST_PREFILL_NORMS:
        return None
    if (
        residual is None
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
        or last_prefill_backbone(norm, x) is None
    ):
        return None
    output = selected_norm_output(norm, norm_fn, x, residual)
    norm.last_prefill_norm_calls += 1
    norm.last_prefill_norm_skipped_rows += x.shape[0] - 1
    return output
