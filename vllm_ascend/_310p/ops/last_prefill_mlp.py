"""Experimental last-layer MLP row selection, without touching attention/cache.

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
    for backbone in model.modules():
        if type(backbone).__name__ != "Qwen3_5Model":
            continue
        count = backbone.config.num_hidden_layers
        if backbone.start_layer != 0 or backbone.end_layer != count:
            continue
        decoder = backbone.layers[count - 1]
        if decoder.layer_type != "full_attention" or decoder.layer_scale:
            continue
        if type(decoder.mlp).__name__ != "Qwen2MoeMLP":
            continue
        # A weak reference avoids adding a recursive module/state_dict edge.
        decoder._last_prefill_mlp_backbone = weakref.ref(backbone)
        decoder.last_prefill_mlp_calls = 0
        decoder.last_prefill_mlp_skipped_rows = 0


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


def maybe_last_prefill_mlp(decoder, x):
    # FULL-profile Dynamo tracing must retain the unmodified MLP graph. Real
    # NONE prefill already bypasses this graph in the DFlash FDO dispatcher.
    if torch.compiler.is_compiling() or not envs.VLLM_ASCEND_LAST_PREFILL_MLP:
        return None
    ref = getattr(decoder, "_last_prefill_mlp_backbone", None)
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
    output = selected_mlp_output(decoder.mlp, x)
    decoder.last_prefill_mlp_calls += 1
    decoder.last_prefill_mlp_skipped_rows += x.shape[0] - 1
    return output
