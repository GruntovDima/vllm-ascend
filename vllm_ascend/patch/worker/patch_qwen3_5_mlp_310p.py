"""Opt-in MLP-only patch for the upstream Qwen3.5 path used by 310P.

Do not import patch_qwen3_5: its attention replacements are not the 310P
integration path. Apart from the post-attention norm, this forward matches
Qwen3NextDecoderLayer.forward in the supported vLLM 0.24 checkout.
"""

import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_ascend import envs
from vllm_ascend._310p.ops.last_prefill_mlp import (
    maybe_last_prefill_attention_output,
    maybe_last_prefill_mlp,
)
from vllm_ascend._310p.ops.prefill_mlp_norm_quant import maybe_fused_mlp_norm_quant
from vllm_ascend.utils import vllm_version_is


def mlp_norm_quant_decoder_forward(self, hidden_states, residual, positions=None, **kwargs):
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    self_attention_output = torch.empty_like(hidden_states)
    if self.layer_type == "linear_attention":
        self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
    elif self.layer_type == "full_attention":
        self.self_attn(hidden_states=hidden_states, output=self_attention_output, positions=positions)
    else:
        raise ValueError("Invalid layer_type")
    hidden_states = self_attention_output

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
        else:
            hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

    fused_norm = maybe_fused_mlp_norm_quant(self, hidden_states, residual)
    if fused_norm is None:
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    else:
        hidden_states, residual = fused_norm
    selected_output = maybe_last_prefill_mlp(self, hidden_states)
    hidden_states = self.mlp(hidden_states) if selected_output is None else selected_output

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
        else:
            assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                f"shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}"
            )
            hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype) + 1)
    return hidden_states, residual


def last_prefill_attention_forward(self, positions, output, hidden_states):
    """Upstream v0.24 attention forward with an opt-in output-tail dispatch."""
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v, gate = self._project_qkv_gate(qkv, positions)
    attn_output = self.attn(q, k, v)
    selected_output = maybe_last_prefill_attention_output(self, attn_output, gate)
    if selected_output is None:
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        selected_output, _ = self.o_proj(attn_output)
    output[:] = selected_output


if envs.VLLM_ASCEND_PREFILL_MLP_NORM_QUANT or envs.VLLM_ASCEND_LAST_PREFILL_MLP:
    if not vllm_version_is("0.24.0"):
        raise RuntimeError("Prefill MLP norm-quant integration is verified only with the vLLM 0.24 forward API")
    Qwen3_5DecoderLayer.forward = mlp_norm_quant_decoder_forward

if envs.VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT:
    if not envs.VLLM_ASCEND_LAST_PREFILL_MLP:
        raise RuntimeError("Selected prefill attention output requires VLLM_ASCEND_LAST_PREFILL_MLP=1")
    if not vllm_version_is("0.24.0"):
        raise RuntimeError("Prefill attention output selection is verified only with the vLLM 0.24 forward API")
    Qwen3NextAttention.forward = last_prefill_attention_forward
