import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.layernorm import RMSNormGated

from vllm_ascend import envs
from vllm_ascend._310p.ops.last_prefill_mlp import maybe_last_prefill_norm
from vllm_ascend._310p.ops.adn_rms_norm import adn_rms_norm_or_fallback
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm

_GEMMA_ADN_MIN_PREFILL_TOKENS = 128
_GEMMA_ADD_RMS_MIN_PREFILL_TOKENS = 128
_GEMMA_ADD_RMS_HIDDEN_SIZE = 4096


def use_gemma_prefill_add_rms_norm(x, residual, weight):
    # Startup-only experimental route. Keep graph/decode and Q/K norms on
    # their existing paths. The fused normalization is not bitwise identical.
    return (
        envs.VLLM_ASCEND_GEMMA_PREFILL_ADD_RMS_NORM
        and residual is not None
        and x.dim() == 2
        and x.shape[0] >= _GEMMA_ADD_RMS_MIN_PREFILL_TOKENS
        and x.shape[-1] == _GEMMA_ADD_RMS_HIDDEN_SIZE
        and residual.shape == x.shape
        and weight.shape == (x.shape[-1],)
        and x.dtype == residual.dtype == weight.dtype == torch.float16
        and is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
    )


def gemma_rms_norm_310(x: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> torch.Tensor:
    # Q/K Gemma norms have [tokens, heads, head_dim] input. Keep short
    # decode/verification shapes and hidden-state norms on their original path.
    if (
        envs.VLLM_ASCEND_GEMMA_PREFILL_ADN
        and x.dim() == 3
        and x.shape[0] >= _GEMMA_ADN_MIN_PREFILL_TOKENS
        and x.shape[-1] == 256
    ):
        return adn_rms_norm_or_fallback(x.contiguous(), gamma, epsilon)
    return torch_npu.npu_rms_norm(x, gamma, epsilon)[0]


class AscendRMSNorm310(AscendRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if self.bias is not None:
                x.add_(self.bias)
            return x, residual

        x = adn_rms_norm_or_fallback(
            x,
            self.weight,
            self.variance_epsilon,
        )
        if self.bias is not None:
            x.add_(self.bias)
        return x


def gemma_residual_rms_norm_310(x, residual, weight, epsilon):
    """The original split arithmetic, shared by full-row and selected-row paths."""
    if residual is not None:
        orig_dtype = residual.dtype
        x = x + residual.to(x.dtype)
        residual = x.to(orig_dtype)
        x = gemma_rms_norm_310(x, 1.0 + weight, epsilon)
        return x, residual
    return gemma_rms_norm_310(x, 1.0 + weight, epsilon)


class AscendGemmaRMSNorm310(AscendGemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        selected = maybe_last_prefill_norm(self, gemma_residual_rms_norm_310, x, residual)
        if selected is not None:
            return selected
        if use_gemma_prefill_add_rms_norm(x, residual, self.weight):
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, 1.0 + self.weight, self.variance_epsilon
            )
            return x, residual
        return gemma_residual_rms_norm_310(x, residual, self.weight, self.variance_epsilon)


class AscendRMSNormGated310(RMSNormGated):
    def _apply_activation(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(z)
        if self.activation in ("silu", "swish"):
            return F.silu(z)
        raise AssertionError(f"Unsupported activation: {self.activation}")

    def forward_oot(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.group_size is not None:
            return super().forward_native(x, z)

        if z is not None and not self.norm_before_gate:
            x = torch.mul(x, self._apply_activation(z))

        x = adn_rms_norm_or_fallback(
            x,
            self.weight,
            self.eps,
        )

        if z is not None and self.norm_before_gate:
            x = torch.mul(x, self._apply_activation(z))

        return x
