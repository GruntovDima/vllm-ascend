import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.layernorm import RMSNormGated

from vllm_ascend import envs
from vllm_ascend._310p.ops.adn_rms_norm import adn_rms_norm_or_fallback
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm

_GEMMA_ADN_MIN_PREFILL_TOKENS = 128


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


class AscendGemmaRMSNorm310(AscendGemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            orig_dtype = residual.dtype
            x = x + residual.to(x.dtype)
            residual = x.to(orig_dtype)
            x = gemma_rms_norm_310(x, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x = gemma_rms_norm_310(x, 1.0 + self.weight, self.variance_epsilon)
        return x


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
