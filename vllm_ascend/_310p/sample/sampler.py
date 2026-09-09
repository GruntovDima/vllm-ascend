#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language govserning permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import torch
import vllm.envs as envs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.sampler import (
    DEFAULT_LOGPROBS_MODE,
    AscendSampler,
    AscendTopKTopPSampler,
)
from vllm_ascend.utils import global_stream, npu_stream_switch

if TYPE_CHECKING:
    from vllm.v1.sample.metadata import SamplingMetadata

_CPU_GENERATOR_CACHE_310P: dict[int, tuple[torch.Generator, torch.Generator]] = {}

_TREE_COMPACT_MAX_TOP_K = 128
# Guard the change in FP32 summation order (ascending full CDF vs. head mass).
# Ambiguous rows use the original filter, not an adjusted sampling threshold.
_TREE_TOP_P_ROUNDING_GUARD = 1e-5


def _try_compact_top_k_310p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    top_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]] | None:
    """Return filtered candidate logits/IDs, or request the unchanged fallback.

    ``top_k`` is a CPU request parameter, not a device tensor scalar read.
    The pinned Ascend filter applies top-p to ORIGINAL vocabulary probabilities,
    independently of top-k, and retains all ties at either cutoff. In particular
    applying top-p to a renormalized top-k list is not equivalent.

    One batched D2H decision per tree step is intentional in this eager-only
    path: it avoids executing the full-vocabulary sort on unambiguous inputs.
    No logits or candidate arrays are transferred to the host.
    """
    if (type(top_k) is not int or not 0 < top_k <= _TREE_COMPACT_MAX_TOP_K
            or top_k >= logits.shape[-1] or logits.dtype != torch.float32 or k is None):
        return None
    probs = logits.softmax(dim=-1)
    # Extra capacity retains small cutoff-tie groups instead of sorting the
    # whole vocabulary. A final sentinel detects groups that still overflow.
    capacity = min(2 * top_k, logits.shape[-1] - 1)
    head, token_ids = probs.topk(capacity + 1, dim=-1)
    safe = (head[:, top_k - 1] > head[:, capacity]) & (k == top_k)
    # 310P isnan/isfinite do not reliably distinguish infinities; inspect bits.
    safe &= ((head.contiguous().view(torch.int32) & 0x7F800000) != 0x7F800000).all(dim=-1)
    head, token_ids = head[:, :capacity], token_ids[:, :capacity]
    cutoff = head[:, top_k - 1:top_k]
    if p is not None and top_k > 1:
        # For descending candidate j, mass through j in the ascending full
        # CDF is total - sum(head[:j]). The maximum is always retained.
        mass_before = torch.cat((torch.zeros_like(head[:, :1]), head[:, :-1].cumsum(-1)), dim=-1)
        ascending_mass = probs.sum(dim=-1, keepdim=True) - mass_before
        threshold = 1 - p.unsqueeze(-1)
        unambiguous = (ascending_mass[:, 1:] - threshold).abs() > _TREE_TOP_P_ROUNDING_GUARD
        safe &= (unambiguous | (head[:, 1:] < cutoff)).all(dim=-1)
        safe &= (p > 0) & (p <= 1)
        retained = ascending_mass > threshold
        retained[:, 0] = True
        p_cutoff = head.masked_fill(~retained, float("inf")).amin(dim=-1, keepdim=True)
        cutoff = torch.maximum(cutoff, p_cutoff)
    # Copy at most 65 booleans once, never one scalar per node. A boundary tie
    # in one hypothetical prefix must not force a vocabulary sort on all rows.
    safe_rows = safe.cpu().tolist()
    if not any(safe_rows):
        return None
    fallback_rows = [row for row, valid in enumerate(safe_rows) if not valid]
    # Compare probabilities, not logits: rounded probability ties are part of
    # the original filter's contract. Keep ID order for inverse-CDF sampling.
    candidate_logits = logits.gather(-1, token_ids).masked_fill(head < cutoff, -float("inf"))
    if fallback_rows:
        # These temporary samples will be replaced using the original filter
        # and the SAME uniforms. Keep their compact CDF finite in the meantime.
        candidate_logits.masked_fill_(~safe.unsqueeze(-1), 0)
    token_ids, order = token_ids.sort(dim=-1)
    return candidate_logits.gather(-1, order), token_ids, fallback_rows


def _prepare_cpu_generators_310p(
    generators: dict[int, torch.Generator],
) -> dict[int, torch.Generator]:
    """Return CPU RNGs while preserving requests across batch reordering."""
    cached_by_source = {
        id(source): (source, cpu_generator) for source, cpu_generator in _CPU_GENERATOR_CACHE_310P.values()
    }
    prepared: dict[int, torch.Generator] = {}
    next_cache: dict[int, tuple[torch.Generator, torch.Generator]] = {}

    for request_index, source in generators.items():
        cache_entry = cached_by_source.get(id(source))
        if cache_entry is None or cache_entry[0] is not source:
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(source.initial_seed())
        else:
            cpu_generator = cache_entry[1]

        prepared[request_index] = cpu_generator
        next_cache[request_index] = (source, cpu_generator)

    _CPU_GENERATOR_CACHE_310P.clear()
    _CPU_GENERATOR_CACHE_310P.update(next_cache)
    return prepared


def _generate_request_uniforms_310p(
    batch_size: int,
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Generate one uniform value per request on pinned CPU memory."""
    uniforms = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device="cpu",
        pin_memory=True,
    )
    for request_index, cpu_generator in _prepare_cpu_generators_310p(generators).items():
        uniforms[request_index] = torch.rand((), dtype=torch.float32, generator=cpu_generator)

    # Exact zero would select a zero-probability prefix in inverse CDF.
    uniforms.clamp_min_(torch.finfo(torch.float32).tiny)
    return uniforms.to(device, non_blocking=True)


def _sample_from_cdf_310p(
    weights: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Sample rows of non-negative weights with 310P-supported NPU ops."""
    cdf = weights.cumsum(dim=-1, dtype=torch.float32)
    thresholds = uniforms.unsqueeze(-1) * cdf[..., -1:]
    return torch.searchsorted(cdf, thresholds, right=True).squeeze(-1)


def fill_exponential_310p(
    reference: torch.Tensor,
    generators: dict[int, torch.Generator],
    active_mask: list[bool] | None = None,
) -> torch.Tensor:
    """Generate exponential values on CPU and transfer them to NPU."""
    batch_size = reference.shape[0]
    cpu_generators = _prepare_cpu_generators_310p(generators)
    needs_default_values = active_mask is not None or len(generators) != batch_size

    if needs_default_values:
        uniforms = torch.rand(
            reference.shape,
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
    else:
        uniforms = torch.empty(
            reference.shape,
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )

    for request_index, cpu_generator in cpu_generators.items():
        if active_mask is not None and not active_mask[request_index]:
            continue
        uniforms[request_index] = torch.rand(
            reference.shape[1:],
            dtype=torch.float32,
            generator=cpu_generator,
        )

    uniforms.clamp_min_(torch.finfo(torch.float32).tiny)
    exponential = -torch.log(uniforms)
    return exponential.to(
        device=reference.device,
        dtype=reference.dtype,
        non_blocking=True,
    )


def _request_uniforms_on_current_stream_310p(
    batch_size: int,
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Advance each request RNG once and order the small H2D copy."""
    with npu_stream_switch(global_stream()):
        uniforms = _generate_request_uniforms_310p(
            batch_size, generators, device,
        )
    torch.npu.current_stream().wait_stream(global_stream())
    return uniforms


def _random_sample_310p(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Generate only one CPU uniform per row; perform inverse CDF on NPU."""
    uniforms = _request_uniforms_on_current_stream_310p(probs.shape[0], generators, probs.device)
    sampled = _sample_from_cdf_310p(probs, uniforms)
    return sampled.view(-1)


class AscendTopKTopPSampler310(AscendTopKTopPSampler):
    def forward_native(self, logits, generators, k, p):
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)
        if get_ascend_config().enable_reduce_sample:
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = cand_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = cand_logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = torch.softmax(cand_logits, dim=-1)
            pos = _random_sample_310p(probs, generators)  # [B]

            next_token = cand_idx.gather(dim=1, index=pos.unsqueeze(1)).squeeze(1)  # [B]
            return next_token, logits_to_return
        else:
            # Only sample_tree installs a CPU top-k hint, scoped to this call.
            # Ordinary decode, graph execution and logprob outputs are unchanged.
            if self.logprobs_mode not in ("processed_logits", "processed_logprobs"):
                compact = _try_compact_top_k_310p(logits, k, p, getattr(self, "tree_top_k", None))
                if compact is not None:
                    candidate_logits, token_ids, fallback_rows = compact
                    probs = candidate_logits.softmax(dim=-1, dtype=torch.float32)
                    if not fallback_rows:
                        positions = _random_sample_310p(probs, generators)
                        return token_ids.gather(-1, positions.unsqueeze(-1)).squeeze(-1), None
                    fallback = torch.tensor(fallback_rows, dtype=torch.int64, device=logits.device)
                    fallback_logits = self.apply_top_k_top_p(
                        logits.index_select(0, fallback), k.index_select(0, fallback),
                        None if p is None else p.index_select(0, fallback),
                    )
                    fallback_probs = fallback_logits.softmax(dim=-1, dtype=torch.float32)
                    uniforms = _request_uniforms_on_current_stream_310p(len(logits), generators, logits.device)
                    positions = _sample_from_cdf_310p(probs, uniforms)
                    sampled = token_ids.gather(-1, positions.unsqueeze(-1)).squeeze(-1)
                    fallback_tokens = _sample_from_cdf_310p(fallback_probs, uniforms.index_select(0, fallback))
                    return sampled.scatter_(0, fallback, fallback_tokens), None
            logits = self.apply_top_k_top_p(logits, k, p)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return _random_sample_310p(probs, generators), logits_to_return


class AscendSampler310(AscendSampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler310(logprobs_mode=logprobs_mode)

    def sample_tree(
        self, logits: torch.Tensor, metadata: SamplingMetadata, *, top_k: int | None = None,
    ) -> torch.Tensor:
        """Sample every tree node with the existing target sampler on NPU.

        The caller validates a single request without history-dependent
        processors. Each row is a hypothetical prefix, not a new request.
        A seeded request uses one advancing RNG shared across rows, NOT a
        generator re-seeded at each node. Prime the existing 310P CPU RNG cache
        before aliasing its source across rows, including on a root-only first
        step. Only scalar uniforms are generated on CPU; logits, filtering,
        probabilities and inverse-CDF selection stay on NPU.

        Unvisited rows also consume draws. Fixed seeds reproduce sampling for
        fixed logits/topology, but are not token-identical to serial AR or to
        another tree width. No new process-global RNG state is introduced.
        """
        if logits.ndim != 2 or logits.shape[0] < 1:
            raise ValueError("Tree sampling requires nonempty [nodes, vocab] logits")
        if metadata.all_greedy:
            return self.greedy_sample(logits)
        if not metadata.all_random:
            raise ValueError("Tree sampling requires one request, not mixed sampling modes")
        if set(metadata.generators) - {0}:
            raise ValueError("Tree sampling requires a single request RNG at index zero")
        holder = getattr(metadata, "thinking_budget_state_holder", None)
        if holder is not None and holder.has_tracked_requests():
            raise ValueError("Tree sampling does not support thinking-token budgets")

        expanded = copy(metadata)
        for name in ("temperature", "top_k", "top_p"):
            value = getattr(metadata, name)
            if value is not None:
                if value.ndim != 1 or value.numel() != 1:
                    raise ValueError(f"Tree sampling requires one {name} value")
                setattr(expanded, name, value.expand(logits.shape[0]))
        _prepare_cpu_generators_310p(metadata.generators)
        expanded.generators = (
            {node: metadata.generators[0] for node in range(logits.shape[0])}
            if metadata.generators else {}
        )
        previous_top_k = getattr(self.topk_topp_sampler, "tree_top_k", None)
        self.topk_topp_sampler.tree_top_k = top_k
        try:
            return self.forward(logits, expanded).sampled_token_ids.flatten()
        finally:
            self.topk_topp_sampler.tree_top_k = previous_top_k
            # Restore request-indexed cache ownership after temporary row
            # expansion, retaining the RNG state advanced by the sampler.
            _prepare_cpu_generators_310p(metadata.generators)
