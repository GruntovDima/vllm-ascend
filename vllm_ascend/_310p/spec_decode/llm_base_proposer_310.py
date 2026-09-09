#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

_original_run_merged_draft = AscendSpecDecodeBaseProposer._run_merged_draft
_original_propose = AscendSpecDecodeBaseProposer._propose
_original_sample_draft_from_logits = AscendSpecDecodeBaseProposer._sample_draft_from_logits


def _tree_topk_token_ids(logits: torch.Tensor, width: int) -> torch.Tensor:
    """Distinct siblings ordered by score, breaking ties by lowest token ID.

    ``topk`` does not guarantee stable tie ordering. Repeated reductions avoid
    sorting the whole vocabulary and keep the first sibling equal to greedy
    argmax. The availability mask also handles rows containing only -inf.
    No logits or sampling buffers owned by the caller are modified.
    """
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("Tree MTP requires floating-point [batch, vocab] logits")
    if width < 1 or width > logits.shape[-1]:
        raise ValueError("Tree MTP width must be between 1 and vocabulary size")
    if width == 1:
        return logits.argmax(dim=-1, keepdim=True)

    available = torch.ones_like(logits, dtype=torch.bool)
    selected = []
    for _ in range(width):
        scores = logits.masked_fill(~available, -torch.inf)
        maximum = scores.max(dim=-1, keepdim=True).values
        # Preserve argmax's first-NaN behavior without selecting a used ID.
        ties = (scores == maximum) | (scores.isnan() & maximum.isnan())
        token_ids = (ties & available).to(torch.int32).argmax(dim=-1)
        selected.append(token_ids)
        available.scatter_(1, token_ids.unsqueeze(-1), False)
    return torch.stack(selected, dim=-1)


class AscendSpecDecodeBaseProposer310(AscendSpecDecodeBaseProposer):
    """310P proposer overrides for NPU-specific spec-decode workarounds."""

    @staticmethod
    def _scale_block_ids_for_slot_mapping(
        block_ids: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Scale block ids without 310P's unstable tiny int32 Mul path."""
        # add(x, x, alpha=n-1) is exactly n*x and lowers to AxpyV2.
        return torch.add(block_ids, block_ids, alpha=block_size - 1)

    def _propose(self, num_speculative_tokens: int, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Draft a comb tree using the existing primary-token backbone.

        The runner attaches ``tree_mtp_config`` only after validating the
        eager tree path. Target sampling may be random; proposal construction
        remains deterministic and must not consume the target request RNG.
        The public budget counts candidate nodes;
        the native proposer instead receives the number of backbone levels.
        Its KV writes remain a linear backbone. The runner must rebuild the
        next first-pass inputs from the accepted target path, not its prefix.
        """
        tree_config = getattr(self, "tree_mtp_config", None)
        if tree_config is None or num_speculative_tokens == 0:
            return _original_propose(self, num_speculative_tokens, *args, **kwargs)

        width, depth = tree_config.width, tree_config.depth
        if (
            isinstance(width, bool)
            or isinstance(depth, bool)
            or not isinstance(width, int)
            or not isinstance(depth, int)
            or width < 1
            or depth < 1
        ):
            raise ValueError("Tree MTP width and depth must be positive integers")
        if num_speculative_tokens != width * depth:
            raise ValueError("Tree MTP candidate budget must equal width * depth")
        if getattr(self, "_enable_probabilistic_draft_probs", False):
            raise ValueError("Tree MTP supports only greedy drafting")
        if getattr(self, "_tree_mtp_sibling_ids", None) is not None:
            raise RuntimeError("Nested tree MTP proposal is not supported")

        previous_num_speculative_tokens = self.num_speculative_tokens
        self._tree_mtp_sibling_ids: list[torch.Tensor] | None = []
        self._tree_mtp_collect_width = width
        try:
            backbone = _original_propose(self, depth, *args, **kwargs)
            siblings = self._tree_mtp_sibling_ids
            if len(siblings) != depth:
                raise RuntimeError(
                    "Tree MTP needs one full-logits sampling call per backbone level; "
                    "disable reduced-vocabulary/local-argmax and parallel drafting"
                )
            if any(row.shape != (backbone.shape[0], width) for row in siblings):
                raise RuntimeError("Tree MTP sibling rows do not match the proposer batch")
            # [batch, depth, width] -> [batch, candidate nodes]. Siblings at
            # level d share the first sibling from level d-1 as their parent.
            return torch.stack(siblings, dim=1).flatten(start_dim=1).contiguous()
        finally:
            self._tree_mtp_sibling_ids = None
            self._tree_mtp_collect_width = 0
            self._last_draft_probs = None
            self.num_speculative_tokens = previous_num_speculative_tokens

    def _sample_draft_from_logits(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        siblings = getattr(self, "_tree_mtp_sibling_ids", None)
        if siblings is None:
            return _original_sample_draft_from_logits(self, logits, sampling_metadata)
        # Target temperature/top-k/top-p do not turn the deterministic MTP
        # proposals into samples. All randomness belongs to target verification.
        token_ids = _tree_topk_token_ids(logits, self._tree_mtp_collect_width)
        siblings.append(token_ids)
        # Only this primary candidate enters the next native MTP iteration.
        return token_ids[:, 0], None

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            result = _original_run_merged_draft(
                self,
                num_input_tokens,
                batch_size,
                token_indices_to_sample,
                target_positions,
                inputs_embeds,
                multi_steps_attn_metadata,
                num_tokens,
                is_prefill,
            )
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)
        return result

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        if not self.needs_extra_input_slots:
            # 310P workaround for MTP:
            # The NPU implementation of the slice assign
            #   self.input_ids[:num_tokens-1] = target_token_ids[1:]
            # can corrupt the tail element (index num_tokens-1) of the
            # persistent drafter input_ids buffer. We save/restore it to
            # avoid feeding garbage to the draft model or later GatherV2.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]

            # Protected shift (310P specific)
            tail_save = self.input_ids[num_tokens - 1].clone()
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            self.input_ids[num_tokens - 1] = tail_save

            # Replace the last token with the next token.
            self.input_ids[token_indices_to_sample] = next_token_ids

            assert self.runner is not None

            # 310P does not support DCP, so skip context-parallel handling.
            ori_token_indices_to_sample = None
            query_lens_d = None

            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]

            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states.view(num_tokens, -1)

            return num_tokens, token_indices_to_sample, cad, (query_lens_d, ori_token_indices_to_sample)
        return super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
