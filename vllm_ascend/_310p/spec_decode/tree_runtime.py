# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in contracts for the eager, batch-one 310P comb-tree reference path.

All step state is owned by the runner and attached to target-layer metadata.
The drafter never receives this context. No process-global tree/cache state is
used. Native kernels are reused; persistent GDN state rounds to FP16 per edge.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vllm_ascend._310p.spec_decode.tree import TokenTree, TreeVerification


@dataclass(frozen=True)
class TreeMTPConfig:
    width: int = 2
    depth: int = 2
    trace: bool = False

    @property
    def num_candidates(self) -> int:
        return self.width * self.depth

    @classmethod
    def from_vllm_config(cls, config: Any) -> "TreeMTPConfig | None":
        raw = (config.additional_config or {}).get("tree_mtp")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("additional_config.tree_mtp must be an object")
        if set(raw) - {"enabled", "width", "depth", "trace"}:
            raise ValueError("Unknown tree_mtp configuration field")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("tree_mtp.enabled must be boolean")
        if not enabled:
            return None
        tree = cls(raw.get("width", 2), raw.get("depth", 2), raw.get("trace", False))
        if type(tree.width) is not int or type(tree.depth) is not int or not isinstance(tree.trace, bool):
            raise ValueError("tree_mtp width/depth must be integers and trace must be boolean")
        if tree.width not in (1, 2, 4, 8, 16) or not 1 <= tree.depth <= 4:
            raise ValueError("310P tree-MTP requires width in {1,2,4,8,16} and depth in [1,4]")
        spec = config.speculative_config
        if spec is None or spec.method != "mtp" or spec.num_speculative_tokens != tree.num_candidates:
            raise ValueError("tree_mtp requires MTP with num_speculative_tokens=width*depth")
        if not config.model_config.enforce_eager or config.scheduler_config.max_num_seqs != 1:
            raise ValueError("tree_mtp requires enforce_eager and max_num_seqs=1")
        parallel = config.parallel_config
        if any(getattr(parallel, name, 1) != 1 for name in (
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size"
        )):
            raise ValueError("tree_mtp currently requires TP=PP=DP=DCP=1")
        if config.scheduler_config.async_scheduling or config.cache_config.enable_prefix_caching:
            raise ValueError("tree_mtp requires synchronous scheduling and disabled prefix caching")
        if config.lora_config is not None or getattr(config, "kv_transfer_config", None) is not None:
            raise ValueError("tree_mtp does not support LoRA or KV transfer")
        if getattr(config.model_config, "logits_processors", None):
            raise ValueError("tree_mtp does not support custom logits processors")
        model_type = getattr(config.model_config.hf_text_config, "model_type", "")
        if model_type not in ("qwen3_5_text", "qwen3_5"):
            raise ValueError("tree_mtp is currently validated only for Qwen3.5 dense models")
        if config.cache_config.mamba_cache_mode != "none":
            raise ValueError("tree_mtp currently requires mamba_cache_mode='none'")
        if getattr(spec, "draft_sample_method", "greedy") != "greedy":
            raise ValueError("tree_mtp requires greedy draft sampling")
        if (config.additional_config or {}).get("enable_reduce_sample", False):
            raise ValueError("tree_mtp requires full-vocabulary draft logits")
        if getattr(spec, "use_local_argmax_reduction", False):
            raise ValueError("tree_mtp does not support local argmax drafting")
        if getattr(spec, "parallel_drafting", False):
            raise ValueError("tree_mtp requires sequential primary-backbone drafting")
        return tree


def validate_tree_sampling(params: Any) -> None:
    """Reject processors whose histories would include flattened siblings."""
    if params.n != 1:
        raise ValueError("tree_mtp currently supports n=1 only")
    neutral = {"repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
    if any(getattr(params, name, value) != value for name, value in neutral.items()):
        raise ValueError("tree_mtp does not yet support sampling penalties")
    if params.logprobs is not None or params.prompt_logprobs is not None:
        raise ValueError("tree_mtp does not support logprobs")
    unsupported = ("structured_outputs", "allowed_token_ids",
                   "logit_bias", "bad_words", "logits_processors", "logprob_token_ids")
    if any(getattr(params, name, None) for name in unsupported):
        raise ValueError("tree_mtp does not support constrained sampling or logprobs")
    if getattr(params, "min_tokens", 0):
        raise ValueError("tree_mtp currently requires min_tokens=0")
    if getattr(params, "min_p", 0):
        raise ValueError("tree_mtp currently requires min_p=0; use top_k/top_p")
    if getattr(params, "thinking_token_budget", None) is not None:
        raise ValueError("tree_mtp does not support a thinking-token budget")


@dataclass
class TreeStepContext:
    tree: TokenTree
    prefix_length: int
    request_id: str = ""
    max_output_tokens: int | None = None
    accepted_input_indices: tuple[int, ...] = ()
    emitted_token_ids: tuple[int, ...] = ()
    committed: bool = False
    _commit_callbacks: dict[Any, Callable[[tuple[int, ...]], None]] = field(default_factory=dict, repr=False)
    gdn_state_indices: dict[str, Any] = field(default_factory=dict, repr=False)
    previous_accepted_tokens: Any = field(default=None, repr=False)
    committed_cache_kinds: tuple[str, ...] = ()
    # Shared read-only splitfuse inputs for this verification step, not a
    # prefix/KV cache. The attention backend keys these by device and topology.
    attention_inputs: dict[Any, tuple[Any, Any, Any]] = field(default_factory=dict, repr=False)

    @property
    def num_nodes(self) -> int:
        return self.tree.num_nodes

    def has_commit(self, key: Any) -> bool:
        return key in self._commit_callbacks

    def register_commit(self, key: Any, callback: Callable[[tuple[int, ...]], None]) -> None:
        if self.committed:
            raise RuntimeError("Cannot register a cache after tree commit")
        if key in self._commit_callbacks:
            raise RuntimeError(f"Duplicate tree cache writer: {key!r}")
        self._commit_callbacks[key] = callback

    def commit(self, result: TreeVerification) -> None:
        if self.committed:
            raise RuntimeError("A tree step may only commit once")
        path = result.accepted_input_indices
        if not path or self.tree.ancestor_indices(path[-1]) != path:
            raise ValueError("Tree commit must be a nonempty root-to-node path")
        if len(path) != len(result.emitted_token_ids):
            raise ValueError("Committed inputs and emitted output lengths must agree")
        for callback in self._commit_callbacks.values():
            callback(path)
        self.accepted_input_indices = path
        self.emitted_token_ids = result.emitted_token_ids
        self.committed = True
        self.committed_cache_kinds = tuple(str(key[0]) for key in self._commit_callbacks)
        self._commit_callbacks.clear()
        self.attention_inputs.clear()
