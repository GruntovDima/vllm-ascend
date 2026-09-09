# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Eager tree GDN with an optional single-launch NPU tree recurrence.

Every edge crosses an FP16 recurrent-state checkpoint, as in ordinary decode.
The linear speculative kernel instead carries FP32 state within its sequence;
that different grouping must not be used as a bitwise reference for this path.
Native caches are read-only until the verifier commits one ancestor path.
Convolution histories are gathered for the entire tree in one batch. Recurrent
dependencies use either the fused tree kernel or a depth-wise reference.
No chunk/compute_wy arithmetic is changed.
"""

from collections.abc import Callable
from typing import Any

import torch


def use_compact_tree_gdn(num_nodes: int, value_heads: int) -> bool:
    """Measured batch-one floor; tensor shapes only, no device synchronization."""
    return value_heads == 32 and 13 <= num_nodes <= 65


def _tree_conv_history_indices(parents: tuple[int, ...], history_length: int) -> tuple[tuple[int, ...], ...]:
    """Index [initial history, raw nodes], never convolved sibling outputs."""
    histories = []
    for node, parent in enumerate(parents):
        if node == 0:
            if parent != -1:
                raise ValueError("Tree convolution requires a root with parent -1")
            histories.append(tuple(range(history_length)))
        else:
            if not 0 <= parent < node:
                raise ValueError("Tree convolution requires parent-before-child order")
            histories.append((*histories[parent][1:], history_length + parent))
    return tuple(histories)


def _copy_node_rows(destination: torch.Tensor, nodes: tuple[int, ...], source: torch.Tensor) -> None:
    """Basic slices use NPU copies, not the AI_CPU advanced-index writer.

    Comb-tree levels are contiguous. The row-wise case also supports other
    parent-before-child orderings without assuming breadth-first numbering.
    """
    if nodes == tuple(range(nodes[0], nodes[0] + len(nodes))):
        destination[nodes[0] : nodes[-1] + 1].copy_(source)
    else:
        for row, node in enumerate(nodes):
            destination[node : node + 1].copy_(source[row : row + 1])


def _check_accepted_path(parents: tuple[int, ...], path: tuple[int, ...]) -> None:
    previous = -1
    for node in path:
        if not isinstance(node, int) or isinstance(node, bool) or not 0 <= node < len(parents):
            raise ValueError("Invalid GDN tree commit node")
        if parents[node] != previous:
            raise ValueError("GDN tree commit must be an unbroken path starting at the root")
        previous = node


def _commit_tree_states(
    *,
    parents: tuple[int, ...],
    path: tuple[int, ...],
    state_indices: torch.Tensor,
    recurrent_cache: torch.Tensor,
    recurrent_snapshots: torch.Tensor | None,
    conv_cache: torch.Tensor,
    initial_conv_history: torch.Tensor,
    raw_inputs: torch.Tensor,
    cache_update_fn: Callable,
    replay_fn: Callable | None = None,
) -> None:
    """Compact checkpoints and the convolution tape into native MTP layout."""
    _check_accepted_path(parents, path)
    if not path:
        return
    selected = torch.tensor(path, dtype=torch.long, device=raw_inputs.device)
    accepted_count = len(path)
    if replay_fn is not None:
        accepted_states = replay_fn(path)
    elif recurrent_snapshots is not None:
        accepted_states = recurrent_snapshots.index_select(0, selected)
    else:
        raise ValueError("Tree GDN commit has neither snapshots nor compact replay")
    # Native next-step accepted_count selects checkpoint accepted_count - 1.
    cache_update_fn(
        recurrent_cache,
        state_indices[:accepted_count].to(torch.int32).reshape(-1, 1),
        accepted_states,
    )

    # Native speculative conv reads [accepted_count-1 : accepted_count+W-2].
    # Keeping W-2 old tokens before all accepted inputs gives exactly that
    # rolling W-1 history, even when the accepted nodes are not a flat prefix.
    tape = torch.cat((initial_conv_history[1:], raw_inputs.index_select(0, selected)), dim=0)
    committed_conv = torch.zeros_like(conv_cache[:1])
    committed_conv[0, : tape.shape[0]].copy_(tape)
    cache_update_fn(conv_cache, state_indices[:1].to(torch.int32).reshape(-1, 1), committed_conv)


def forward_tree_gdn(
    layer: Any,
    attn_metadata: Any,
    context: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    *,
    gating_fn: Callable,
    recurrent_fn: Callable,
    tree_recurrent_fn: Callable | None = None,
    compact_recurrent_fn: Callable | None = None,
    conv_fn: Callable | None = None,
    cache_update_fn: Callable | None = None,
) -> None:
    """Batch convolution across all nodes; recurrent work obeys tree depth.

    The caller must restrict this reference implementation to one eager text
    request. Dependencies are explicit to keep its CPU oracle tests independent
    of vLLM registration and the NPU runtime.
    """
    num_nodes = context.num_nodes
    if tree_recurrent_fn is not None and compact_recurrent_fn is not None:
        raise ValueError("Select exactly one tree-GDN kernel implementation")
    parents = context.tree.parents
    depths = context.tree.depths
    is_spec = attn_metadata.num_spec_decodes == 1 and attn_metadata.num_decodes == 0
    is_root_only = num_nodes == 1 and attn_metadata.num_spec_decodes == 0 and attn_metadata.num_decodes == 1
    if attn_metadata.num_prefills != 0 or attn_metadata.num_actual_tokens != num_nodes or not (is_spec or is_root_only):
        raise ValueError("Tree GDN requires a single unpadded decode request")
    if is_spec:
        indices = attn_metadata.spec_state_indices_tensor
        if indices is None or indices.ndim != 2 or indices.shape[0] != 1 or indices.shape[1] < num_nodes:
            raise ValueError("Tree GDN requires one native state checkpoint per scheduled node")
        state_indices = indices[0].to(dtype=torch.long).clone()
    else:
        indices = attn_metadata.non_spec_state_indices_tensor
        if indices is None or indices.ndim != 1 or indices.numel() != 1:
            raise ValueError("Root-only tree GDN requires one native state index")
        state_indices = indices.to(dtype=torch.long).clone()

    # A scheduler-truncated step may expose fewer checkpoint columns than the
    # previous accepted count. Retain the previous table, not its activations,
    # in the request-scoped context to select the correct committed checkpoint.
    initial_indices = context.gdn_state_indices.get(layer.prefix)
    accepted = attn_metadata.num_accepted_tokens if is_spec else context.previous_accepted_tokens
    if initial_indices is None:
        initial_indices = state_indices
        if is_root_only:
            # First decode after prefill has exactly one populated checkpoint.
            accepted = torch.ones(1, dtype=torch.int32, device=mixed_qkv.device)
    if accepted is None or accepted.numel() != 1:
        raise ValueError("Tree GDN requires the previous accepted-token count")

    conv_cache, recurrent_cache = layer.kv_cache[:2]
    conv_weights = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)).transpose(0, 1)
    width, channels = conv_weights.shape
    # The native speculative convolution tape contract is implemented for W=4.
    if width != 4 or conv_cache.ndim != 3 or conv_cache.shape[1] < width - 2 + num_nodes:
        raise ValueError("Tree GDN requires the native width-4 speculative convolution cache")
    if recurrent_cache.dtype != torch.float16 or conv_cache.shape[2] != channels:
        raise ValueError("Tree GDN requires FP16 recurrent checkpoints and matching convolution channels")
    if mixed_qkv.shape != (num_nodes, channels):
        raise ValueError("Tree GDN input must contain exactly the scheduled tree nodes")

    previous_checkpoint = accepted.to(dtype=torch.long).reshape(1) - 1
    # Device-side indexing avoids a separate NPU-to-CPU sync for every layer.
    initial_index = initial_indices.index_select(0, previous_checkpoint)
    initial_recurrent = recurrent_cache.index_select(0, initial_index)
    history_indices = torch.arange(width - 1, device=mixed_qkv.device) + previous_checkpoint
    initial_history = conv_cache.index_select(0, initial_indices[:1])[0].index_select(0, history_indices)
    raw_inputs = mixed_qkv.clone()
    recurrent_snapshots = (
        recurrent_cache.new_empty((num_nodes, *recurrent_cache.shape[1:]))
        if compact_recurrent_fn is None else None
    )
    g, beta = gating_fn(layer.A_log, a, b, layer.dt_bias)
    if conv_fn is None:
        conv_fn = torch.ops._C_ascend.npu_causal_conv1d_310
    if cache_update_fn is None:
        cache_update_fn = torch.ops.npu.npu_scatter_nd_update_

    # Causal convolution stores raw inputs, so all parent histories are known
    # before any convolution runs. Gather independent writable copies once.
    history_rows = torch.tensor(
        _tree_conv_history_indices(parents, width - 1), dtype=torch.long, device=mixed_qkv.device
    )
    conv_histories = torch.cat((initial_history, raw_inputs), dim=0).index_select(0, history_rows.reshape(-1))
    conv_histories = conv_histories.reshape(num_nodes, width - 1, channels)
    conv_output = conv_fn(
        raw_inputs,
        conv_weights,
        bias=layer.conv1d.bias,
        conv_states=conv_histories,
        query_start_loc=None,
        cache_indices=torch.arange(num_nodes, dtype=torch.int32, device=mixed_qkv.device),
        initial_state_mode=None,
        num_accepted_tokens=None,
        activation_mode=1 if layer.activation else 0,
        pad_slot_id=-1,
        run_mode=1,
    )
    all_q, all_k, all_v = layer.rearrange_mixed_qkv(conv_output)

    replay_fn = None
    if compact_recurrent_fn is not None:
        replay_fn = compact_recurrent_fn(
            q=all_q, k=all_k, v=all_v, g=g, beta=beta,
            initial_state=initial_recurrent, parents=parents, out=core_attn_out,
        )
    elif tree_recurrent_fn is not None:
        tree_recurrent_fn(
            q=all_q, k=all_k, v=all_v, g=g, beta=beta,
            initial_state=initial_recurrent, parents=parents,
            out=core_attn_out, snapshots=recurrent_snapshots,
        )

    fused = tree_recurrent_fn is not None or compact_recurrent_fn is not None
    for depth in (() if fused else range(max(depths) + 1)):
        nodes = tuple(node for node, node_depth in enumerate(depths) if node_depth == depth)
        node_indices = torch.tensor(nodes, dtype=torch.long, device=mixed_qkv.device)
        batch_size = len(nodes)
        if depth == 0:
            level_recurrent = initial_recurrent.clone()
        else:
            parent_indices = torch.tensor(tuple(parents[node] for node in nodes), dtype=torch.long, device=mixed_qkv.device)
            # index_select materializes independent rows even for shared parents.
            level_recurrent = recurrent_snapshots.index_select(0, parent_indices)
        local_indices = torch.arange(batch_size, dtype=torch.int32, device=mixed_qkv.device)
        level_output = recurrent_fn(
            q=all_q.index_select(1, node_indices),
            k=all_k.index_select(1, node_indices),
            v=all_v.index_select(1, node_indices),
            g=g.index_select(1, node_indices),
            beta=beta.index_select(1, node_indices),
            state=level_recurrent,
            cu_seqlens=torch.arange(batch_size + 1, dtype=torch.int32, device=mixed_qkv.device),
            ssm_state_indices=local_indices,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=True,
        )
        _copy_node_rows(core_attn_out, nodes, level_output.squeeze(0))
        _copy_node_rows(recurrent_snapshots, nodes, level_recurrent)

    def commit(path: tuple[int, ...]) -> None:
        _commit_tree_states(
            parents=parents,
            path=path,
            state_indices=state_indices,
            recurrent_cache=recurrent_cache,
            recurrent_snapshots=recurrent_snapshots,
            conv_cache=conv_cache,
            initial_conv_history=initial_history,
            raw_inputs=raw_inputs,
            cache_update_fn=cache_update_fn,
            replay_fn=replay_fn,
        )

    context.register_commit(("gdn", layer.prefix), commit)
    context.gdn_state_indices[layer.prefix] = state_indices
