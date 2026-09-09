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

from numbers import Integral
from typing import Any

import torch
import torch_npu
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)

from vllm_ascend._310p.attention.attention_mask import (
    AttentionMaskBuilder310,
    is_compressed_mask_supported,
)
from vllm_ascend._310p.attention.metadata_builder import (
    AscendAttentionMetadataBuilder310,
    get_query_lens_cpu,
    get_splitfuse_mask_nz,
)
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_spec

MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION = 3
MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION = 5
TREE_MASK_ALIGNMENT = 16


def _tree_context_from_metadata(attn_metadata: Any) -> Any | None:
    # The context is explicitly attached to this step's metadata. Inspecting
    # stored attributes also avoids treating dynamic proxy attributes as opt-in.
    return vars(attn_metadata).get("tree_mtp_context") if attn_metadata is not None else None


def _tree_dimensions(context: Any) -> tuple[int, int, tuple[int, ...]]:
    if context is None:
        raise ValueError("Tree attention requires an explicitly attached tree step context.")
    num_nodes, prefix_length = context.num_nodes, context.prefix_length
    if not isinstance(num_nodes, Integral) or isinstance(num_nodes, bool) or num_nodes < 1:
        raise ValueError("Tree attention requires a positive integer node count.")
    if not isinstance(prefix_length, Integral) or isinstance(prefix_length, bool) or prefix_length < 0:
        raise ValueError("Tree attention requires a nonnegative integer prefix length.")
    parents = tuple(context.tree.parents)
    if len(parents) != num_nodes or parents[0] != -1:
        raise ValueError("Tree attention requires exactly one parent entry per node and root parent -1.")
    for node, parent in enumerate(parents[1:], 1):
        if not isinstance(parent, Integral) or isinstance(parent, bool) or not 0 <= parent < node:
            raise ValueError("Tree attention requires parent-before-child node order.")
    return int(num_nodes), int(prefix_length), parents


def build_tree_attention_mask(context: Any, device: torch.device) -> torch.Tensor:
    """Build an additive mask in physical prefix-plus-flat-node coordinates.

    The aligned key tail is explicitly masked: nd_to_nz_spec pads with zero,
    which must not accidentally expose unused cache slots. Build the small
    tree's row indices on CPU and upload the mask once, not once per node.
    """
    num_nodes, prefix_length, parents = _tree_dimensions(context)
    context_length = prefix_length + num_nodes
    aligned_length = (context_length + TREE_MASK_ALIGNMENT - 1) // TREE_MASK_ALIGNMENT * TREE_MASK_ALIGNMENT
    mask = torch.full((num_nodes, aligned_length), -float("inf"), dtype=torch.float16, device="cpu")
    mask[:, :prefix_length] = 0
    for node in range(num_nodes):
        ancestor = node
        while ancestor >= 0:
            mask[node, prefix_length + ancestor] = 0
            ancestor = parents[ancestor]
    return mask.to(device=device, non_blocking=True)


def register_tree_cache_commit(
    context: Any,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Snapshot rotated dense K/V; compact only the accepted input-node path.

    The original slots must describe C + flat_index, independently of the
    tree's C + depth RoPE positions. Reusing their prefix handles arbitrary
    physical block boundaries without decoding or overlapping NZ-cache reads.
    """
    num_nodes, _, parents = _tree_dimensions(context)
    if key.ndim != 3 or value.ndim != 3 or min(key.shape[0], value.shape[0]) < num_nodes:
        raise ValueError("Tree cache snapshots require K/V shaped [at_least_num_nodes, heads, dim].")
    if slot_mapping.ndim != 1 or slot_mapping.numel() < num_nodes:
        raise ValueError("Tree cache snapshots require one physical slot per node.")
    commit_key = ("tree_attention_kv", id(key_cache), id(value_cache))
    if context.has_commit(commit_key):
        # Some upstream paths expose both writer entry points. The first
        # snapshot belongs to the actual tree pass and must not be replaced.
        return
    saved_key = key[:num_nodes].contiguous().clone()
    saved_value = value[:num_nodes].contiguous().clone()
    saved_slots = slot_mapping[:num_nodes].contiguous().clone()

    def commit(accepted_indices: tuple[int, ...]) -> None:
        path = tuple(accepted_indices)
        if not path:
            return
        if len(path) > num_nodes or path[0] != 0:
            raise ValueError("The accepted input path must start at the root.")
        for rank, node in enumerate(path):
            if not isinstance(node, Integral) or isinstance(node, bool) or not 0 <= node < num_nodes:
                raise ValueError("Accepted input indices must refer to real tree nodes.")
            if rank and parents[node] != path[rank - 1]:
                raise ValueError("The accepted input path must follow direct parent-child edges.")
        indices = torch.tensor(path, dtype=torch.long, device=saved_key.device)
        # Gather BOTH sources before any write: source and destination slots
        # can overlap, and the live projection/slot buffers may already change.
        selected_key = saved_key.index_select(0, indices).contiguous()
        selected_value = saved_value.index_select(0, indices).contiguous()
        DeviceOperator.reshape_and_cache(
            key=selected_key,
            value=selected_value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=saved_slots[: len(path)].contiguous(),
        )

    context.register_commit(commit_key, commit)


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend310(AscendAttentionBackend):
    def __init__(self, *args, **kwargs):
        """
        Initializes the 310P backend and sets up the device-specific mask builder.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ):
        """
        Determines the shape of the Key-Value (KV) cache tensor.

        The 310P hardware requires specific memory alignment for optimal performance.
        This method defines a 5D tensor shape where the head size dimension is
        split to ensure alignment to multiples of 16.

        Args:
            num_blocks (int): Number of memory blocks.
            block_size (int): Size of each block.
            num_kv_heads (int): Number of KV heads.
            head_size (int): Dimension size of each head.

        Returns:
            tuple: The specific 5D shape required by the hardware
                   (2, num_blocks, hidden_dim_aligned, block_size, 16).
        """
        # Align to a multiple of 16, as required by the 310P device.
        return (2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)

    @staticmethod
    def get_impl_cls():
        """
        Returns the implementation class for the attention operations.
        """
        return AscendAttentionBackendImpl310

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        """
        Returns the metadata builder class specifically for 310P.
        """
        return AscendAttentionMetadataBuilder310

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128, 64]


class AscendAttentionBackendImpl310(AscendAttentionBackendImpl):
    """
    Implementation of attention operations (Prefill, Decode, Chunked Prefill)
    optimized for the Ascend 310P architecture.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.support_compressed_mask = is_compressed_mask_supported()

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping) -> None:
        metadata = get_forward_context().attn_metadata if is_forward_context_available() else None
        if isinstance(metadata, dict):
            metadata = metadata.get(layer.layer_name)
        context = _tree_context_from_metadata(metadata)
        if context is not None:
            if self.kv_sharing_target_layer_name is not None:
                return
            if len(kv_cache) < 2:
                raise ValueError("Tree attention requires allocated K and V caches.")
            register_tree_cache_commit(context, key, value, kv_cache[0], kv_cache[1], slot_mapping)
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        context = _tree_context_from_metadata(attn_metadata)
        if context is not None and self.kv_sharing_target_layer_name is None:
            if len(kv_cache) < 2:
                raise ValueError("Tree attention requires allocated K and V caches.")
            register_tree_cache_commit(context, key, value, kv_cache[0], kv_cache[1], attn_metadata.slot_mapping)
        return super().reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

    def forward_tree_attention_310(self, query, attn_metadata, output):
        """Verify all flat tree nodes with an explicit, uncompressed mask.

        This composition deliberately uses the legacy splitfuse mask interface.
        A compressed causal mask cannot describe arbitrary ancestry. Device
        correctness for this mask family must be established separately.
        """
        context = _tree_context_from_metadata(attn_metadata)
        num_nodes, prefix_length, _ = _tree_dimensions(context)
        if is_forward_context_available() and _EXTRA_CTX.capturing:
            raise NotImplementedError("Tree attention currently requires eager execution.")
        if self.attn_type != AttentionType.DECODER or self.sliding_window is not None:
            raise NotImplementedError(
                "Tree attention currently requires full decoder attention without sliding windows."
            )
        if attn_metadata.attn_state != AscendAttentionState.SpecDecoding and not (
            num_nodes == 1 and attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        ):
            raise ValueError("Tree attention metadata must describe a speculative verification pass.")
        if attn_metadata.num_actual_tokens != num_nodes or min(query.shape[0], output.shape[0]) < num_nodes:
            raise ValueError("Tree attention query/output count does not match the tree.")
        if query.dtype != torch.float16:
            raise TypeError("310P tree attention requires FP16 queries.")
        if attn_metadata.block_tables.shape[0] != 1 or attn_metadata.seq_lens.numel() != 1:
            raise ValueError("Tree attention currently supports batch size one only.")
        if self.key_cache is None or self.value_cache is None:
            raise ValueError("Tree attention requires initialized K and V caches.")
        if not hasattr(torch_npu, "_npu_paged_attention_splitfuse"):
            raise RuntimeError("Tree attention requires the legacy splitfuse custom-mask operator.")
        context_length = prefix_length + num_nodes
        block_size = self.key_cache.shape[2]
        if attn_metadata.block_tables.shape[1] * block_size < context_length:
            raise ValueError("Tree attention block table does not cover the physical tree span.")
        additive_mask = build_tree_attention_mask(context, query.device)
        mask = torch_npu.npu_format_cast(nd_to_nz_spec(additive_mask).contiguous(), ACL_FORMAT_FRACTAL_NZ)
        # These are physical lengths, never max(depth)+1. Explicitly bound them
        # so optimistic scheduler metadata cannot expose stale sibling slots.
        query_lens = torch.tensor([num_nodes], dtype=torch.int32, device="cpu")
        context_lens = torch.tensor([context_length], dtype=torch.int32, device=query.device)
        torch_npu._npu_paged_attention_splitfuse(
            query=query[:num_nodes],
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=mask,
            block_table=attn_metadata.block_tables,
            seq_len=query_lens,
            context_lens=context_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output[:num_nodes],
        )
        return output

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        seq_len: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.support_compressed_mask:
            torch_npu._npu_flash_attention(
                query=query,
                key=key,
                value=value,
                mask=mask,
                seq_len=seq_len,
                scale_value=self.scale,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                out=output,
            )
            return output

        torch_npu._npu_flash_attention_v3(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=seq_len,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            mask_type=MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._flash_attention(
            query,
            key,
            value,
            attn_metadata.attn_mask,
            attn_metadata.seq_lens,
            output,
        )

    def forward_paged_attention(
        self,
        query: Any,
        attn_metadata: AscendMetadata,
        output: Any | None = None,
    ) -> Any:
        """
        Executes Paged Attention (typically for the decode phase).

        Ensures that the sequence length metadata is on the correct device
        before invoking the base implementation.

        Args:
            query (Any): The query tensor.
            attn_metadata (AscendMetadata): Metadata associated with the attention request.
            output (Any | None): Optional output tensor.

        Returns:
            Any: The result of the attention operation.
        """
        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )

        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def forward_prefill_310(self, query, key, value, attn_metadata, output):
        """
        Executes Flash Attention for the prefill phase on 310P.

        This method handles memory alignment padding. If the query shape implies
        padding (aligned_tokens > real_tokens), it adjusts the sequence length
        of the last request to account for the delta, ensuring the NPU operator
        processes the data correctly.

        Args:
            query, key, value: Input tensors.
            attn_metadata (AscendMetadata): Attention metadata containing masks and seq_lens.
            output: Output tensor.

        Returns:
            The output tensor after flash attention.
        """
        # seq_lens.sum().item() is a synchronous D2H copy, which ACL
        # forbids inside a graph capture (the drafter is captured whole under FULL).
        # num_actual_tokens is the same quantity already available on the host -- the
        # sibling splitfuse path below reads it the same way.
        _nat = getattr(attn_metadata, "num_actual_tokens", None)
        if _nat is not None:
            real_tokens = int(_nat)
        else:
            real_tokens = int(attn_metadata.seq_lens.sum().item())
        seq_len = attn_metadata.seq_lens
        aligned_tokens = int(query.shape[0])
        delta = aligned_tokens - real_tokens

        # Adjust sequence length if padding (alignment) was applied to the inputs
        if delta:
            seq_len = seq_len.clone()
            seq_len[-1] += delta

        mask = attn_metadata.attn_mask
        return self._flash_attention(query, key, value, mask, seq_len, output)

    def forward_chunked_prefill_310(self, query, attn_metadata, output):
        """
        Executes SplitFuse (Chunked Prefill) attention on 310P.

        This handles scenarios where the prefill is split into chunks. It prepares
        the necessary metadata (query lengths, block tables) and generates the
        specific splitfuse mask before calling the NPU operator.

        Args:
            query: The query tensor.
            attn_metadata (AscendMetadata): Metadata containing start locations and block tables.
            output: The output tensor.
        """
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        query = query[:num_actual_tokens]
        output_slice = output[:num_actual_tokens]

        # Host qLens filled in AscendAttentionMetadataBuilder310.build(); eager fallback only.
        qlens = get_query_lens_cpu(attn_metadata)
        if qlens is None:
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX

            if _EXTRA_CTX.capturing:
                raise RuntimeError(
                    "310P splitfuse requires attn_metadata.query_lens_cpu during graph capture; "
                    "ensure AscendAttentionMetadataBuilder310.build() ran before forward."
                )
            qsl_cpu = attn_metadata.query_start_loc.cpu()
            qlens = qsl_cpu[1:] - qsl_cpu[:-1]

        block_table = attn_metadata.block_tables

        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )

        if self.support_compressed_mask:
            # splitfuse_v2 requires fixed ND [2048, 2048]; parent build() may set FRACTAL_NZ mask.
            mask = AttentionMaskBuilder310.get_compressed_splitfuse_mask(query.device)
            torch_npu._npu_paged_attention_splitfuse_v2(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                mask=mask,
                block_table=block_table,
                seq_len=qlens,
                context_lens=attn_metadata.seq_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                mask_type=MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION,
                out=output_slice,
            )
            return output

        # Prefer the mask precomputed in the metadata builder (capture-safe);
        # building it here does sync copies that abort an ACL graph capture.
        mask = get_splitfuse_mask_nz(attn_metadata)
        if mask is None:
            mask = AttentionMaskBuilder310.get_splitfuse_mask(attn_metadata, query.device)
        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=mask,
            block_table=block_table,
            seq_len=qlens,
            context_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output_slice,
        )

        return output

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        """
        Main dispatch method for attention operations.

        Routes the execution to Decode, Prefill, or Chunked Prefill methods
        based on the current attention state found in metadata.

        Args:
            query, key, value: Input tensors (Key/Value usually empty for decode/chunked).
            kv_cache: The KV cache structure.
            attn_metadata: Metadata determining the state (Prefill vs Decode).
            output: Tensor to write results to.

        Returns:
            The output tensor.

        Raises:
            NotImplementedError: If the attention state is not supported on 310P.
        """
        if _tree_context_from_metadata(attn_metadata) is not None:
            return self.forward_tree_attention_310(query, attn_metadata, output)
        state = attn_metadata.attn_state
        # Condition for PrefillNoCache: No previous tokens have been processed yet
        if state == AscendAttentionState.PrefillNoCache:
            output = self.forward_prefill_310(query, key, value, attn_metadata, output)
        # Condition for DecodeOnly: Pure decoding phase where each request generates one token
        elif state == AscendAttentionState.DecodeOnly:
            output = self.forward_paged_attention(query, attn_metadata, output)
        # ChunkedPrefill / PrefillCacheHit: chunked prefill or mixed batches.
        # SpecDecoding: MTP uniform spec verify (splitfuse on 310P).
        elif (
            state in [AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit]
            or state == AscendAttentionState.SpecDecoding
        ):
            output = self.forward_chunked_prefill_310(query, attn_metadata, output)
        else:
            raise NotImplementedError(f"AscendAttentionState: {state} is not supported for 310P currently.")
        return output
