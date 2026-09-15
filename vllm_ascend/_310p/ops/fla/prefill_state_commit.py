# SPDX-License-Identifier: Apache-2.0
"""Host-indexed single-prefill commit; recurrent/graph paths stay unchanged."""


def prefill_host_state_slot(metadata, block_table_cpu, *, cache_mode,
                           prefix_caching, num_reqs, num_reqs_padded,
                           for_capture, pcp_size, dcp_size):
    # Do not read device sequence lengths or indices. In `none` mode the
    # GDN builder takes exactly column zero of this same group's block table.
    if (cache_mode != "none" or prefix_caching or for_capture
            or num_reqs != 1 or num_reqs_padded != 1
            or pcp_size != 1 or dcp_size != 1
            or metadata.num_prefills != 1 or metadata.num_decodes != 0
            or metadata.num_spec_decodes != 0
            or metadata.spec_sequence_masks is not None):
        return None
    if (block_table_cpu.device.type != "cpu" or block_table_cpu.ndim != 2
            or block_table_cpu.shape[0] < 1 or block_table_cpu.shape[1] < 1
            or str(block_table_cpu.dtype) not in ("torch.int32", "torch.int64")):
        return None
    slot = int(block_table_cpu[0, 0])
    # Snapshot a scalar, not a mutable CPU-table view across async steps.
    return slot if slot >= 0 else None


def copy_single_prefill_state(cache, value, slot):
    """Return False for unsupported metadata/layout, preserving the old path."""
    if (type(slot) is not int or len(cache.shape) != 4 or len(value.shape) != 4
            or slot < 0 or slot >= cache.shape[0]
            or value.shape[0] != 1 or tuple(value.shape[1:]) != tuple(cache.shape[1:])
            or value.dtype != cache.dtype or value.device != cache.device):
        return False
    cache[slot:slot + 1].copy_(value)
    return True
