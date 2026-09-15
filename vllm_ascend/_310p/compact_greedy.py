"""Greedy canonical token selection without materializing a full vocabulary."""

import torch


def make_compact_mapping(inv_map: torch.Tensor, size: int) -> tuple[torch.Tensor, int]:
    """Validate on CPU and retain the first omitted ID as a floor-logit row.

    Rows must be in canonical order to preserve argmax tie and NaN ordering.
    The omitted row also preserves the old finite-floor behavior for -inf.
    """
    assert inv_map.device.type == "cpu" and inv_map.ndim == 1
    ids = torch.arange(inv_map.numel(), dtype=torch.long)
    kept = ids[inv_map != size]
    if not torch.equal(inv_map[kept], torch.arange(size, dtype=inv_map.dtype)):
        raise ValueError("compact greedy requires a one-to-one, canonical-order inverse map")
    missing = ids[inv_map == size]
    if not missing.numel():
        return kept, -1
    first_missing = int(missing[0])
    insertion = int(torch.searchsorted(kept, first_missing))
    mapping = torch.cat((kept[:insertion], missing[:1], kept[insertion:]))
    return mapping, insertion


def compact_greedy_ids(logits: torch.Tensor, mapping: torch.Tensor, insertion: int) -> torch.Tensor:
    if insertion >= 0:
        floor = torch.full_like(logits[..., :1], torch.finfo(logits.dtype).min)
        logits = torch.cat((logits[..., :insertion], floor, logits[..., insertion:]), dim=-1)
    indices = logits.argmax(dim=-1)
    return mapping.index_select(0, indices.reshape(-1)).reshape(indices.shape)
