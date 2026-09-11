#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify that an int8 lm-head prune pack belongs to one checkpoint.

This is deliberately a CPU-only, fail-closed tool.  It imports neither vLLM
nor torch_npu, loads the prune pack with ``weights_only=True``, and asks
safetensors for only the seven tensors belonging to the selected lm_head.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Torch 2.10 can discover torch_npu through backend entry points at import
# time.  This verifier must remain CPU-only even inside a serving container.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
from safetensors import safe_open

DEFAULT_PACK = Path("/home/models/lmhead_prune_v3_int8.pt")
DEFAULT_CHECKPOINT = Path("/home/models/Qwen3.5-9B-w8a8-lmhead-mtp")
HASH_CHUNK_BYTES = 64 * 1024 * 1024

PACK_KEYS = {
    "mode",
    "weight",
    "deq_scale",
    "quant_bias",
    "keep_ids",
    "inv_map",
    "orig_vocab",
}
OPTIONAL_PACK_KEYS = {"note"}
MAX_PACK_NOTE_BYTES = 1024
HEAD_SUFFIXES = (
    "weight",
    "deq_scale",
    "quant_bias",
    "input_offset",
    "input_scale",
    "weight_offset",
    "weight_scale",
)
ROW_SUFFIXES = (
    "deq_scale",
    "quant_bias",
    "weight_offset",
    "weight_scale",
)


class VerificationError(RuntimeError):
    """A structural or value mismatch that makes the pack unsafe to use."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--head-prefix",
        help="Exact safetensors prefix; auto-detected only when unambiguous.",
    )
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Allow replacing the explicitly named report file.",
    )
    return parser.parse_args()


def _add_check(report: dict[str, Any], name: str, passed: bool, **details: Any) -> bool:
    entry: dict[str, Any] = {"passed": bool(passed)}
    entry.update(details)
    report["checks"][name] = entry
    return bool(passed)


def _require(report: dict[str, Any], name: str, condition: bool, **details: Any) -> None:
    if not _add_check(report, name, condition, **details):
        raise VerificationError(f"required check failed: {name}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    flat_bytes = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    view = memoryview(flat_bytes.numpy())
    digest = hashlib.sha256()
    for offset in range(0, len(view), HASH_CHUNK_BYTES):
        digest.update(view[offset : offset + HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _tensor_summary(tensor: torch.Tensor, *, include_hash: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "nbytes": tensor.numel() * tensor.element_size(),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
    }
    if tensor.numel():
        summary["min"] = tensor.min().item()
        summary["max"] = tensor.max().item()
    if include_hash:
        summary["sha256"] = _tensor_sha256(tensor)
    return summary


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _safe_shard_path(checkpoint: Path, relative_name: str) -> Path:
    shard = (checkpoint / relative_name).resolve(strict=True)
    if not shard.is_relative_to(checkpoint):
        raise VerificationError(f"safetensors index escapes checkpoint directory: {relative_name!r}")
    if shard.suffix != ".safetensors" or not shard.is_file():
        raise VerificationError(f"invalid safetensors shard: {shard}")
    return shard


def _tensor_catalog(checkpoint: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    index_candidates = sorted(checkpoint.glob("*.safetensors.index.json"))
    if len(index_candidates) > 1:
        raise VerificationError(
            "multiple safetensors indices found: " + ", ".join(path.name for path in index_candidates)
        )

    catalog: dict[str, Path] = {}
    provenance: dict[str, Any]
    if index_candidates:
        index_path = index_candidates[0]
        index = _load_json(index_path)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise VerificationError(f"invalid or empty weight_map in {index_path}")
        for key, relative_name in weight_map.items():
            if not isinstance(key, str) or not isinstance(relative_name, str):
                raise VerificationError(f"non-string weight_map entry in {index_path}")
            if key in catalog:
                raise VerificationError(f"duplicate tensor key in index: {key}")
            catalog[key] = _safe_shard_path(checkpoint, relative_name)
        provenance = {
            "mode": "index",
            "index": str(index_path),
            "tensor_count": len(catalog),
            "shards": sorted({str(path) for path in catalog.values()}),
        }
    else:
        shards = sorted(checkpoint.glob("*.safetensors"))
        if not shards:
            raise VerificationError(f"no safetensors files found in {checkpoint}")
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in catalog:
                        raise VerificationError(f"duplicate tensor key {key!r} in {shard}")
                    catalog[key] = shard.resolve(strict=True)
        provenance = {
            "mode": "scanned_headers",
            "tensor_count": len(catalog),
            "shards": [str(path.resolve(strict=True)) for path in shards],
        }
    return catalog, provenance


def _load_tensor(catalog: dict[str, Path], key: str) -> torch.Tensor:
    shard = catalog[key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key)
    if tensor.device.type != "cpu":
        raise VerificationError(f"safetensors returned non-CPU tensor for {key}: {tensor.device}")
    return tensor


def _select_head_prefix(catalog: dict[str, Path], requested: str | None) -> tuple[str, list[str]]:
    candidates = sorted(
        key.removesuffix(".weight")
        for key in catalog
        if key.endswith(".weight") and "lm_head" in key.split(".")
    )
    if requested is not None:
        if requested not in candidates:
            raise VerificationError(
                f"requested head prefix {requested!r} is absent; candidates={candidates}"
            )
        return requested, candidates
    if len(candidates) != 1:
        raise VerificationError(
            "lm_head prefix is not unambiguous; pass --head-prefix; "
            f"candidates={candidates}"
        )
    return candidates[0], candidates


def _config_dimension(config: dict[str, Any], name: str) -> tuple[int, str]:
    candidates: list[tuple[int, str]] = []
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get(name), int):
        candidates.append((text_config[name], f"text_config.{name}"))
    if isinstance(config.get(name), int):
        candidates.append((config[name], name))
    if not candidates:
        raise VerificationError(f"config.json has no integer {name}")
    values = {value for value, _ in candidates}
    if len(values) != 1:
        raise VerificationError(f"config.json has conflicting {name} values: {candidates}")
    return candidates[0]


def _require_pack_tensor(
    report: dict[str, Any],
    pack: dict[str, Any],
    name: str,
    dtype: torch.dtype,
    ndim: int,
) -> torch.Tensor:
    value = pack[name]
    _require(report, f"pack.{name}.is_tensor", isinstance(value, torch.Tensor), actual=type(value).__name__)
    tensor = value
    _require(report, f"pack.{name}.cpu", tensor.device.type == "cpu", actual=str(tensor.device))
    _require(report, f"pack.{name}.dtype", tensor.dtype == dtype, expected=str(dtype), actual=str(tensor.dtype))
    _require(report, f"pack.{name}.ndim", tensor.ndim == ndim, expected=ndim, actual=tensor.ndim)
    return tensor


def _verify_pack_schema(report: dict[str, Any], pack: Any) -> dict[str, Any]:
    _require(report, "pack.is_plain_dict", type(pack) is dict, actual=type(pack).__name__)
    keys = set(pack)
    _require(
        report,
        "pack.required_schema_keys_present",
        PACK_KEYS <= keys,
        expected=sorted(PACK_KEYS),
        actual=sorted(str(key) for key in keys),
    )
    unknown_keys = keys - PACK_KEYS - OPTIONAL_PACK_KEYS
    _require(
        report,
        "pack.no_unknown_schema_keys",
        not unknown_keys,
        allowed_optional=sorted(OPTIONAL_PACK_KEYS),
        unknown=sorted(str(key) for key in unknown_keys),
    )
    if "note" in pack:
        _require(
            report,
            "pack.note.type",
            type(pack["note"]) is str,
            expected="str",
            actual=type(pack["note"]).__name__,
        )
        note_bytes = pack["note"].encode("utf-8")
        _require(
            report,
            "pack.note.size",
            len(note_bytes) <= MAX_PACK_NOTE_BYTES,
            maximum_bytes=MAX_PACK_NOTE_BYTES,
            actual_bytes=len(note_bytes),
        )
        report["pack_note"] = {
            "present": True,
            "utf8_bytes": len(note_bytes),
            "sha256": hashlib.sha256(note_bytes).hexdigest(),
            "content_recorded": False,
        }
    else:
        report["pack_note"] = {"present": False}
    _require(report, "pack.mode", pack["mode"] == "int8", expected="int8", actual=repr(pack["mode"]))
    _require(
        report,
        "pack.orig_vocab.type",
        type(pack["orig_vocab"]) is int,
        expected="int",
        actual=type(pack["orig_vocab"]).__name__,
    )
    return pack


def _verify_canonical_mapping(
    report: dict[str, Any],
    keep_ids: torch.Tensor,
    inv_map: torch.Tensor,
    orig_vocab: int,
    pruned_vocab: int,
) -> None:
    _require(
        report,
        "pack.prunes_vocabulary",
        0 < pruned_vocab < orig_vocab,
        pruned_vocab=pruned_vocab,
        orig_vocab=orig_vocab,
    )
    _require(
        report,
        "pack.keep_ids.shape",
        tuple(keep_ids.shape) == (pruned_vocab,),
        expected=[pruned_vocab],
        actual=list(keep_ids.shape),
    )
    _require(
        report,
        "pack.inv_map.shape",
        tuple(inv_map.shape) == (orig_vocab,),
        expected=[orig_vocab],
        actual=list(inv_map.shape),
    )
    in_range = bool(
        keep_ids.numel()
        and (keep_ids >= 0).all().item()
        and (keep_ids < orig_vocab).all().item()
    )
    _require(report, "pack.keep_ids.canonical_range", in_range)
    strictly_increasing = bool(
        keep_ids.numel() <= 1 or (keep_ids[1:] > keep_ids[:-1]).all().item()
    )
    _require(report, "pack.keep_ids.strictly_increasing_unique", strictly_increasing)

    expected_positions = torch.arange(pruned_vocab, dtype=torch.int64)
    kept_positions_match = torch.equal(inv_map.index_select(0, keep_ids), expected_positions)
    _require(report, "pack.inv_map.kept_positions_exact", kept_positions_match)

    kept_mask = torch.zeros(orig_vocab, dtype=torch.bool)
    kept_mask.index_fill_(0, keep_ids, True)
    omitted = inv_map[~kept_mask]
    omitted_use_sentinel = bool((omitted == pruned_vocab).all().item())
    _require(
        report,
        "pack.inv_map.omitted_ids_use_padding_column",
        omitted_use_sentinel,
        sentinel=pruned_vocab,
        omitted_ids=omitted.numel(),
    )
    all_values_valid = bool(((inv_map >= 0) & (inv_map <= pruned_vocab)).all().item())
    _require(report, "pack.inv_map.values_in_closed_range", all_values_valid)
    canonical_roundtrip = torch.equal(
        keep_ids.index_select(0, inv_map.index_select(0, keep_ids)),
        keep_ids,
    )
    _require(report, "pack.canonical_token_id_roundtrip", canonical_roundtrip)
    report["canonical_mapping"] = {
        "kept_ids": pruned_vocab,
        "omitted_ids": orig_vocab - pruned_vocab,
        "padding_column": pruned_vocab,
        "padding_columns": 1,
        "inv_map_length": inv_map.numel(),
        "meaning": "kept token id -> packed row; omitted token id -> K padding column",
    }


def _verify_checkpoint_contract(
    report: dict[str, Any],
    checkpoint: Path,
    catalog: dict[str, Path],
    prefix: str,
    orig_vocab: int,
) -> tuple[int, int]:
    expected_keys = {f"{prefix}.{suffix}" for suffix in HEAD_SUFFIXES}
    actual_keys = {key for key in catalog if key.startswith(f"{prefix}.")}
    _require(
        report,
        "checkpoint.head_tensor_keys_exact",
        actual_keys == expected_keys,
        expected=sorted(expected_keys),
        actual=sorted(actual_keys),
    )

    config_path = checkpoint / "config.json"
    _require(report, "checkpoint.config_json_exists", config_path.is_file(), path=str(config_path))
    config = _load_json(config_path)
    _require(report, "checkpoint.config_json_object", isinstance(config, dict), actual=type(config).__name__)
    config_vocab, vocab_source = _config_dimension(config, "vocab_size")
    hidden_size, hidden_source = _config_dimension(config, "hidden_size")
    _require(
        report,
        "checkpoint.config_vocab_matches_pack",
        config_vocab == orig_vocab,
        checkpoint=config_vocab,
        pack=orig_vocab,
        source=vocab_source,
    )

    quant_path = checkpoint / "quant_model_description.json"
    _require(report, "checkpoint.quant_description_exists", quant_path.is_file(), path=str(quant_path))
    quant_description = _load_json(quant_path)
    _require(
        report,
        "checkpoint.quant_description_object",
        isinstance(quant_description, dict),
        actual=type(quant_description).__name__,
    )
    quant_values = {key: quant_description.get(key) for key in sorted(expected_keys)}
    _require(
        report,
        "checkpoint.head_quant_description_w8a8",
        all(value == "W8A8" for value in quant_values.values()),
        values=quant_values,
    )
    report["checkpoint"]["config"] = {
        "path": str(config_path),
        "vocab_size": config_vocab,
        "vocab_size_source": vocab_source,
        "hidden_size": hidden_size,
        "hidden_size_source": hidden_source,
    }
    report["checkpoint"]["quant_description"] = {
        "path": str(quant_path),
        "head_values": quant_values,
    }
    return config_vocab, hidden_size


def _verify(args: argparse.Namespace, report: dict[str, Any]) -> None:
    _require(
        report,
        "runtime.torch_npu_not_imported",
        "torch_npu" not in sys.modules,
        imported_modules=[name for name in sys.modules if name == "torch_npu" or name.startswith("torch_npu.")],
    )
    report["runtime"] = {
        "torch_version": torch.__version__,
        "torch_device_backend_autoload": os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"],
        "execution": "CPU_ONLY",
    }
    pack_path = args.pack.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    _require(report, "inputs.pack_is_file", pack_path.is_file(), path=str(pack_path))
    _require(report, "inputs.checkpoint_is_directory", checkpoint.is_dir(), path=str(checkpoint))
    report["inputs"] = {
        "pack": str(pack_path),
        "checkpoint": str(checkpoint),
        "head_prefix_requested": args.head_prefix,
    }
    report["loader_contract"] = {
        "source": "vllm_ascend/_310p/lmhead_prune.py",
        "supported_mode": "int8",
        "replaced_from_pack": ["weight", "deq_scale", "quant_bias"],
        "retained_from_checkpoint": ["input_scale", "input_offset"],
        "runtime_weight_transform": "maybe_trans_nz(pack.weight).transpose(0, 1)",
        "full_vocab_expansion": "K packed logits plus one omitted-token padding column selected by inv_map",
    }

    pack_object = torch.load(pack_path, weights_only=True, map_location="cpu")
    pack = _verify_pack_schema(report, pack_object)
    weight = _require_pack_tensor(report, pack, "weight", torch.int8, 2)
    deq_scale = _require_pack_tensor(report, pack, "deq_scale", torch.int64, 1)
    quant_bias = _require_pack_tensor(report, pack, "quant_bias", torch.int32, 1)
    keep_ids = _require_pack_tensor(report, pack, "keep_ids", torch.int64, 1)
    inv_map = _require_pack_tensor(report, pack, "inv_map", torch.int64, 1)
    orig_vocab = pack["orig_vocab"]
    pruned_vocab, pack_hidden = weight.shape

    for name, tensor in (
        ("weight", weight),
        ("deq_scale", deq_scale),
        ("quant_bias", quant_bias),
        ("keep_ids", keep_ids),
        ("inv_map", inv_map),
    ):
        _require(report, f"pack.{name}.contiguous", tensor.is_contiguous())
    _require(
        report,
        "pack.deq_scale.shape",
        tuple(deq_scale.shape) == (pruned_vocab,),
        expected=[pruned_vocab],
        actual=list(deq_scale.shape),
    )
    _require(
        report,
        "pack.quant_bias.shape",
        tuple(quant_bias.shape) == (pruned_vocab,),
        expected=[pruned_vocab],
        actual=list(quant_bias.shape),
    )
    _verify_canonical_mapping(report, keep_ids, inv_map, orig_vocab, pruned_vocab)
    report["pack"] = {
        "file_size": pack_path.stat().st_size,
        "file_sha256": _file_sha256(pack_path),
        "mode": pack["mode"],
        "orig_vocab": orig_vocab,
        "pruned_vocab": pruned_vocab,
        "hidden_size": pack_hidden,
        "note": report.pop("pack_note"),
        "tensors": {
            name: _tensor_summary(pack[name])
            for name in ("weight", "deq_scale", "quant_bias", "keep_ids", "inv_map")
        },
    }

    catalog, catalog_provenance = _tensor_catalog(checkpoint)
    prefix, candidates = _select_head_prefix(catalog, args.head_prefix)
    report["checkpoint"] = {
        "catalog": catalog_provenance,
        "head_prefix": prefix,
        "head_prefix_candidates": candidates,
        "tensor_shards": {
            suffix: str(catalog.get(f"{prefix}.{suffix}", "MISSING"))
            for suffix in HEAD_SUFFIXES
        },
    }
    _, config_hidden = _verify_checkpoint_contract(
        report,
        checkpoint,
        catalog,
        prefix,
        orig_vocab,
    )
    _require(
        report,
        "pack.hidden_size_matches_config",
        pack_hidden == config_hidden,
        pack=pack_hidden,
        checkpoint=config_hidden,
    )

    head_tensors = {
        suffix: _load_tensor(catalog, f"{prefix}.{suffix}")
        for suffix in HEAD_SUFFIXES
    }
    head_weight = head_tensors["weight"]
    _require(
        report,
        "checkpoint.weight.dtype",
        head_weight.dtype == torch.int8,
        expected=str(torch.int8),
        actual=str(head_weight.dtype),
    )
    _require(report, "checkpoint.weight.ndim", head_weight.ndim == 2, actual=head_weight.ndim)
    row_layout = head_weight.shape[1] == config_hidden and head_weight.shape[0] >= orig_vocab
    column_layout = head_weight.shape[0] == config_hidden and head_weight.shape[1] >= orig_vocab
    _require(
        report,
        "checkpoint.weight.vocab_axis_unambiguous",
        row_layout != column_layout,
        shape=list(head_weight.shape),
        hidden_size=config_hidden,
        orig_vocab=orig_vocab,
    )
    vocab_axis = 0 if row_layout else 1
    _require(
        report,
        "checkpoint.weight.vocab_hidden_row_layout",
        vocab_axis == 0,
        expected="[vocab_extent, hidden_size]",
        actual=list(head_weight.shape),
    )
    checkpoint_vocab_extent = head_weight.shape[vocab_axis]
    hidden_extent = head_weight.shape[1 - vocab_axis]
    _require(
        report,
        "checkpoint.weight.hidden_extent",
        hidden_extent == config_hidden,
        expected=config_hidden,
        actual=hidden_extent,
    )

    expected_dtypes = {
        "deq_scale": torch.int64,
        "quant_bias": torch.int32,
        "input_offset": torch.float16,
        "input_scale": torch.float16,
        "weight_offset": torch.float16,
        "weight_scale": torch.float16,
    }
    for suffix, dtype in expected_dtypes.items():
        tensor = head_tensors[suffix]
        _require(
            report,
            f"checkpoint.{suffix}.dtype",
            tensor.dtype == dtype,
            expected=str(dtype),
            actual=str(tensor.dtype),
        )
    expected_row_shapes = {
        "deq_scale": (checkpoint_vocab_extent,),
        "quant_bias": (checkpoint_vocab_extent,),
        "weight_offset": (checkpoint_vocab_extent, 1),
        "weight_scale": (checkpoint_vocab_extent, 1),
    }
    for suffix in ROW_SUFFIXES:
        tensor = head_tensors[suffix]
        _require(
            report,
            f"checkpoint.{suffix}.layout",
            tuple(tensor.shape) == expected_row_shapes[suffix],
            expected=list(expected_row_shapes[suffix]),
            actual=list(tensor.shape),
        )
        _require(report, f"checkpoint.{suffix}.contiguous", tensor.is_contiguous())
    for suffix in ("input_scale", "input_offset"):
        tensor = head_tensors[suffix]
        _require(
            report,
            f"checkpoint.{suffix}.layout",
            tuple(tensor.shape) == (1,),
            expected=[1],
            actual=list(tensor.shape),
        )
        _require(report, f"checkpoint.{suffix}.contiguous", tensor.is_contiguous())
        _require(
            report,
            f"checkpoint.{suffix}.finite",
            bool(torch.isfinite(tensor).all().item()),
            value=tensor.reshape(-1)[0].item(),
        )
    _require(
        report,
        "checkpoint.input_scale.nonzero",
        bool((head_tensors["input_scale"] != 0).all().item()),
        value=head_tensors["input_scale"].reshape(-1)[0].item(),
    )

    selected_weight = (
        head_weight.index_select(0, keep_ids)
        if vocab_axis == 0
        else head_weight.index_select(1, keep_ids).transpose(0, 1).contiguous()
    )
    selected_deq_scale = head_tensors["deq_scale"].index_select(0, keep_ids).reshape(-1)
    selected_quant_bias = head_tensors["quant_bias"].index_select(0, keep_ids).reshape(-1)
    weight_matches = torch.equal(weight, selected_weight)
    deq_matches = torch.equal(deq_scale, selected_deq_scale)
    bias_matches = torch.equal(quant_bias, selected_quant_bias)
    _add_check(report, "values.weight_selected_rows_exact", weight_matches)
    _add_check(report, "values.deq_scale_selected_rows_exact", deq_matches)
    _add_check(report, "values.quant_bias_selected_rows_exact", bias_matches)

    selected_auxiliary = {
        suffix: head_tensors[suffix].index_select(0, keep_ids)
        for suffix in ("weight_scale", "weight_offset")
    }
    report["checkpoint"]["weight_layout"] = {
        "stored_shape": list(head_weight.shape),
        "stored_dtype": str(head_weight.dtype),
        "vocab_axis": vocab_axis,
        "hidden_axis": 1 - vocab_axis,
        "checkpoint_vocab_extent": checkpoint_vocab_extent,
        "canonical_vocab": orig_vocab,
        "padding_rows": checkpoint_vocab_extent - orig_vocab,
        "padding_location": "suffix_after_canonical_ids",
        "pack_selects_only_canonical_prefix": bool((keep_ids < orig_vocab).all().item()),
    }
    _require(
        report,
        "checkpoint.padding_extent_nonnegative",
        checkpoint_vocab_extent >= orig_vocab,
        checkpoint_vocab_extent=checkpoint_vocab_extent,
        canonical_vocab=orig_vocab,
    )
    report["checkpoint"]["tensors"] = {
        suffix: _tensor_summary(tensor)
        for suffix, tensor in head_tensors.items()
    }
    report["selected_rows"] = {
        "weight": _tensor_summary(selected_weight),
        "deq_scale": _tensor_summary(selected_deq_scale),
        "quant_bias": _tensor_summary(selected_quant_bias),
        "weight_scale": _tensor_summary(selected_auxiliary["weight_scale"]),
        "weight_offset": _tensor_summary(selected_auxiliary["weight_offset"]),
    }
    report["activation_quantization"] = {
        "pack_fields_present": [],
        "loader_behavior": "input_scale and input_offset remain loaded from this checkpoint",
        "input_scale": _tensor_summary(head_tensors["input_scale"]),
        "input_offset": _tensor_summary(head_tensors["input_offset"]),
        "compatibility_basis": (
            "the pack exactly matches this checkpoint's selected weight/deq_scale/quant_bias rows; "
            "the canonical pack schema contains no activation override"
        ),
    }
    _add_check(
        report,
        "activation.pack_does_not_override_checkpoint_input_quantization",
        "input_scale" not in pack and "input_offset" not in pack,
    )

    report["value_hash_pairs"] = {
        "weight": {
            "pack": report["pack"]["tensors"]["weight"]["sha256"],
            "checkpoint_selected": report["selected_rows"]["weight"]["sha256"],
        },
        "deq_scale": {
            "pack": report["pack"]["tensors"]["deq_scale"]["sha256"],
            "checkpoint_selected": report["selected_rows"]["deq_scale"]["sha256"],
        },
        "quant_bias": {
            "pack": report["pack"]["tensors"]["quant_bias"]["sha256"],
            "checkpoint_selected": report["selected_rows"]["quant_bias"]["sha256"],
        },
    }
    binding_payload = {
        "head_prefix": prefix,
        "orig_vocab": orig_vocab,
        "hidden_size": config_hidden,
        "pack_weight": report["pack"]["tensors"]["weight"]["sha256"],
        "pack_deq_scale": report["pack"]["tensors"]["deq_scale"]["sha256"],
        "pack_quant_bias": report["pack"]["tensors"]["quant_bias"]["sha256"],
        "checkpoint_input_scale": report["activation_quantization"]["input_scale"]["sha256"],
        "checkpoint_input_offset": report["activation_quantization"]["input_offset"]["sha256"],
    }
    report["checkpoint_binding"] = {
        "sha256": hashlib.sha256(
            json.dumps(binding_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "components": binding_payload,
    }

    del selected_weight, selected_deq_scale, selected_quant_bias, selected_auxiliary
    del head_weight, head_tensors
    gc.collect()

    failed_checks = sorted(name for name, value in report["checks"].items() if not value["passed"])
    report["failed_checks"] = failed_checks
    if failed_checks:
        raise VerificationError("value verification failed: " + ", ".join(failed_checks))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    report_path = args.report.resolve(strict=False)
    if report_path.exists() and not args.overwrite_report:
        print(
            f"refusing to overwrite existing report: {report_path}; use --overwrite-report",
            file=sys.stderr,
        )
        return 2

    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "verify_tree_prune_pack.py",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "checks": {},
    }
    try:
        with torch.inference_mode():
            _verify(args, report)
        report["status"] = "PASS"
        report["safe_for_pruned_graph_control"] = True
        exit_code = 0
    except Exception as error:
        report["status"] = "FAIL"
        report["safe_for_pruned_graph_control"] = False
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        report["failed_checks"] = sorted(
            name for name, value in report["checks"].items() if not value["passed"]
        )
        exit_code = 1
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_report(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "safe_for_pruned_graph_control": report["safe_for_pruned_graph_control"],
                "failed_checks": report.get("failed_checks", []),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

