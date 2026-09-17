#!/usr/bin/env python3
"""Correctness-only device screen for the guarded QBMM K pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch_npu


FRACTAL_NZ = 29
VDEQ16_MARKER = 0x00004000
SEED = 20260917
SMALL_GOLDEN_MAX_N = 64
INTEGER_GOLDEN_K_BLOCK = 32

# name, M, K, N, expected host-policy selection when the new attr is true.
CASES = (
    ("small_m1", 1, 4096, 64, False),
    ("small_m15", 15, 4096, 64, False),
    ("small_m16", 16, 4096, 64, False),
    ("selected_down_m1266", 1266, 12288, 4096, True),
    ("fallback_down_m782", 782, 12288, 4096, False),
    ("fallback_down_m2048", 2048, 12288, 4096, False),
    ("fallback_output_m1266", 1266, 4096, 4096, False),
)


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor bytes, so FP16 +0 and -0 are not treated as equal."""
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def encode_vdeq16_scale(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Pack FP32 channel scales exactly as the existing kernel expects."""
    if scale_fp32.dtype != torch.float32 or scale_fp32.device.type != "cpu":
        raise ValueError("scale_fp32 must be a CPU float32 tensor")
    bits = scale_fp32.contiguous().view(torch.int32).to(torch.int64)
    return (bits & 0xFFFFFFFF) | (VDEQ16_MARKER << 32)


def deterministic_operands(m: int, k: int, n: int, with_bias: bool):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + m * 3 + k * 5 + n * 7 + int(with_bias))
    # Small integer values keep the exact CPU oracle comfortably in FP16.
    x = torch.randint(-1, 2, (m, k), dtype=torch.int8, generator=generator)
    weight = torch.randint(-1, 2, (n, k), dtype=torch.int8, generator=generator)
    # Exact binary scales avoid decimal conversion and tie ambiguity.
    scales = torch.pow(2.0, -(torch.arange(n, dtype=torch.float32) % 3))
    encoded_scales = encode_vdeq16_scale(scales)
    bias = (
        torch.randint(-64, 65, (n,), dtype=torch.int32, generator=generator)
        if with_bias
        else None
    )
    return x, weight, scales, encoded_scales, bias


def exact_blocked_integer_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Exact integer accumulation in K0=32 blocks, then defined FP16 dequant."""
    m, k = x.shape
    n = weight.shape[0]
    accumulator = torch.zeros((m, n), dtype=torch.int64)
    for k_start in range(0, k, INTEGER_GOLDEN_K_BLOCK):
        k_end = min(k_start + INTEGER_GOLDEN_K_BLOCK, k)
        accumulator += (
            x[:, k_start:k_end].to(torch.int64)
            @ weight[:, k_start:k_end].to(torch.int64).transpose(0, 1)
        )
    if bias is not None:
        accumulator += bias.to(torch.int64)
    if accumulator.abs().max().item() >= 2048:
        raise AssertionError("small-case golden exceeded exact FP16 integer range")
    return (accumulator.to(torch.float32) * scales.unsqueeze(0)).to(torch.float16)


def load_raw_op(variant: str):
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError(
            "custom-op extension failed to load; rebuild/install host, kernel, "
            "generated ACLNN/op-api, and torch extension from one revision"
        )
    op = getattr(torch.ops._C_ascend, "quant_batch_matmul_v3_x", None)
    if op is None:
        raise RuntimeError("_C_ascend.quant_batch_matmul_v3_x is absent")
    schema = str(torch._C._dispatch_find_schema_or_throw(
        "_C_ascend::quant_batch_matmul_v3_x", ""
    ).schema())
    has_new_attr = "enable_k_pipeline" in schema
    if variant == "old" and has_new_attr:
        raise RuntimeError("--variant old requires the pre-attribute binary/schema")
    if variant != "old" and not has_new_attr:
        raise RuntimeError(
            f"--variant {variant} requires the rebuilt enable_k_pipeline schema; "
            "the boolean will not be passed to this old binary"
        )
    return op, schema


def call_raw(op, x, weight_nz, scale, bias, requested: bool | None):
    kwargs = {"bias": bias, "transpose_x2": True}
    if requested is not None:
        kwargs["enable_k_pipeline"] = requested
    return op(x, weight_nz, scale, **kwargs)


def requested_sequence(variant: str) -> tuple[bool | None, ...]:
    if variant == "old":
        return (None,)
    if variant == "off":
        return (False,)
    if variant == "on":
        return (True,)
    return (False, True, False)


def result_key(name: str, with_bias: bool) -> str:
    return f"{name}|bias={int(with_bias)}"


def expected_result_keys() -> set[str]:
    return {
        result_key(name, with_bias)
        for name, *_ in CASES
        for with_bias in (False, True)
    }


def load_reference(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS" or payload.get("variant") != "old":
        raise ValueError("reference must be a successful --variant old result")
    if payload.get("seed") != SEED:
        raise ValueError("reference seed differs from this runner")
    schema = payload.get("schema")
    if not isinstance(schema, str) or "quant_batch_matmul_v3_x" not in schema:
        raise ValueError("reference does not contain the expected QBMM schema")
    if "enable_k_pipeline" in schema:
        raise ValueError("reference schema is not the pre-attribute schema")
    if payload.get("runner_sha256") != tensor_file_sha256(Path(__file__)):
        raise ValueError("reference was not produced by this exact runner")
    indexed: dict[str, dict[str, Any]] = {}
    for item in payload.get("cases", []):
        key = result_key(item["name"], item["bias"])
        if key in indexed:
            raise ValueError(f"duplicate reference case: {key}")
        if item.get("variant") != "old" or item.get("requested_sequence") != [None]:
            raise ValueError(f"reference case is not an old-binary capture: {key}")
        indexed[key] = item
    if set(indexed) != expected_result_keys():
        missing = sorted(expected_result_keys() - set(indexed))
        extra = sorted(set(indexed) - expected_result_keys())
        raise ValueError(f"reference case matrix differs; missing={missing}, extra={extra}")
    return indexed


def tensor_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(
    op,
    variant: str,
    name: str,
    m: int,
    k: int,
    n: int,
    expected_pipeline: bool,
    with_bias: bool,
    reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    x_cpu, weight_cpu, scales_cpu, encoded_cpu, bias_cpu = deterministic_operands(
        m, k, n, with_bias
    )
    input_hashes = {
        "x": tensor_sha256(x_cpu),
        "weight_nd": tensor_sha256(weight_cpu),
        "scale_encoded": tensor_sha256(encoded_cpu),
        "bias": tensor_sha256(bias_cpu) if bias_cpu is not None else None,
    }
    x = x_cpu.npu()
    weight_nz = torch_npu.npu_format_cast(weight_cpu.npu(), FRACTAL_NZ)
    scale = encoded_cpu.npu()
    bias = bias_cpu.npu() if bias_cpu is not None else None
    x_before = x.clone()
    weight_before = weight_nz.clone()
    scale_before = scale.clone()
    bias_before = bias.clone() if bias is not None else None

    flags = requested_sequence(variant)
    outputs = [call_raw(op, x, weight_nz, scale, bias, flag) for flag in flags]
    torch.npu.synchronize()
    expected_shape = (m, n)
    for index, output in enumerate(outputs):
        if tuple(output.shape) != expected_shape or output.dtype != torch.float16:
            raise AssertionError(
                f"{name}/call{index}: expected FP16 {expected_shape}, got "
                f"{tuple(output.shape)} {output.dtype}"
            )
        if not torch.isfinite(output).all().item():
            raise AssertionError(f"{name}/call{index}: non-finite output")
    for output in outputs[1:]:
        if not bitwise_equal(outputs[0], output):
            raise AssertionError(f"{name}: OFF/ON/OFF bitwise equality failed")

    if not torch.equal(x, x_before):
        raise AssertionError(f"{name}: x was mutated")
    if not torch.equal(weight_nz, weight_before):
        raise AssertionError(f"{name}: weight was mutated")
    if not torch.equal(scale, scale_before):
        raise AssertionError(f"{name}: scale was mutated")
    if bias is not None and not torch.equal(bias, bias_before):
        raise AssertionError(f"{name}: bias was mutated")

    output_cpu = outputs[0].detach().cpu().contiguous()
    output_hash = tensor_sha256(output_cpu)
    reference_match = None
    if reference:
        reference_item = reference.get(result_key(name, with_bias))
        if reference_item is None:
            raise AssertionError(f"{name}: reference entry is missing")
        if input_hashes != reference_item.get("input_sha256"):
            raise AssertionError(f"{name}: deterministic input provenance differs from reference")
        if output_hash != reference_item.get("output_sha256"):
            raise AssertionError(f"{name}: output differs from old custom-op reference")
        reference_match = True

    golden_kind = "old_custom_reference" if reference else "baseline_capture"
    golden_match = None
    if n <= SMALL_GOLDEN_MAX_N:
        golden = exact_blocked_integer_golden(
            x_cpu, weight_cpu, scales_cpu, bias_cpu
        )
        if not bitwise_equal(output_cpu, golden):
            mismatch = torch.count_nonzero(
                output_cpu.view(torch.int16) != golden.view(torch.int16)
            ).item()
            raise AssertionError(f"{name}: exact blocked-integer golden mismatch ({mismatch})")
        golden_kind = "exact_blocked_integer_cpu"
        golden_match = True
    elif reference_match:
        golden_match = True

    result = {
        "name": name,
        "shape": {"M": m, "K": k, "N": n},
        "bias": with_bias,
        "variant": variant,
        "requested_sequence": list(flags),
        # This is an exact-picker expectation, not observed device routing.
        "expected_pipeline_route": expected_pipeline if True in flags else False,
        "observed_pipeline_route": None,
        "finite": True,
        "input_immutable": True,
        "intra_run_bitwise_equal": True,
        "golden_kind": golden_kind,
        "golden_match": golden_match,
        "old_reference_match": reference_match,
        "input_sha256": input_hashes,
        "output_sha256": output_hash,
        "sample_fp16_bits": output_cpu.view(torch.int16).flatten()[:16].tolist(),
    }

    del outputs, output_cpu, x_before, weight_before, scale_before, x, weight_nz, scale
    if bias is not None:
        del bias_before, bias
    torch.npu.empty_cache()
    return result


def check_logical_padding_invariance(op, variant: str) -> list[dict[str, Any]]:
    """Numerical prefix check only; this is not OOB-sanitizer evidence."""
    results = []
    requests = tuple(dict.fromkeys(requested_sequence(variant)))
    for m in (1, 15):
        k, n = 4096, 64
        x_cpu, weight_cpu, _, encoded_cpu, bias_cpu = deterministic_operands(m, k, n, True)
        padded_x = torch.zeros((16, k), dtype=torch.int8)
        padded_x[:m].copy_(x_cpu)
        weight_nz = torch_npu.npu_format_cast(weight_cpu.npu(), FRACTAL_NZ)
        for requested in requests:
            logical = call_raw(
                op, x_cpu.npu(), weight_nz, encoded_cpu.npu(), bias_cpu.npu(), requested
            )
            padded = call_raw(
                op, padded_x.npu(), weight_nz, encoded_cpu.npu(), bias_cpu.npu(), requested
            )
            torch.npu.synchronize()
            if not bitwise_equal(logical, padded[:m]):
                raise AssertionError(
                    f"logical/padded prefix mismatch for M={m}, requested={requested}"
                )
            results.append({
                "M": m,
                "variant": variant,
                "requested_pipeline": requested,
                "bitwise_logical_vs_zero_padded_prefix": True,
                "sanitizer_evidence": False,
            })
            del logical, padded
        del weight_nz
        torch.npu.empty_cache()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("old", "off", "on", "cycle"), required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument(
        "--reference-json",
        type=Path,
        help="old-binary result required as the oracle for every new-schema variant",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("an authorized Ascend NPU environment is required")
    if args.variant != "old" and args.reference_json is None:
        raise RuntimeError(
            f"--variant {args.variant} requires --reference-json from --variant old"
        )
    reference = load_reference(args.reference_json)
    op, schema = load_raw_op(args.variant)
    results = []
    for name, m, k, n, expected_pipeline in CASES:
        for with_bias in (False, True):
            result = run_case(
                op, args.variant, name, m, k, n, expected_pipeline,
                with_bias, reference,
            )
            results.append(result)
            print(json.dumps({"type": "case", **result}, sort_keys=True), flush=True)

    padding = check_logical_padding_invariance(op, args.variant)
    summary = {
        "status": "PASS",
        "variant": args.variant,
        "schema": schema,
        "seed": SEED,
        "runner_sha256": tensor_file_sha256(Path(__file__)),
        "runtime": {
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        },
        "cases": results,
        "logical_padding_invariance": padding,
        "sanitizer_evidence": False,
        "note": (
            "No timing was collected. Padding invariance is not OOB-sanitizer evidence. "
            "Pipeline routing is expected from the exact picker, not observed at runtime."
        ),
    }
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"type": "summary", **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
