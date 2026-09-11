# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auditable batch-one Tree-MTP/linear-MTP benchmark for Ascend 310P.

The public process is a supervisor.  It records configuration before starting a
worker process, so a native abort still leaves a machine-readable report and
captured stdout/stderr.  Importing this module does not import vLLM or torch;
the worker imports them only after configuration and environment validation.
"""

import argparse
import dataclasses
import enum
import hashlib
import json
import numbers
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/home/models/Qwen3.5-9B-w8a8-mtp/"
INPUT_TOKEN_COUNT = 1024
BENCHMARK_OUTPUT_TOKEN_COUNT = 2048
SMOKE_OUTPUT_TOKEN_COUNT = 32
WARMUP_OUTPUT_TOKEN_COUNT = 8
MAX_MODEL_LEN = 4096
MAX_NUM_BATCHED_TOKENS = 2048
REQUEST_SEED = 42
EXPECTED_PROMPT_SHA256 = "51c0e67b7cad2db9cb9c41ae0e1f2eb1b0f77da5f1b22bed3d1cdbd2db439b74"
SUPPORTED_TREE_DEPTHS = frozenset((2, 3, 4))
UNVERIFIED_TREE_DEPTHS = frozenset((5, 6))
SUPPORTED_TREE_WIDTHS = frozenset((1, 2, 4, 8, 16))
SUPPORTED_LINEAR_DEPTHS = frozenset((2, 3, 4, 5, 6))
VALID_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
REPORT_SCHEMA_VERSION = 1
SPEC_METRIC_PREFIX = "vllm:spec_decode_"
VERIFICATION_COUNTER = "vllm:spec_decode_num_drafts"
PROPOSED_COUNTER = "vllm:spec_decode_num_draft_tokens"
ACCEPTED_COUNTER = "vllm:spec_decode_num_accepted_tokens"
OPTIMIZATION_ENV_KEYS = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_CUSTOM_QBMM",
    "VLLM_ASCEND_TREE_GDN_COMPACT",
    "VLLM_LMHEAD_PRUNE_PACK",
    "VLLM_BATCH_INVARIANT",
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "PYTORCH_NPU_ALLOC_CONF",
    "ASCEND_LAUNCH_BLOCKING",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
VERSION_DISTRIBUTIONS = (
    "vllm",
    "vllm-ascend",
    "torch",
    "torch-npu",
    "transformers",
    "safetensors",
)


@dataclasses.dataclass(frozen=True)
class BenchmarkPlan:
    mode: str
    depth: int
    width: int | None
    speculative_tokens: int
    sampling: str
    stage: str
    repetitions: int
    input_tokens: int
    output_tokens: int
    performance_eligible: bool
    blocked_reason: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("tree", "linear-eager", "linear-graph"))
    parser.add_argument("--depth", required=True, type=int, help="Tree depth or linear speculative-token count")
    parser.add_argument("--width", type=int, help="Required for tree; forbidden for linear controls")
    parser.add_argument("--sampling", required=True, choices=("greedy", "t1"))
    parser.add_argument("--stage", required=True, choices=("smoke", "screening", "confirmation"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--physical-device",
        required=True,
        type=int,
        help="Host NPU mounted into the container; provenance only",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Logical device inside the container; must match ASCEND_RT_VISIBLE_DEVICES",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--safetensors-load-strategy", choices=("lazy", "eager"), default="eager")
    parser.add_argument("--tree-trace", action="store_true", help="Enable verbose tree trace records")
    parser.add_argument("--expected-prompt-sha256", default=EXPECTED_PROMPT_SHA256)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def build_plan(args: argparse.Namespace) -> BenchmarkPlan:
    if not VALID_RUN_ID.fullmatch(args.run_id):
        raise ValueError("run-id must contain only letters, digits, dot, underscore, or dash")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be strictly between zero and one")
    if args.physical_device < 0 or args.device < 0:
        raise ValueError("physical and logical device indices must be non-negative")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_prompt_sha256):
        raise ValueError("expected-prompt-sha256 must be 64 lowercase hexadecimal characters")

    blocked_reason = None
    if args.mode == "tree":
        if args.width is None:
            raise ValueError("tree mode requires --width")
        if args.width not in SUPPORTED_TREE_WIDTHS:
            raise ValueError(f"tree width must be one of {sorted(SUPPORTED_TREE_WIDTHS)}")
        speculative_tokens = args.depth * args.width
        if args.depth in UNVERIFIED_TREE_DEPTHS:
            blocked_reason = (
                f"Tree depth {args.depth} is not enabled: native correctness and bounds have not been verified. "
                "The requested depth is recorded unchanged; the harness never clamps it."
            )
        elif args.depth not in SUPPORTED_TREE_DEPTHS:
            raise ValueError(f"tree depth must be one of {sorted(SUPPORTED_TREE_DEPTHS | UNVERIFIED_TREE_DEPTHS)}")
    else:
        if args.width is not None:
            raise ValueError("linear controls do not accept --width")
        if args.depth not in SUPPORTED_LINEAR_DEPTHS:
            raise ValueError(f"linear speculative-token count must be one of {sorted(SUPPORTED_LINEAR_DEPTHS)}")
        speculative_tokens = args.depth

    output_tokens = SMOKE_OUTPUT_TOKEN_COUNT if args.stage == "smoke" else BENCHMARK_OUTPUT_TOKEN_COUNT
    return BenchmarkPlan(
        mode=args.mode,
        depth=args.depth,
        width=args.width,
        speculative_tokens=speculative_tokens,
        sampling=args.sampling,
        stage=args.stage,
        repetitions=args.repetitions,
        input_tokens=INPUT_TOKEN_COUNT,
        output_tokens=output_tokens,
        performance_eligible=args.stage != "smoke",
        blocked_reason=blocked_reason,
    )


def sampling_parameters(preset: str) -> dict[str, Any]:
    if preset == "greedy":
        return {"temperature": 0.0, "top_k": -1, "top_p": 1.0, "seed": REQUEST_SEED}
    if preset == "t1":
        return {"temperature": 1.0, "top_k": 50, "top_p": 0.9, "seed": REQUEST_SEED}
    raise ValueError(f"Unknown sampling preset: {preset}")


def build_engine_args(args: argparse.Namespace, plan: BenchmarkPlan) -> dict[str, Any]:
    additional_config: dict[str, Any] = {
        "ascend_compilation_config": {"enable_npugraph_ex": False},
    }
    if plan.mode == "tree":
        additional_config["tree_mtp"] = {
            "enabled": True,
            "width": plan.width,
            "depth": plan.depth,
            "trace": args.tree_trace,
        }

    engine_args: dict[str, Any] = {
        "model": args.model,
        "dtype": "float16",
        "quantization": "ascend",
        "mamba_ssm_cache_dtype": "float16",
        "mamba_cache_mode": "none",
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "uni",
        "enforce_eager": plan.mode != "linear-graph",
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "max_num_seqs": 1,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "language_model_only": True,
        "skip_mm_profiling": True,
        "seed": 0,
        "disable_log_stats": False,
        "safetensors_load_strategy": args.safetensors_load_strategy,
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": plan.speculative_tokens,
        },
        "additional_config": additional_config,
    }
    if plan.mode == "linear-graph":
        engine_args["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    return engine_args


def construct_exact_prompt(tokenizer: Any, token_count: int = INPUT_TOKEN_COUNT) -> list[int]:
    marker = "TREE_MTP_EXACT_INPUT_MARKER"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if template.count(marker) != 1:
        raise ValueError("Chat template did not preserve the exact-input marker exactly once")
    prefix, suffix = template.split(marker)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(
        "\nUsing the notes above, write a detailed guide to designing reliable automated software tests."
        " Include explanations and practical Python examples.\n"
        + suffix,
        add_special_tokens=False,
    )
    filler_ids = tokenizer.encode(
        "Technical note: A test should be repeatable, isolated and easy to understand. "
        "Check normal cases, boundary cases and invalid input.\n",
        add_special_tokens=False,
    )
    fill_count = token_count - len(prefix_ids) - len(suffix_ids)
    if fill_count < 0 or not filler_ids:
        raise ValueError("Exact input is too short for the fixed template and question")
    repeats = (fill_count + len(filler_ids) - 1) // len(filler_ids)
    prompt_token_ids = prefix_ids + (filler_ids * repeats)[:fill_count] + suffix_ids
    if len(prompt_token_ids) != token_count:
        raise AssertionError("Exact-token prompt construction failed")
    return prompt_token_ids


def prompt_sha256(prompt_token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(prompt_token_ids).encode("utf-8")).hexdigest()


def _class_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _json_safe(value: Any, depth: int = 0) -> Any:
    if isinstance(value, enum.Enum):
        return {"class": _class_name(value), "name": value.name, "value": _json_safe(value.value, depth + 1)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth > 8:
        return {"class": _class_name(value), "repr": repr(value)[:1000]}
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value), depth + 1)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth + 1) for item in value]
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return _tensor_metadata(value)
    if hasattr(value, "value") and isinstance(getattr(value, "value", None), (str, int, float, bool)):
        return _json_safe(getattr(value, "value"), depth + 1)
    return {"class": _class_name(value), "repr": repr(value)[:4000]}


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    shape = getattr(tensor, "shape", None)
    try:
        shape_value = [int(dimension) for dimension in shape] if shape is not None else None
    except (TypeError, ValueError):
        shape_value = repr(shape)
    device = getattr(tensor, "device", None)
    return {
        "class": _class_name(tensor),
        "dtype": str(getattr(tensor, "dtype", None)),
        "shape": shape_value,
        "device": str(device) if device is not None else None,
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }


def _attribute_path(root: Any, path: str) -> Any:
    value = root
    for name in path.split("."):
        value = getattr(value, name)
    return value


def _find_model_runner(root: Any) -> tuple[Any | None, str | None]:
    direct_paths = (
        "model_runner",
        "worker.model_runner",
        "driver_worker.model_runner",
        "driver_worker.worker.model_runner",
        "model_executor.driver_worker.model_runner",
        "model_executor.driver_worker.worker.model_runner",
        "llm_engine.model_executor.driver_worker.model_runner",
        "llm_engine.model_executor.driver_worker.worker.model_runner",
        "llm_engine.engine_core.model_executor.driver_worker.model_runner",
        "llm_engine.engine_core.model_executor.driver_worker.worker.model_runner",
        "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner",
        "llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner",
    )
    for path in direct_paths:
        try:
            candidate = _attribute_path(root, path)
        except (AttributeError, RuntimeError):
            continue
        if candidate is not None and hasattr(candidate, "model"):
            return candidate, path
    return None, None


def _find_lm_head(model: Any) -> tuple[Any | None, str | None]:
    owner = model
    owner_path = ""
    for _ in range(4):
        head = getattr(owner, "lm_head", None)
        if head is not None:
            locator = f"{owner_path}.lm_head" if owner_path else "lm_head"
            return head, locator
        next_owner = getattr(owner, "language_model", None)
        next_name = "language_model"
        if next_owner is None:
            next_owner = getattr(owner, "model", None)
            next_name = "model"
        if next_owner is None or next_owner is owner:
            break
        owner = next_owner
        owner_path = f"{owner_path}.{next_name}" if owner_path else next_name
    for path in ("language_model.lm_head", "model.lm_head", "model.model.lm_head"):
        try:
            head = _attribute_path(model, path)
        except (AttributeError, RuntimeError):
            continue
        if head is not None:
            return head, path
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        matches = []
        try:
            for name, module in named_modules():
                if name == "lm_head" or name.endswith(".lm_head"):
                    matches.append((name, module))
        except (AttributeError, RuntimeError, TypeError):
            matches = []
        if matches:
            return min(matches, key=lambda match: (match[0].count("."), len(match[0])))[1], min(
                matches, key=lambda match: (match[0].count("."), len(match[0]))
            )[0]
    return None, None


def _read_selected_attributes(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    selected = {}
    if value is None:
        return selected
    for name in names:
        try:
            attribute = getattr(value, name)
        except (AttributeError, RuntimeError):
            continue
        if callable(attribute):
            continue
        selected[name] = _json_safe(attribute)
    return selected


def _head_metadata(model: Any, role: str) -> dict[str, Any]:
    if model is None:
        return {"role": role, "status": "UNAVAILABLE", "reason": "runtime model object was not found"}
    head, locator = _find_lm_head(model)
    if head is None:
        return {
            "role": role,
            "status": "UNAVAILABLE",
            "model_class": _class_name(model),
            "reason": "runtime lm_head module was not found",
        }
    try:
        weight = getattr(head, "weight")
    except (AttributeError, RuntimeError):
        weight = None
    quant_method = getattr(head, "quant_method", None)
    metadata = {
        "role": role,
        "status": "AVAILABLE" if weight is not None else "UNAVAILABLE",
        "model_class": _class_name(model),
        "head_locator": locator,
        "head_class": _class_name(head),
        "weight": _tensor_metadata(weight) if weight is not None else None,
        "quant_method_class": _class_name(quant_method) if quant_method is not None else None,
        "quant_method_repr": repr(quant_method)[:4000] if quant_method is not None else None,
        "quant_method_instance_state": (
            _json_safe(vars(quant_method)) if quant_method is not None and hasattr(quant_method, "__dict__") else None
        ),
        "quant_method": _read_selected_attributes(
            quant_method,
            (
                "quant_type",
                "quantization",
                "in_dtype",
                "out_dtype",
                "group_size",
                "is_monolithic",
            ),
        ),
        "head_quant_fields": _read_selected_attributes(
            head,
            ("quant_type", "quantization", "input_dtype", "output_dtype", "is_monolithic"),
        ),
    }
    for tensor_name in ("weight_nz", "bias", "deq_scale", "quant_bias", "input_scale"):
        tensor = getattr(head, tensor_name, None)
        if tensor is not None:
            metadata[tensor_name] = _tensor_metadata(tensor)
    if weight is None:
        metadata["reason"] = "runtime lm_head exists but exposes no weight tensor"
    return metadata


def _draft_weight_metadata(model: Any) -> list[dict[str, Any]]:
    """Distinguish the MTP network's linears from its possibly shared lm_head."""
    if model is None or not callable(getattr(model, "named_modules", None)):
        return []
    result = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or len(getattr(weight, "shape", ())) < 2:
            continue
        quant_method = getattr(module, "quant_method", None)
        result.append({
            "name": name,
            "module_class": _class_name(module),
            "weight": _tensor_metadata(weight),
            "weight_nz": (
                _tensor_metadata(module.weight_nz) if getattr(module, "weight_nz", None) is not None else None
            ),
            "quant_method_class": _class_name(quant_method) if quant_method is not None else None,
        })
    return result


def _effective_config_metadata(worker: Any, runner: Any) -> dict[str, Any]:
    vllm_config = getattr(worker, "vllm_config", None) or getattr(runner, "vllm_config", None)
    if vllm_config is None:
        return {"status": "UNAVAILABLE", "reason": "worker/model runner exposes no vllm_config"}
    sections = {
        "model_config": (
            "model",
            "dtype",
            "quantization",
            "max_model_len",
            "enforce_eager",
        ),
        "speculative_config": ("method", "num_speculative_tokens"),
        "compilation_config": ("level", "cudagraph_mode", "max_capture_size"),
        "scheduler_config": ("max_num_seqs", "max_num_batched_tokens", "async_scheduling"),
        "parallel_config": ("tensor_parallel_size", "distributed_executor_backend"),
        "cache_config": ("mamba_cache_mode", "mamba_ssm_cache_dtype", "enable_prefix_caching"),
    }
    result: dict[str, Any] = {"status": "AVAILABLE", "vllm_config_class": _class_name(vllm_config)}
    for section, names in sections.items():
        result[section] = _read_selected_attributes(getattr(vllm_config, section, None), names)
    result["additional_config"] = _json_safe(getattr(vllm_config, "additional_config", None))
    return result


def _worker_runtime_metadata(worker: Any) -> dict[str, Any]:
    runner, runner_locator = _find_model_runner(worker)
    if runner is None:
        return {
            "status": "UNAVAILABLE",
            "source": "runtime_worker",
            "worker_class": _class_name(worker),
            "reason": "model runner was not reachable from the runtime worker",
        }
    target_model = getattr(runner, "model", None)
    drafter = getattr(runner, "drafter", None)
    mtp_model = getattr(drafter, "model", None) if drafter is not None else None
    target = _head_metadata(target_model, "target")
    mtp = _head_metadata(mtp_model, "mtp")
    available = target["status"] == "AVAILABLE" and mtp["status"] == "AVAILABLE"
    return {
        "status": "AVAILABLE" if available else "UNAVAILABLE",
        "source": "runtime_worker",
        "worker_class": _class_name(worker),
        "model_runner_class": _class_name(runner),
        "model_runner_locator": runner_locator,
        "drafter_class": _class_name(drafter) if drafter is not None else None,
        "target_head": target,
        "mtp_head": mtp,
        "mtp_weight_modules": _draft_weight_metadata(mtp_model),
        "target_and_mtp_head_same_object": (
            target_model is not None
            and mtp_model is not None
            and _find_lm_head(target_model)[0] is _find_lm_head(mtp_model)[0]
        ),
        "effective_config": _effective_config_metadata(worker, runner),
    }


def collect_runtime_metadata(llm: Any) -> dict[str, Any]:
    direct = _worker_runtime_metadata(llm)
    if direct.get("status") == "AVAILABLE":
        direct["transport"] = "direct_in_process"
        return direct
    errors = [{"transport": "direct_in_process", "result": direct}]
    collective_rpc = getattr(llm, "collective_rpc", None)
    if callable(collective_rpc):
        try:
            results = collective_rpc(_worker_runtime_metadata)
            if not isinstance(results, list):
                results = [results]
            normalized = [_json_safe(result) for result in results]
            if results and all(isinstance(result, dict) and result.get("status") == "AVAILABLE" for result in results):
                return {"status": "AVAILABLE", "transport": "collective_rpc", "workers": normalized}
            errors.append({"transport": "collective_rpc", "result": normalized})
        except Exception as error:  # Runtime API/version differences are reportable evidence.
            errors.append(
                {
                    "transport": "collective_rpc",
                    "error_type": _class_name(error),
                    "error": str(error),
                }
            )
    return {
        "status": "UNAVAILABLE",
        "reason": "actual target and MTP lm_head metadata could not be obtained from the loaded runtime",
        "attempts": errors,
    }


def effective_config_contract(runtime_metadata: dict[str, Any], plan: BenchmarkPlan) -> dict[str, Any]:
    worker_records = runtime_metadata.get("workers")
    if not isinstance(worker_records, list):
        worker_records = [runtime_metadata]
    expected_graph = "FULL_DECODE_ONLY" if plan.mode == "linear-graph" else "NONE"
    workers = []
    for index, worker_record in enumerate(worker_records):
        effective = worker_record.get("effective_config", {}) if isinstance(worker_record, dict) else {}
        speculative = effective.get("speculative_config", {})
        compilation = effective.get("compilation_config", {})
        cache = effective.get("cache_config", {})
        observed = {
            "method": speculative.get("method"),
            "num_speculative_tokens": speculative.get("num_speculative_tokens"),
            "cudagraph_mode": compilation.get("cudagraph_mode"),
            "mamba_cache_mode": cache.get("mamba_cache_mode"),
            "enable_prefix_caching": cache.get("enable_prefix_caching"),
        }
        graph_text = json.dumps(observed["cudagraph_mode"], sort_keys=True).upper()
        checks = {
            "method_is_mtp": observed["method"] == "mtp",
            "speculative_tokens_exact": observed["num_speculative_tokens"] == plan.speculative_tokens,
            "cudagraph_mode_exact": expected_graph in graph_text,
            "mamba_cache_mode_none": observed["mamba_cache_mode"] == "none",
            "prefix_caching_disabled": observed["enable_prefix_caching"] is False,
        }
        workers.append(
            {
                "worker_index": index,
                "observed": observed,
                "expected": {
                    "method": "mtp",
                    "num_speculative_tokens": plan.speculative_tokens,
                    "cudagraph_mode": expected_graph,
                    "mamba_cache_mode": "none",
                    "enable_prefix_caching": False,
                },
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    return {
        "status": "PASS" if workers and all(worker["status"] == "PASS" for worker in workers) else "FAIL",
        "workers": workers,
    }


def _metric_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return None


def snapshot_spec_metrics(llm: Any) -> dict[str, Any]:
    counters: dict[str, int | float] = {}
    raw = []
    for metric in llm.get_metrics():
        name = getattr(metric, "name", None)
        if not isinstance(name, str) or not name.startswith(SPEC_METRIC_PREFIX):
            continue
        value = getattr(metric, "value", None)
        numeric_value = _metric_number(value)
        raw.append(
            {
                "class": _class_name(metric),
                "name": name,
                "value": numeric_value if numeric_value is not None else _json_safe(value),
                "labels": _json_safe(getattr(metric, "labels", None)),
            }
        )
        if numeric_value is not None:
            counters[name] = counters.get(name, 0) + numeric_value
    return {"counters": counters, "raw": raw}


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | float]:
    before_counters = before.get("counters", {})
    after_counters = after.get("counters", {})
    return {
        name: after_counters.get(name, 0) - before_counters.get(name, 0)
        for name in sorted(set(before_counters) | set(after_counters))
    }


def _first_timestamp(metrics: Any, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = getattr(metrics, name, None)
        if isinstance(value, numbers.Real):
            return float(value)
    return None


def derive_timing_and_acceptance(
    metrics: Any,
    output_token_count: int,
    elapsed_s: float,
    delta: dict[str, int | float],
) -> dict[str, Any]:
    arrival = _first_timestamp(metrics, ("arrival_time", "arrival_ts", "request_arrival_time"))
    first = _first_timestamp(metrics, ("first_token_ts", "first_token_time"))
    last = _first_timestamp(metrics, ("last_token_ts", "last_token_time"))
    ttft_ms = (first - arrival) * 1000 if arrival is not None and first is not None else None
    decode_span_ms = (last - first) * 1000 if first is not None and last is not None else None
    tpot_ms = (
        decode_span_ms / (output_token_count - 1)
        if decode_span_ms is not None and output_token_count > 1
        else None
    )
    verification_cycles = delta.get(VERIFICATION_COUNTER)
    proposed_tokens = delta.get(PROPOSED_COUNTER)
    accepted_tokens = delta.get(ACCEPTED_COUNTER)
    valid_cycles = (
        verification_cycles
        if isinstance(verification_cycles, numbers.Real) and verification_cycles > 0
        else None
    )
    valid_proposals = proposed_tokens if isinstance(proposed_tokens, numbers.Real) and proposed_tokens > 0 else None
    valid_accepted = accepted_tokens if isinstance(accepted_tokens, numbers.Real) else None
    return {
        "ttft_ms": ttft_ms,
        "decode_span_ms_excluding_prefill": decode_span_ms,
        "tpot_ms_excluding_prefill": tpot_ms,
        "generate_wall_s_including_prefill": elapsed_s,
        "wall_ms_per_output_token_including_prefill": (
            elapsed_s * 1000 / output_token_count if output_token_count else None
        ),
        "verification_cycles": verification_cycles,
        "proposed_draft_tokens": proposed_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "accepted_drafts_per_verification": (
            valid_accepted / valid_cycles if valid_accepted is not None and valid_cycles else None
        ),
        "emitted_tokens_per_verification_actual": output_token_count / valid_cycles if valid_cycles else None,
        "emitted_tokens_per_verification_counter_derived": (
            1 + valid_accepted / valid_cycles if valid_accepted is not None and valid_cycles else None
        ),
        "draft_acceptance_rate": (
            valid_accepted / valid_proposals if valid_accepted is not None and valid_proposals else None
        ),
        "mean_decode_span_ms_per_verification": (
            decode_span_ms / valid_cycles if decode_span_ms and valid_cycles else None
        ),
    }


def _request_row(
    output: Any,
    elapsed_s: float,
    before: dict[str, Any],
    after: dict[str, Any],
    expected_prompt_ids: list[int],
    max_tokens: int,
    label: str,
    repetition: int | None,
) -> dict[str, Any]:
    completion = output.outputs[0]
    output_token_ids = list(completion.token_ids)
    actual_prompt_ids = list(output.prompt_token_ids)
    metrics = output.metrics
    delta = counter_delta(before, after)
    row = {
        "label": label,
        "repetition": repetition,
        "requested_prompt_token_count": len(expected_prompt_ids),
        "actual_prompt_token_count": len(actual_prompt_ids),
        "requested_output_token_count": max_tokens,
        "actual_output_token_count": len(output_token_ids),
        "prompt_token_sha256": prompt_sha256(actual_prompt_ids),
        "output_token_ids": output_token_ids,
        "output_text": completion.text,
        "request_metrics": _json_safe(metrics),
        "spec_metrics_before": before,
        "spec_metrics_after": after,
        "spec_counter_delta": delta,
        "metrics_evidence": {
            "required_counters": [VERIFICATION_COUNTER, ACCEPTED_COUNTER],
            "optional_counters": [PROPOSED_COUNTER],
            "missing_required_counters": [
                name for name in (VERIFICATION_COUNTER, ACCEPTED_COUNTER) if name not in delta
            ],
        },
        "request_contract": {
            "prompt_token_ids_exact": actual_prompt_ids == expected_prompt_ids,
            "output_token_count_exact": len(output_token_ids) == max_tokens,
        },
    }
    row["metrics_evidence"]["status"] = (
        "AVAILABLE" if not row["metrics_evidence"]["missing_required_counters"] else "INCOMPLETE"
    )
    row.update(derive_timing_and_acceptance(metrics, len(output_token_ids), elapsed_s, delta))
    row["request_contract"]["status"] = (
        "PASS" if all(row["request_contract"].values()) else "FAIL"
    )
    if actual_prompt_ids != expected_prompt_ids:
        row["actual_prompt_token_ids_on_mismatch"] = actual_prompt_ids
    return row


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_command(repository: Path, arguments: list[str], binary: bool = False) -> bytes | str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def collect_git_state(repository: Path) -> dict[str, Any]:
    head = _git_command(repository, ["rev-parse", "HEAD"])
    branch = _git_command(repository, ["branch", "--show-current"])
    status = _git_command(repository, ["status", "--porcelain=v1", "--untracked-files=all"])
    diff = _git_command(repository, ["diff", "--binary", "HEAD", "--"], binary=True)
    status_text = status.strip() if isinstance(status, str) else None
    return {
        "repository": str(repository),
        "head": head.strip() if isinstance(head, str) else None,
        "branch": branch.strip() if isinstance(branch, str) else None,
        "dirty": bool(status_text) if status_text is not None else None,
        "status_porcelain_v1": status_text.splitlines() if status_text else [],
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if isinstance(diff, bytes) else None,
        "tracked_diff_bytes": len(diff) if isinstance(diff, bytes) else None,
    }


def collect_versions() -> dict[str, Any]:
    packages = {}
    for distribution in VERSION_DISTRIBUTIONS:
        try:
            installed = importlib_metadata.distribution(distribution)
            direct_url_text = installed.read_text("direct_url.json")
            try:
                direct_url = json.loads(direct_url_text) if direct_url_text else None
            except json.JSONDecodeError:
                direct_url = {"raw": direct_url_text}
            packages[distribution] = {
                "version": installed.version,
                "location": str(installed.locate_file("")),
                "direct_url": direct_url,
            }
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def runtime_versions(torch_module: Any, torch_npu_module: Any, vllm_module: Any) -> dict[str, Any]:
    versions = {
        "torch.__version__": str(getattr(torch_module, "__version__", None)),
        "torch_npu.__version__": str(getattr(torch_npu_module, "__version__", None)),
        "vllm.__version__": str(getattr(vllm_module, "__version__", None)),
    }
    try:
        from vllm import _version as vllm_version

        versions["vllm._version"] = _read_selected_attributes(
            vllm_version,
            ("__version__", "__version_tuple__", "__commit_id__", "__git_revision__"),
        )
    except (ImportError, AttributeError):
        versions["vllm._version"] = None
    return versions


def source_hashes(repository: Path) -> dict[str, str | None]:
    paths = (
        Path(__file__).resolve(),
        repository / "examples" / "tree_mtp_310p_benchmark.py",
        repository / "docs" / "en" / "developer_guide" / "experimental_tree_mtp_310p.md",
    )
    result = {}
    for path in paths:
        try:
            result[str(path.relative_to(repository))] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError):
            result[str(path)] = None
    return result


def environment_contract(args: argparse.Namespace, plan: BenchmarkPlan) -> dict[str, Any]:
    environment = {key: os.environ.get(key) for key in OPTIMIZATION_ENV_KEYS}
    all_vllm_environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("VLLM_")
    }
    checks = {
        "logical_visibility_matches": environment["ASCEND_RT_VISIBLE_DEVICES"] == str(args.device),
        "custom_qbmm_enabled": environment["VLLM_CUSTOM_QBMM"] == "1",
        "compact_gdn_explicit": environment["VLLM_ASCEND_TREE_GDN_COMPACT"] == "1",
        "lmhead_prune_disabled": environment["VLLM_LMHEAD_PRUNE_PACK"] in (None, ""),
    }
    return {
        "values": environment,
        "all_vllm_environment": all_vllm_environment,
        "required": {
            "ASCEND_RT_VISIBLE_DEVICES": str(args.device),
            "VLLM_CUSTOM_QBMM": "1",
            "VLLM_ASCEND_TREE_GDN_COMPACT": "1",
            "VLLM_LMHEAD_PRUNE_PACK": "unset/empty",
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "semantics": {
            "VLLM_CUSTOM_QBMM": "active for every benchmark mode",
            "VLLM_ASCEND_TREE_GDN_COMPACT": (
                "active for tree mode" if plan.mode == "tree" else "explicitly set but inert for this linear control"
            ),
            "VLLM_LMHEAD_PRUNE_PACK": "kept off pending verified runtime target/MTP head compatibility",
        },
    }


def _base_report(args: argparse.Namespace, plan: BenchmarkPlan, argv: list[str]) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "STARTED",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [sys.executable, str(Path(__file__).resolve()), *argv],
        "requested_config": {
            "mode": plan.mode,
            "depth": plan.depth,
            "width": plan.width,
            "speculative_tokens": plan.speculative_tokens,
            "sampling_preset": plan.sampling,
            "stage": plan.stage,
            "repetitions": plan.repetitions,
            "model": args.model,
            "input_tokens": plan.input_tokens,
            "output_tokens": plan.output_tokens,
            "batch_size": 1,
            "expected_prompt_sha256": args.expected_prompt_sha256,
            "performance_eligible": plan.performance_eligible,
        },
        "device_mapping": {
            "physical_host_device": args.physical_device,
            "logical_container_device": args.device,
            "mapping_source": "explicit command-line provenance",
        },
        "sampling_params": sampling_parameters(plan.sampling),
        "engine_args": build_engine_args(args, plan),
        "environment_contract": environment_contract(args, plan),
        "versions": collect_versions(),
        "git": collect_git_state(repository),
        "source_sha256": source_hashes(repository),
        "replication": {
            "engine_lifetime": "one worker engine shared by warmup and every recorded repetition",
            "independent_engine_initializations": 1,
            "statistical_independence": "not claimed",
        },
        "runs": [],
    }


def _run_one_request(
    llm: Any,
    torch_module: Any,
    sampling_params_class: Any,
    sampling_params: dict[str, Any],
    prompt_token_ids: list[int],
    max_tokens: int,
    label: str,
    repetition: int | None,
) -> dict[str, Any]:
    parameters = sampling_params_class(
        **sampling_params,
        max_tokens=max_tokens,
        min_tokens=0,
        ignore_eos=True,
    )
    before = snapshot_spec_metrics(llm)
    torch_module.npu.synchronize()
    started = time.perf_counter()
    outputs = llm.generate([{"prompt_token_ids": prompt_token_ids}], parameters, use_tqdm=False)
    torch_module.npu.synchronize()
    elapsed_s = time.perf_counter() - started
    after = snapshot_spec_metrics(llm)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("Batch-one/single-completion contract was not met")
    return _request_row(
        outputs[0],
        elapsed_s,
        before,
        after,
        prompt_token_ids,
        max_tokens,
        label,
        repetition,
    )
def run_worker(args: argparse.Namespace, plan: BenchmarkPlan, output: Path, argv: list[str]) -> int:
    report = _base_report(args, plan, argv)
    report["worker"] = {"pid": os.getpid(), "status": "STARTED"}
    _write_json(output, report)
    started = time.perf_counter()
    try:
        if report["environment_contract"]["status"] != "PASS":
            raise RuntimeError(
                "Experiment environment contract failed: "
                + json.dumps(report["environment_contract"]["checks"], sort_keys=True)
            )

        import torch
        import torch_npu
        import vllm
        from vllm import LLM, SamplingParams

        report["runtime_versions"] = runtime_versions(torch, torch_npu, vllm)
        llm = LLM(**report["engine_args"])
        report["initialization_s"] = time.perf_counter() - started
        report["runtime_model_evidence"] = collect_runtime_metadata(llm)
        report["effective_config_contract"] = effective_config_contract(report["runtime_model_evidence"], plan)
        _write_json(output, report)
        if report["runtime_model_evidence"]["status"] != "AVAILABLE":
            raise RuntimeError(
                "Required actual target/MTP lm_head dtype and quantization evidence is unavailable; "
                "checkpoint metadata is not substituted"
            )
        if report["effective_config_contract"]["status"] != "PASS":
            raise RuntimeError(
                "Loaded runtime configuration does not match the requested MTP/cache/graph contract: "
                + json.dumps(report["effective_config_contract"], sort_keys=True)
            )

        prompt_token_ids = construct_exact_prompt(llm.get_tokenizer())
        actual_hash = prompt_sha256(prompt_token_ids)
        report["prompt"] = {
            "token_count": len(prompt_token_ids),
            "token_sha256": actual_hash,
            "expected_token_sha256": args.expected_prompt_sha256,
            "token_ids": prompt_token_ids,
        }
        _write_json(output, report)
        if actual_hash != args.expected_prompt_sha256:
            raise RuntimeError(
                f"Deterministic prompt hash mismatch: expected {args.expected_prompt_sha256}, observed {actual_hash}"
            )

        report["warmup"] = _run_one_request(
            llm,
            torch,
            SamplingParams,
            report["sampling_params"],
            prompt_token_ids,
            WARMUP_OUTPUT_TOKEN_COUNT,
            "warmup",
            None,
        )
        _write_json(output, report)
        if report["warmup"]["request_contract"]["status"] != "PASS":
            raise RuntimeError("Warmup prompt/output token-count contract was not met")
        for repetition in range(1, plan.repetitions + 1):
            row = _run_one_request(
                llm,
                torch,
                SamplingParams,
                report["sampling_params"],
                prompt_token_ids,
                plan.output_tokens,
                f"{args.run_id}-rep-{repetition:02d}",
                repetition,
            )
            report["runs"].append(row)
            _write_json(output, report)
            print("INTEGRATED_TREE_MTP_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
            if row["request_contract"]["status"] != "PASS":
                raise RuntimeError(
                    "Measured prompt/output token-count contract was not met: "
                    + repr(row["request_contract"])
                )
            if row["metrics_evidence"]["status"] != "AVAILABLE":
                raise RuntimeError(
                    "Required speculative-decoding counters are missing: "
                    + repr(row["metrics_evidence"]["missing_required_counters"])
                )
        report["status"] = "PASS"
        report["worker"]["status"] = "PASS"
        report["elapsed_s"] = time.perf_counter() - started
        _write_json(output, report)
        return 0
    except Exception as error:
        report["status"] = "ERROR"
        report["worker"]["status"] = "ERROR"
        report["elapsed_s"] = time.perf_counter() - started
        report["error"] = {
            "type": _class_name(error),
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(output, report)
        traceback.print_exc()
        return 1


def _tail(path: Path, limit: int = 16000) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text[-limit:]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_supervisor(args: argparse.Namespace, plan: BenchmarkPlan, argv: list[str]) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    report = _base_report(args, plan, argv)
    if plan.blocked_reason is not None:
        report["status"] = "BLOCKED_UNSUPPORTED"
        report["blocked_reason"] = plan.blocked_reason
        _write_json(output, report)
        print(f"BLOCKED_UNSUPPORTED {plan.blocked_reason}", flush=True)
        return 0
    if report["environment_contract"]["status"] != "PASS":
        report["status"] = "ERROR"
        report["error"] = {
            "type": "EnvironmentContractError",
            "message": "Required experiment environment flags are absent or inconsistent",
        }
        _write_json(output, report)
        return 2

    worker_output = output.with_name(f"{output.name}.worker.json")
    stdout_path = output.with_name(f"{output.name}.stdout.log")
    stderr_path = output.with_name(f"{output.name}.stderr.log")
    collisions = [path for path in (worker_output, stdout_path, stderr_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite worker artifacts: {collisions}")
    report["artifacts"] = {
        "worker_report": str(worker_output),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    _write_json(output, report)

    command = [
        sys.executable,
        "-P",
        str(Path(__file__).resolve()),
        *argv,
        "--_worker",
        "--_worker-output",
        str(worker_output),
    ]
    started = time.perf_counter()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            process = subprocess.run(command, stdout=stdout_file, stderr=stderr_file, check=False)
        returncode = process.returncode
    except Exception as error:
        report["status"] = "ERROR"
        report["supervisor"] = {
            "elapsed_s": time.perf_counter() - started,
            "error_type": _class_name(error),
            "error": str(error),
        }
        _write_json(output, report)
        raise

    worker_report = _load_json(worker_output)
    if worker_report is not None:
        final_report = worker_report
        final_report["artifacts"] = report["artifacts"]
    else:
        final_report = report
    final_report["supervisor"] = {
        "pid": os.getpid(),
        "command": command,
        "returncode": returncode,
        "elapsed_s": time.perf_counter() - started,
        "stdout_tail": _tail(stdout_path),
        "stderr_tail": _tail(stderr_path),
    }
    worker_status = worker_report.get("status") if worker_report is not None else None
    if returncode == 0 and worker_status == "PASS":
        final_report["status"] = "PASS"
    elif worker_status == "ERROR":
        final_report["status"] = "ERROR"
    else:
        final_report["worker_status_before_exit"] = worker_status
        final_report["status"] = "CRASH"
        final_report["error"] = {
            "type": "WorkerProcessCrash",
            "message": f"Worker exited with return code {returncode} without a complete ERROR report",
        }
    _write_json(output, final_report)
    return 0 if final_report["status"] == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(effective_argv)
    plan = build_plan(args)
    if args._worker:
        if args._worker_output is None:
            raise ValueError("internal worker invocation requires --_worker-output")
        return run_worker(args, plan, args._worker_output.resolve(), effective_argv)
    return run_supervisor(args, plan, effective_argv)


if __name__ == "__main__":
    raise SystemExit(main())
