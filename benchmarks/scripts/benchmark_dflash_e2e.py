#!/usr/bin/env python3
"""Measure a single-request DFlash workload and report TTFT, TPOT and acceptance.

The implementation deliberately uses the engine step API.  This records the time at
which the first output becomes visible and reads speculative-decoding counters from
the same engine, rather than estimating acceptance from the generated text.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _set_cpu_affinity(cpu_list: str) -> list[int] | None:
    if not cpu_list:
        return None
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("--process-cpus is supported only on Linux")
    requested: set[int] = set()
    for field in cpu_list.split(","):
        bounds = field.split("-")
        if len(bounds) not in (1, 2) or not all(v.isdigit() for v in bounds):
            raise ValueError("CPU list must contain IDs or ascending ranges")
        begin, end = int(bounds[0]), int(bounds[-1])
        if end < begin:
            raise ValueError("CPU range must be ascending")
        requested.update(range(begin, end + 1))
    allowed = set(os.sched_getaffinity(0))
    if not requested or not requested.issubset(allowed):
        raise ValueError(f"Requested CPUs are outside the allowed mask: {sorted(allowed)}")
    os.sched_setaffinity(0, requested)
    return sorted(os.sched_getaffinity(0))


def _metrics(llm: Any) -> list[dict[str, Any]]:
    return [dataclasses.asdict(metric) for metric in llm.get_metrics()]


def _metric_delta(
    before: list[dict[str, Any]], after: list[dict[str, Any]], name: str
) -> float:
    def total(rows: list[dict[str, Any]]) -> float:
        return sum(row.get("value", 0) for row in rows if row["name"] == name)

    return total(after) - total(before)


def _run_request(
    llm: Any,
    prompt_ids: list[int],
    output_len: int,
    request_id: str,
    temperature: float,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    sampling = SamplingParams(
        temperature=temperature,
        top_k=50 if temperature else -1,
        top_p=0.9 if temperature else 1.0,
        seed=42,
        max_tokens=output_len,
        ignore_eos=True,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    before = _metrics(llm)
    engine = llm.llm_engine
    started = time.perf_counter()
    engine.add_request(request_id, {"prompt_token_ids": prompt_ids}, sampling)

    first = None
    last = None
    token_ids: list[int] = []
    finish_reason = None
    engine_steps = 0
    while engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        engine_steps += 1
        for output in outputs:
            if output.request_id != request_id or not output.outputs:
                continue
            completion = output.outputs[0]
            current = list(completion.token_ids)
            if len(current) > len(token_ids):
                first = now if first is None else first
                last = now
            token_ids = current
            finish_reason = completion.finish_reason

    after = _metrics(llm)
    if len(token_ids) != output_len or first is None or last is None:
        raise RuntimeError(
            f"Incomplete request: {len(token_ids)}/{output_len}, reason={finish_reason}"
        )

    verifications = _metric_delta(
        before, after, "vllm:spec_decode_num_drafts"
    )
    drafted = _metric_delta(
        before, after, "vllm:spec_decode_num_draft_tokens"
    )
    accepted = _metric_delta(
        before, after, "vllm:spec_decode_num_accepted_tokens"
    )
    return {
        "input_tokens": len(prompt_ids),
        "output_tokens": len(token_ids),
        "ttft_ms": (first - started) * 1000,
        "tpot_ms": (last - first) * 1000 / max(1, len(token_ids) - 1),
        "total_ms": (time.perf_counter() - started) * 1000,
        "acceptance_rate": accepted / drafted if drafted else None,
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
        "verification_cycles": verifications,
        "accepted_draft_per_verification": (
            accepted / verifications if verifications else None
        ),
        "mean_acceptance_length_with_bonus": (
            1 + accepted / verifications if verifications else None
        ),
        "engine_step_calls": engine_steps,
        "finish_reason": finish_reason,
        "token_ids": token_ids,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _git_head(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={path}", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--spec-tokens", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-output-len", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-batched-tokens", type=int, default=1280)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--graph", choices=("NONE", "FULL_DECODE_ONLY"), default="FULL_DECODE_ONLY"
    )
    parser.add_argument("--process-cpus", default="")
    parser.add_argument("--prompt-json", type=Path)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 1 or args.input_len < 1 or args.output_len < 2:
        parser.error("repeats/input-len must be positive and output-len must be at least 2")
    if args.spec_tokens < 1:
        parser.error("spec-tokens must be positive for DFlash acceptance measurement")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input-len + output-len exceeds max-model-len")
    args.result_dir.mkdir(parents=True, exist_ok=False)
    effective_affinity = _set_cpu_affinity(args.process_cpus)

    # Import accelerator/model packages only after setting process affinity so
    # worker threads inherit the requested mask.
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.prompt_json:
        prompt_ids = json.loads(args.prompt_json.read_text(encoding="utf-8"))["token_ids"]
    else:
        paragraph = (
            "In machine learning, gradient descent minimizes a loss function by moving "
            "parameters in the direction of steepest descent. Momentum and adaptive "
            "step sizes affect convergence, while regularization and validation help "
            "us distinguish learning from memorization. "
        )
        tail = tokenizer.encode(
            "\nContinue the discussion in detail, with worked examples and practical advice:",
            add_special_tokens=False,
        )
        prompt_ids = tokenizer.encode(
            paragraph * args.input_len, add_special_tokens=False
        )[: args.input_len - len(tail)] + tail
    if len(prompt_ids) != args.input_len:
        raise ValueError(f"Prompt has {len(prompt_ids)} tokens, expected {args.input_len}")

    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_batched_tokens,
        "enable_chunked_prefill": True,
        "gpu_memory_utilization": 0.75,
        "enforce_eager": args.graph == "NONE",
        "enable_prefix_caching": False,
        "async_scheduling": True,
        "seed": 42,
        "disable_log_stats": False,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "load_format": "safetensors",
        "safetensors_load_strategy": "eager",
        "additional_config": {
            "enable_cpu_binding": False,
            "ascend_compilation_config": {
                "enable_npugraph_ex": False,
                "fuse_norm_quant": False,
                "fuse_qknorm_rope": True,
                "enable_static_kernel": False,
                "fuse_muls_add": True,
            },
        },
        "compilation_config": {"cudagraph_mode": args.graph},
        "speculative_config": {
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.spec_tokens,
            "draft_tensor_parallel_size": 1,
        },
    }
    if args.graph != "NONE":
        engine_kwargs["compilation_config"]["cudagraph_capture_sizes"] = [
            args.spec_tokens + 1
        ]

    tracked_env = sorted(
        key for key in os.environ if key.startswith(("VLLM_", "ASCEND_RT_", "OMP_"))
    )
    manifest = {
        "args": vars(args),
        "engine_kwargs": engine_kwargs,
        "cpu_affinity": effective_affinity,
        "environment": {key: os.environ[key] for key in tracked_env},
        "source_heads": {
            "/workspace/vllm-ascend": _git_head("/workspace/vllm-ascend"),
            "/workspace/vllm-024": _git_head("/workspace/vllm-024"),
        },
    }
    _write_json(args.result_dir / "manifest.json", manifest)
    _write_json(args.result_dir / "prompt.json", {"token_ids": prompt_ids})

    llm = LLM(**engine_kwargs)
    results: list[dict[str, Any]] = []
    try:
        if args.warmup_output_len:
            warmup = _run_request(
                llm, prompt_ids, args.warmup_output_len, "warmup", args.temperature
            )
            _write_json(args.result_dir / "warmup.json", warmup)
        for repeat in range(args.repeats):
            result = _run_request(
                llm,
                prompt_ids,
                args.output_len,
                f"measure-{repeat}",
                args.temperature,
            )
            results.append(result)
            _write_json(args.result_dir / f"repeat-{repeat}.json", result)
            print(
                "E2E_RESULT "
                + json.dumps(
                    {
                        "repeat": repeat,
                        "ttft_ms": result["ttft_ms"],
                        "tpot_ms": result["tpot_ms"],
                        "acceptance_rate": result["acceptance_rate"],
                        "accepted_draft_tokens": result["accepted_draft_tokens"],
                        "draft_tokens": result["draft_tokens"],
                        "verification_cycles": result["verification_cycles"],
                    }
                ),
                flush=True,
            )
    finally:
        llm.llm_engine.engine_core.shutdown()

    accepted_total = sum(row["accepted_draft_tokens"] for row in results)
    drafted_total = sum(row["draft_tokens"] for row in results)
    summary = {
        "ttft_ms": _mean_std([row["ttft_ms"] for row in results]),
        "tpot_ms": _mean_std([row["tpot_ms"] for row in results]),
        "acceptance_rate": accepted_total / drafted_total if drafted_total else None,
        "accepted_draft_tokens": accepted_total,
        "draft_tokens": drafted_total,
        "repeats": len(results),
    }
    _write_json(args.result_dir / "summary.json", summary)
    print("E2E_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
