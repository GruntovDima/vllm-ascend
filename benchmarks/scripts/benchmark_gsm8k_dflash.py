#!/usr/bin/env python3
"""Run GSM8K and report accuracy, TTFT, TPOT and optional DFlash acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
from typing import Any


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("question"), str) or not isinstance(
                row.get("answer"), str
            ):
                raise ValueError(f"Invalid GSM8K row at {path}:{line_number}")
            rows.append({"question": row["question"], "answer": row["answer"]})
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").strip().rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _gold_answer(answer: str) -> str | None:
    marker = answer.rsplit("####", maxsplit=1)
    candidates = NUMBER_RE.findall(marker[-1])
    return _canonical_number(candidates[-1] if candidates else None)


def _predicted_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{\s*([^{}]+?)\s*\}", text)
    if boxed:
        candidates = NUMBER_RE.findall(boxed[-1])
        if candidates:
            return _canonical_number(candidates[-1])
    marker = re.findall(r"####\s*([^\n]+)", text)
    if marker:
        candidates = NUMBER_RE.findall(marker[-1])
        if candidates:
            return _canonical_number(candidates[-1])
    candidates = NUMBER_RE.findall(text)
    return _canonical_number(candidates[-1] if candidates else None)


def _strict_predicted_answer(text: str) -> str | None:
    marker = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return _canonical_number(marker.group(1) if marker else None)


def _build_prompt(row: dict[str, str], fewshot_rows: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for example in fewshot_rows:
        parts.append(f"Question: {example['question']}\nAnswer: {example['answer']}")
    parts.append(f"Question: {row['question']}\nAnswer:")
    return "\n\n".join(parts)


def _set_cpu_affinity(cpu_list: str) -> list[int] | None:
    if not cpu_list:
        return None
    requested: set[int] = set()
    for field in cpu_list.split(","):
        bounds = field.split("-")
        if len(bounds) not in (1, 2) or not all(item.isdigit() for item in bounds):
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
    import dataclasses

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
    max_tokens: int,
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
        max_tokens=max_tokens,
        ignore_eos=False,
        stop=["Question:", "</s>", "<|im_end|>"],
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
    if not token_ids or first is None or last is None:
        raise RuntimeError(f"Empty request result, reason={finish_reason}")
    drafted = _metric_delta(before, after, "vllm:spec_decode_num_draft_tokens")
    accepted = _metric_delta(before, after, "vllm:spec_decode_num_accepted_tokens")
    verifications = _metric_delta(before, after, "vllm:spec_decode_num_drafts")
    return {
        "input_tokens": len(prompt_ids),
        "output_tokens": len(token_ids),
        "ttft_ms": (first - started) * 1000,
        "decode_elapsed_ms": (last - first) * 1000,
        "tpot_ms": (last - first) * 1000 / max(1, len(token_ids) - 1),
        "total_ms": (time.perf_counter() - started) * 1000,
        "acceptance_rate": accepted / drafted if drafted else None,
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
        "verification_cycles": verifications,
        "engine_step_calls": engine_steps,
        "finish_reason": finish_reason,
        "token_ids": token_ids,
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


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", default="")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all cases")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--fewshot", type=int, default=5)
    parser.add_argument("--fewshot-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup-output-len", type=int, default=1)
    parser.add_argument("--spec-tokens", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-batched-tokens", type=int, default=1280)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--graph", choices=("NONE", "FULL_DECODE_ONLY"), default="FULL_DECODE_ONLY"
    )
    parser.add_argument("--process-cpus", default="")
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 0 or args.fewshot < 0:
        parser.error("offset, limit and fewshot must be non-negative")
    if args.max_tokens < 2 or args.warmup_output_len < 0:
        parser.error("max-tokens must be >=2 and warmup-output-len non-negative")
    if args.draft and args.spec_tokens < 1:
        parser.error("spec-tokens must be positive when a draft model is configured")
    if not args.draft:
        args.spec_tokens = 0

    train_rows = _load_jsonl(args.train_jsonl)
    all_test_rows = _load_jsonl(args.test_jsonl)
    stop = None if args.limit == 0 else args.offset + args.limit
    test_rows = all_test_rows[args.offset : stop]
    if not test_rows:
        parser.error("selected test range is empty")
    if args.fewshot > len(train_rows):
        parser.error("fewshot exceeds train split size")
    rng = random.Random(args.fewshot_seed)
    fewshot_rows = rng.sample(train_rows, args.fewshot)

    args.result_dir.mkdir(parents=True, exist_ok=False)
    effective_affinity = _set_cpu_affinity(args.process_cpus)
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
    }
    if args.draft:
        engine_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.spec_tokens,
            "draft_tensor_parallel_size": 1,
        }
    if args.graph != "NONE":
        engine_kwargs["compilation_config"]["cudagraph_capture_sizes"] = [
            args.spec_tokens + 1 if args.draft else 1
        ]

    tracked_env = sorted(
        key for key in os.environ if key.startswith(("VLLM_", "ASCEND_RT_", "OMP_"))
    )
    ascend_root = str(Path(__file__).resolve().parents[2])
    vllm_checkout = os.environ.get("VLLM_CHECKOUT")
    manifest = {
        "args": vars(args),
        "dataset": {
            "train_sha256": _sha256(args.train_jsonl),
            "test_sha256": _sha256(args.test_jsonl),
            "train_rows": len(train_rows),
            "test_rows": len(all_test_rows),
            "selected_rows": len(test_rows),
            "fewshot_indices": [train_rows.index(row) for row in fewshot_rows],
        },
        "engine_kwargs": engine_kwargs,
        "cpu_affinity": effective_affinity,
        "environment": {key: os.environ[key] for key in tracked_env},
        "source_heads": {
            ascend_root: _git_head(ascend_root),
            **({vllm_checkout: _git_head(vllm_checkout)} if vllm_checkout else {}),
        },
    }
    _write_json(args.result_dir / "manifest.json", manifest)

    llm = LLM(**engine_kwargs)
    results: list[dict[str, Any]] = []
    try:
        if args.warmup_output_len:
            warmup_prompt = _build_prompt(test_rows[0], fewshot_rows)
            warmup_ids = tokenizer.encode(warmup_prompt, add_special_tokens=True)
            warmup = _run_request(
                llm,
                warmup_ids,
                args.warmup_output_len,
                "gsm8k-warmup",
                args.temperature,
            )
            warmup.pop("token_ids")
            _write_json(args.result_dir / "warmup.json", warmup)
        for relative_index, row in enumerate(test_rows):
            case_index = args.offset + relative_index
            prompt = _build_prompt(row, fewshot_rows)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(prompt_ids) + args.max_tokens > args.max_model_len:
                raise ValueError(
                    f"case {case_index}: {len(prompt_ids)} prompt tokens + "
                    f"{args.max_tokens} max output exceeds max-model-len"
                )
            result = _run_request(
                llm,
                prompt_ids,
                args.max_tokens,
                f"gsm8k-{case_index}",
                args.temperature,
            )
            text = tokenizer.decode(result.pop("token_ids"), skip_special_tokens=True)
            predicted = _predicted_answer(text)
            strict_predicted = _strict_predicted_answer(text)
            expected = _gold_answer(row["answer"])
            result.update(
                {
                    "case_index": case_index,
                    "question": row["question"],
                    "expected_answer": expected,
                    "predicted_answer": predicted,
                    "strict_predicted_answer": strict_predicted,
                    "strict_match": strict_predicted == expected,
                    "flexible_extract": predicted == expected,
                    "output_text": text,
                }
            )
            results.append(result)
            with (args.result_dir / "cases.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                "GSM8K_RESULT "
                + json.dumps(
                    {
                        key: result[key]
                        for key in (
                            "case_index",
                            "strict_match",
                            "flexible_extract",
                            "input_tokens",
                            "output_tokens",
                            "ttft_ms",
                            "tpot_ms",
                            "acceptance_rate",
                        )
                    }
                ),
                flush=True,
            )
    finally:
        llm.llm_engine.engine_core.shutdown()

    accepted = sum(row["accepted_draft_tokens"] for row in results)
    drafted = sum(row["draft_tokens"] for row in results)
    decode_tokens = sum(max(0, row["output_tokens"] - 1) for row in results)
    decode_elapsed = sum(row["decode_elapsed_ms"] for row in results)
    summary = {
        "cases": len(results),
        "strict_match": sum(row["strict_match"] for row in results) / len(results),
        "flexible_extract": (
            sum(row["flexible_extract"] for row in results) / len(results)
        ),
        "input_tokens": _stats([float(row["input_tokens"]) for row in results]),
        "output_tokens": _stats([float(row["output_tokens"]) for row in results]),
        "ttft_ms": _stats([row["ttft_ms"] for row in results]),
        "tpot_ms_per_request": _stats([row["tpot_ms"] for row in results]),
        "tpot_ms_weighted": decode_elapsed / max(1, decode_tokens),
        "acceptance_rate": accepted / drafted if drafted else None,
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
    }
    _write_json(args.result_dir / "summary.json", summary)
    print("GSM8K_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
