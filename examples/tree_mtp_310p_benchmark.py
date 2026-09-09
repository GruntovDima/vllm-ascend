# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-one eager text probe; reports wall/token separately from decode TPOT."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--workloads", nargs="+", choices=("sky", "code", "russian", "long_prefix", "eos", "one_token"))
    parser.add_argument("--max-case-tokens", type=int)
    parser.add_argument("--spec-metrics", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-tree-trace", action="store_true")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--safetensors-load-strategy", choices=("lazy", "eager"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0, help="Request seed; -1 uses unseeded sampling")
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.device):
        raise RuntimeError("Probe device and ASCEND_RT_VISIBLE_DEVICES must match.")
    if args.tree and args.mtp != args.tree_width * args.tree_depth:
        raise ValueError("The comb-tree probe requires mtp=tree_width*tree_depth")
    if not 0 < args.gpu_memory_utilization < 1 or args.repeats < 1:
        raise ValueError("Invalid memory utilization or repeat count")
    if args.max_case_tokens is not None and args.max_case_tokens < 1:
        raise ValueError("max-case-tokens must be positive")
    if args.input_tokens is not None:
        if args.input_tokens < 1 or args.input_tokens + args.max_tokens + args.mtp + 1 > args.max_model_len:
            raise ValueError("Exact input/output and speculative scratch must fit max-model-len")
        if args.workloads or args.max_case_tokens is not None:
            raise ValueError("Exact-input mode cannot be combined with workload filtering or output capping")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"mode": "tree_mtp" if args.tree else "linear_mtp" if args.mtp else "autoregressive",
              "mtp_tokens": args.mtp, "physical_npu": args.device,
              "pid": os.getpid(), "status": "started", "runs": []}
    report["requested_input_tokens"] = args.input_tokens
    report["sampling_params"] = {
        "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
        "seed": None if args.seed == -1 else args.seed,
    }
    report["process_env"] = {
        key: os.environ.get(key)
        for key in ("ASCEND_RT_VISIBLE_DEVICES", "VLLM_WORKER_MULTIPROC_METHOD")
    }
    def record():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    record()
    started = time.perf_counter()
    try:
        import torch
        import torch_npu  # noqa: F401
        from vllm import LLM, SamplingParams

        kwargs = dict(model=args.model, dtype="float16", quantization="ascend",
                      mamba_ssm_cache_dtype="float16", tensor_parallel_size=1,
                      distributed_executor_backend="uni", enforce_eager=True,
                      enable_prefix_caching=False, async_scheduling=False,
                      max_num_seqs=1, max_model_len=args.max_model_len,
                      max_num_batched_tokens=2048, gpu_memory_utilization=args.gpu_memory_utilization,
                      language_model_only=True, skip_mm_profiling=True,
                      seed=0, disable_log_stats=False,
                      additional_config={"ascend_compilation_config": {
                          "enable_npugraph_ex": False}})
        if args.safetensors_load_strategy is not None:
            kwargs["safetensors_load_strategy"] = args.safetensors_load_strategy
        if args.mtp:
            kwargs["speculative_config"] = {
                "method": "qwen3_5_mtp", "num_speculative_tokens": args.mtp}
        if args.tree:
            kwargs["mamba_cache_mode"] = "none"
            kwargs["additional_config"]["tree_mtp"] = {
                "enabled": True, "width": args.tree_width, "depth": args.tree_depth,
                "trace": not args.no_tree_trace}
        report["engine_args"] = kwargs
        record()
        llm = LLM(**kwargs)

        def spec_counters():
            counters = {}
            if args.spec_metrics:
                for metric in llm.get_metrics():
                    if metric.name.startswith("vllm:spec_decode_") and hasattr(metric, "value"):
                        counters[metric.name] = counters.get(metric.name, 0) + metric.value
            return counters

        report["initialization_s"] = time.perf_counter() - started
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain in plain English why the sky looks blue."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        cases = [("warmup", 8, prompt, True)]
        if args.input_tokens is not None:
            marker = "TREE_MTP_EXACT_INPUT_MARKER"
            template = tokenizer.apply_chat_template(
                [{"role": "user", "content": marker}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            if template.count(marker) != 1:
                raise ValueError("Chat template did not preserve the exact-input marker")
            prefix, suffix = template.split(marker)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(
                "\nUsing the notes above, write a detailed guide to designing reliable automated software tests."
                " Include explanations and practical Python examples.\n" + suffix,
                add_special_tokens=False)
            filler_ids = tokenizer.encode(
                "Technical note: A test should be repeatable, isolated and easy to understand. "
                "Check normal cases, boundary cases and invalid input.\n", add_special_tokens=False)
            fill_count = args.input_tokens - len(prefix_ids) - len(suffix_ids)
            if fill_count < 0 or not filler_ids:
                raise ValueError("Exact input is too short for the fixed template and question")
            repeated = (filler_ids * ((fill_count + len(filler_ids) - 1) // len(filler_ids)))[:fill_count]
            exact_ids = prefix_ids + repeated + suffix_ids
            assert len(exact_ids) == args.input_tokens
            for repeat in range(args.repeats):
                cases.append((f"exact_input_{repeat + 1}", args.max_tokens,
                              {"prompt_token_ids": exact_ids}, True))
        elif args.suite:
            def chat(content):
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            workloads = [
                ("sky", args.max_tokens, prompt, True),
                ("code", 48, chat("Write a short Python function that sums the even numbers in a list."), True),
                ("russian", 48, chat("Объясни простыми словами, зачем программисту нужны автоматические тесты."), True),
                ("long_prefix", 32, chat("Background notes:\n" + "A test should be repeatable and isolated.\n" * 80
                                         + "Summarize the useful advice in one paragraph."), True),
                ("eos", 16, chat("What is 2 + 2? Reply with only the number."), False),
                ("one_token", 1, chat("Name a primary color."), True),
            ]
            for name, limit, case_prompt, ignore_eos in workloads:
                if args.workloads and name not in args.workloads:
                    continue
                if args.max_case_tokens is not None:
                    limit = min(limit, args.max_case_tokens)
                for repeat in range(args.repeats):
                    cases.append((f"{name}_{repeat + 1}", limit, case_prompt, ignore_eos))
        else:
            cases.append(("measured", args.max_tokens, prompt, True))
        for label, max_tokens, case_prompt, ignore_eos in cases:
            params = SamplingParams(**report["sampling_params"], max_tokens=max_tokens,
                                    min_tokens=0, ignore_eos=ignore_eos)
            counters_before = spec_counters()
            torch.npu.synchronize()
            begin = time.perf_counter()
            outputs = llm.generate([case_prompt], params, use_tqdm=False)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - begin
            counters_after = spec_counters()
            output = outputs[0]
            completion = output.outputs[0]
            metrics = output.metrics
            metrics_dict = dataclasses.asdict(metrics) if dataclasses.is_dataclass(metrics) else None
            row = {"label": label, "output_token_ids": completion.token_ids,
                   "ignore_eos": ignore_eos, "max_tokens": max_tokens,
                   "output_text": completion.text,
                   "prompt_token_count": len(output.prompt_token_ids),
                   "prompt_token_sha256": hashlib.sha256(
                       json.dumps(output.prompt_token_ids).encode("utf-8")).hexdigest(),
                   "output_token_count": len(completion.token_ids),
                   "generate_wall_s": elapsed,
                   "wall_ms_per_output_token_including_prefill": elapsed * 1000 / len(completion.token_ids),
                   "request_metrics": metrics_dict,
                   "decode_tpot_ms": None}
            if label.startswith("exact_input_"):
                if row["prompt_token_count"] != args.input_tokens or row["output_token_count"] != args.max_tokens:
                    raise RuntimeError("Exact input/output token-count contract was not met")
                row["prompt_token_ids"] = list(output.prompt_token_ids)
            if args.spec_metrics:
                row["spec_counter_delta"] = {
                    name: value - counters_before.get(name, 0) for name, value in counters_after.items()
                }
                drafts = row["spec_counter_delta"].get("vllm:spec_decode_num_drafts", 0)
                accepted = row["spec_counter_delta"].get("vllm:spec_decode_num_accepted_tokens", 0)
                row["mean_tokens_per_verification"] = 1 + accepted / drafts if drafts else None
            if metrics is not None and len(completion.token_ids) > 1:
                first = getattr(metrics, "first_token_ts", None)
                last = getattr(metrics, "last_token_ts", None)
                if first is None or last is None:
                    first = getattr(metrics, "first_token_time", None)
                    last = getattr(metrics, "last_token_time", None)
                if first is not None and last is not None:
                    row["decode_tpot_ms"] = (last - first) * 1000 / (len(completion.token_ids) - 1)
            report["runs"].append(row)
            record()
            print("BASELINE_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
        report["status"] = "pass"
        record()
        return 0
    except Exception:
        report["status"] = "fail"
        report["error"] = traceback.format_exc()
        report["elapsed_s"] = time.perf_counter() - started
        record()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
