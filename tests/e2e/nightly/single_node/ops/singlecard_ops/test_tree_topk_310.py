# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exact draft candidate ordering and paired NPU component timing, no model.

Uses an AST loader for the production selector only: vLLM import is not needed.
CPU stable sorting is the golden for widths >1, including boundary ties.
Width one preserves native argmax, whose nonfinite behavior on the pinned NPU
can differ from CPU argmax; that difference is recorded, not hidden.
This is not full-model accuracy or a native kernel port certification.
"""

import argparse
import ast
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def load_selector(torch, path):
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_tree_topk_token_ids")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_tree_topk_token_ids"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 1:
        parser.error("Use a new output file and positive iterations")
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{args.device}")
    source = Path(__file__).resolve().parents[6] / "vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py"
    select = load_selector(torch, source)
    report = {"status": "started", "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "source_sha256_lf": hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "checks": [], "timings": [], "versions": {"torch": torch.__version__}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    record()
    rng = torch.Generator().manual_seed(310)
    cases = {
        "random_small": torch.randn((3, 127), generator=rng),
        "ties_small": torch.randint(-3, 4, (3, 127), generator=rng).float(),
        "all_negative_inf": torch.full((2, 127), -torch.inf),
        "all_positive_inf": torch.full((2, 127), torch.inf),
        "all_zero": torch.zeros((2, 127)),
        "nan_inf": torch.tensor([[1., torch.nan, torch.nan, torch.inf, -torch.inf, 0., -0., 1.],
                                 [torch.inf, 1., torch.inf, -torch.inf, torch.nan, -1., 0., 0.]]),
        "random_full_vocab": torch.randn((1, 248320), generator=rng),
        "ties_full_vocab": torch.randint(-3, 4, (1, 248320), generator=rng).float(),
        "adjacent_fp32": torch.arange(0x3F800000, 0x3F800080, dtype=torch.int32).view(torch.float32).flip(0)[None],
        "nan_payloads": torch.tensor([[0x7F800001, 0x7FC00000, 0x7F800000, 0, -2147483648, -8388608]],
                                     dtype=torch.int32).view(torch.float32),
    }
    with torch.inference_mode():
        for dtype in (torch.float16, torch.float32):
            for name, values in cases.items():
                cpu = values.to(dtype)
                logits = cpu.to(device)
                golden = cpu.argsort(dim=-1, descending=True, stable=True)
                for width in (1, 2, 4, 8, 16):
                    if width > cpu.shape[-1]:
                        continue
                    expected = logits.argmax(dim=-1, keepdim=True).cpu() if width == 1 else golden[:, :width]
                    exact = all(torch.equal(select(logits, width).cpu(), expected) for _ in range(3))
                    unchanged = torch.equal(logits.cpu().view(torch.int16 if dtype == torch.float16 else torch.int32),
                                            cpu.view(torch.int16 if dtype == torch.float16 else torch.int32))
                    report["checks"].append(dict(case=name, dtype=str(dtype), width=width,
                                                 exact=exact, input_unchanged=unchanged,
                                                 golden="native_argmax" if width == 1 else "cpu_stable_sort",
                                                 golden_matches_cpu=bool(torch.equal(expected, golden[:, :width]))))
                    record()
                    if not exact or not unchanged:
                        report["status"] = "fail"
                        record()
                        raise AssertionError(report["checks"][-1])
        candidates = {"optimized": select}
        if args.baseline_source:
            candidates["reference"] = load_selector(torch, args.baseline_source)
        for dtype in (torch.float16, torch.float32):
            logits = cases["random_full_vocab"].to(dtype=dtype, device=device)
            for width in (1, 2, 4, 8, 16):
                for name, fn in candidates.items():
                    for _ in range(3):
                        fn(logits, width)
                    torch.npu.synchronize()
                    times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            fn(logits, width)
                        torch.npu.synchronize()
                        times.append((time.perf_counter() - start) * 1000 / args.iterations)
                    row = dict(algorithm=name, dtype=str(dtype), width=width,
                               wall_ms=times, median_ms=statistics.median(times))
                    report["timings"].append(row)
                    record()
                    print(json.dumps(row), flush=True)
    report["status"] = "pass"
    record()


if __name__ == "__main__":
    main()
