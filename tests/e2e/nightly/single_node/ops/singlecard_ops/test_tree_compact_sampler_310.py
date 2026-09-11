# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Validate the legacy compact filter and the active 310P sampling route."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("Select an authorized physical NPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(status="started", visible_devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                  legacy_checks=[], runtime_checks=[], timings=[])

    def record():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    record()
    try:
        import torch
        import torch_npu
        import vllm_ascend.sample.sampler as base
        import vllm_ascend._310p.sample.sampler as module

        torch_npu.npu.set_device(args.device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        device = torch.device(f"npu:{args.device}")
        rng = torch.Generator().manual_seed(310)
        config = SimpleNamespace(enable_reduce_sample=False)
        runtime_filter = base.apply_top_k_top_p
        legacy_filter = base._apply_top_k_top_p_pytorch
        report["filter_route"] = {
            "runtime_filter": runtime_filter.__name__,
            "legacy_compact_compatible": module._compact_filter_compatible(runtime_filter),
            "compact_status": "disabled_for_fused_filter",
            "reason": "fused TopKTopP applies top-p after top-k renormalization",
        }
        assert runtime_filter is base._apply_top_k_top_p_custom
        assert not module._compact_filter_compatible(runtime_filter)

        def check(name, cpu, top_k, top_p, expected_fast=None):
            logits = cpu.to(device)
            k = torch.full((len(cpu),), top_k, dtype=torch.int32, device=device)
            p = None if top_p is None else torch.full((len(cpu),), top_p, device=device)
            compact = module._try_compact_top_k_310p(logits, k, p, top_k)
            if expected_fast is not None:
                assert (compact is not None) == expected_fast, name
            row = dict(name=name, top_k=top_k, top_p=top_p, fast=compact is not None)
            if compact is not None:
                values, ids, fallback_rows = compact
                safe_rows = [row for row in range(len(cpu)) if row not in fallback_rows]
                row["fallback_rows"] = fallback_rows
                expected = legacy_filter(logits.clone(), k, p)
                expected_cpu = expected.cpu()
                actual = torch.full_like(cpu, -float("inf")).scatter(-1, ids.cpu(), values.cpu())
                assert torch.equal(actual[safe_rows], expected_cpu[safe_rows]), (name, top_k, top_p, "filter mismatch")
                actual_probs = torch.zeros_like(cpu).scatter(-1, ids.cpu(), values.softmax(-1).cpu())
                expected_probs = expected.softmax(-1).cpu()
                torch.testing.assert_close(actual_probs[safe_rows], expected_probs[safe_rows], atol=3e-7, rtol=5e-6)
                row["max_probability_error"] = (actual_probs[safe_rows] - expected_probs[safe_rows]).abs().max().item()
                # Common uniforms away from exact CDF boundaries: compare IDs,
                # not just allowed candidates. No extra RNG consumed by helper.
                for uniform in (0.001, 0.1, 0.5, 0.9, 0.999):
                    u = torch.full((len(cpu),), uniform, device=device)
                    pos = module._sample_from_cdf_310p(values.softmax(-1), u)
                    sampled = ids.gather(-1, pos[:, None]).squeeze(-1)
                    golden = module._sample_from_cdf_310p(expected.softmax(-1), u)
                    assert torch.equal(sampled.cpu()[safe_rows], golden.cpu()[safe_rows]), (name, uniform)
            report["legacy_checks"].append(row)
            record()

        def check_runtime_tree_hint(name, cpu, top_k, top_p):
            logits = cpu.to(device)
            rows = len(cpu)
            k = torch.full((rows,), top_k, dtype=torch.int32, device=device)
            p = None if top_p is None else torch.full((rows,), top_p, device=device)
            uniforms = torch.linspace(0.05, 0.95, rows, device=device)

            def sample_with_common_uniforms(probs, generators):
                return module._sample_from_cdf_310p(probs, uniforms)

            sampler = module.AscendTopKTopPSampler310()
            with (
                patch.object(module, "_random_sample_310p", side_effect=sample_with_common_uniforms),
                patch.object(
                    module,
                    "_try_compact_top_k_310p",
                    side_effect=AssertionError("fused route must not enter the legacy compact helper"),
                ) as compact,
            ):
                sampler.tree_top_k = None
                no_hint, _ = sampler.forward_native(logits.clone(), {}, k, p)
                sampler.tree_top_k = top_k
                with_hint, _ = sampler.forward_native(logits.clone(), {}, k, p)
            torch.npu.synchronize()
            assert torch.equal(with_hint.cpu(), no_hint.cpu()), name
            compact.assert_not_called()
            report["runtime_checks"].append({
                "name": name,
                "top_k": top_k,
                "top_p": top_p,
                "common_uniforms": True,
                "tree_hint_matches_no_hint": True,
                "compact_used": False,
            })
            record()

        def bench(name, fn):
            for _ in range(3):
                fn()
            torch.npu.synchronize()
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(15):
                    fn()
                torch.npu.synchronize()
                samples.append((time.perf_counter() - start) * 1000 / 15)
            report["timings"].append(dict(name=name, wall_ms=samples, median_ms=statistics.median(samples)))
            record()

        with (torch.inference_mode(), patch.object(base, "get_ascend_config", return_value=config),
              patch.object(module, "get_ascend_config", return_value=config)):
            for vocab in (997, 248320):
                for scale in (0.1, 1.0, 3.0, 10.0):
                    cpu = torch.randn(17, vocab, generator=rng) * scale
                    for top_k in (1, 4, 50, 128):
                        for top_p in (None, 0.1, 0.5, 0.9, 1.0):
                            check(f"random_v{vocab}_s{scale}", cpu, top_k, top_p)
            check("cutoff_ties", torch.tensor([[4., 3., 3., 3., 1., 0.]]), 2, 0.9, True)
            check("overflow_ties", torch.tensor([[4., 3., 3., 3., 3., 3., 1., 0.]]), 2, 0.9, False)
            check("internal_ties", torch.tensor([[4., 3., 3., 3., 1., 0.]]), 4, 0.7, True)
            check("all_equal", torch.zeros(17, 997), 50, 0.9, False)
            mixed = torch.randn(17, 248320, generator=rng).half().float()
            mixed[::4] = 0
            check("mixed_quantized_and_tied_rows", mixed, 50, 0.9, True)
            for p in (0.6 - 1e-6, 0.6, 0.6 + 1e-6):
                check("p_boundary", torch.tensor([[0.6, 0.3, 0.1]]).log(), 2, p, False)
            check_runtime_tree_hint(
                "fused_random_v97_k50_p0.9",
                torch.randn(5, 97, generator=rng),
                50,
                0.9,
            )
            check_runtime_tree_hint(
                "fused_random_v997_s0.1_k4_p0.1",
                torch.randn(5, 997, generator=rng) * 0.1,
                4,
                0.1,
            )
            for nodes in (5, 17, 33, 65):
                logits = torch.randn(nodes, 248320, generator=rng).to(device)
                k = torch.full((nodes,), 50, dtype=torch.int32, device=device)
                p = torch.full((nodes,), 0.9, device=device)
                sampler = module.AscendTopKTopPSampler310()
                for label, hint in (("fused", None), ("fused_tree_hint", 50)):
                    sampler.tree_top_k = hint
                    bench(
                        f"sampler_{label}_n{nodes}",
                        lambda: sampler.forward_native(logits.clone(), {}, k, p),
                    )
                if nodes == 17:
                    logits = mixed.to(device)
                    for label, hint in (("fused", None), ("fused_tree_hint", 50)):
                        sampler.tree_top_k = hint
                        bench(
                            f"sampler_mixed_{label}_n{nodes}",
                            lambda: sampler.forward_native(logits.clone(), {}, k, p),
                        )
        report["status"] = "pass"
        record()
        print(json.dumps(dict(
            status=report["status"],
            legacy_checks=len(report["legacy_checks"]),
            runtime_checks=len(report["runtime_checks"]),
            filter_route=report["filter_route"],
            timings=report["timings"],
        )))
    except Exception as error:
        report.update(status="fail", error=repr(error))
        record()
        raise


if __name__ == "__main__":
    main()
