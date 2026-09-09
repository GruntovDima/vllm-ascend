# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Standalone NPU target-sampler checks; no model or attention kernel is mocked.

Only the initialized engine configuration is replaced with its supported
enable_reduce_sample=False value. Native filtering, softmax, RNG adapter and
CDF selection are exercised. Full-model behavior needs the separate probe.
"""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True, help="Logical device after visibility selection")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("Select an authorized physical NPU explicitly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"], "cases": []}

    def record():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    record()
    try:
        import torch
        import torch_npu
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env()
        torch_npu.npu.set_device(args.device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        import vllm_ascend.ops  # noqa: F401
        import vllm_ascend.sample.sampler as base_module
        import vllm_ascend._310p.sample.sampler as module

        device = torch.device(f"npu:{args.device}")
        config = SimpleNamespace(enable_reduce_sample=False)

        def metadata(temperature, top_k, top_p, seed=42):
            return SimpleNamespace(
                temperature=torch.tensor([temperature], device=device),
                all_greedy=temperature == 0, all_random=temperature != 0,
                top_k=None if top_k is None else torch.tensor([top_k], dtype=torch.int32, device=device),
                top_p=None if top_p is None else torch.tensor([top_p], device=device),
                generators={} if seed is None else {0: torch.Generator(device=device).manual_seed(seed)},
                max_num_logprobs=None, logprob_token_ids=None, no_penalties=True,
                prompt_token_ids=None, output_token_ids=[[]], spec_token_ids=None,
                allowed_token_ids_mask=None, bad_words_token_ids={},
                logitsprocs=SimpleNamespace(argmax_invariant=[], non_argmax_invariant=[]),
                thinking_budget_state_holder=None,
            )

        with (torch.inference_mode(),
              patch.object(base_module, "get_ascend_config", return_value=config),
              patch.object(module, "get_ascend_config", return_value=config)):
            sampler = module.AscendSampler310()
            rng = torch.Generator().manual_seed(310)
            for nodes in (1, 5, 17, 65):
                logits = torch.randn(nodes, 97, generator=rng).to(device)
                for temperature, top_k, top_p in ((0.0, None, None), (1.0, 50, 0.9),
                                                  (0.7, 1, None), (1.3, None, 0.8), (1.0, None, None)):
                    actual = sampler.sample_tree(logits.clone(), metadata(temperature, top_k, top_p), top_k=top_k)
                    torch_npu.npu.synchronize()
                    serial_metadata = metadata(temperature, top_k, top_p)
                    expected = torch.cat([
                        sampler(logits[node:node + 1].clone(), serial_metadata).sampled_token_ids.flatten()
                        for node in range(nodes)
                    ])
                    torch_npu.npu.synchronize()
                    assert torch.equal(actual.cpu(), expected.cpu()), (nodes, temperature, top_k, top_p)
                    repeated = sampler.sample_tree(logits.clone(), metadata(temperature, top_k, top_p), top_k=top_k)
                    assert torch.equal(actual.cpu(), repeated.cpu()), "seeded sampling is not repeatable"
                    report["cases"].append({"nodes": nodes, "temperature": temperature, "top_k": top_k,
                                            "top_p": top_p, "serial_match": True, "seed_repeatable": True})
                    record()

            # Empirical target mass, including zero-weight candidates. This
            # complements the exact finite-tree enumeration in the CPU suite.
            probabilities = torch.tensor([0.0, 0.1, 0.2, 0.7])
            trials = 8192
            logits = probabilities.log().repeat(trials, 1).to(device)
            actual = sampler.sample_tree(logits, metadata(1.0, None, None, seed=91)).cpu()
            frequencies = torch.bincount(actual.long(), minlength=4).double() / trials
            assert frequencies[0] == 0
            for observed, expected in zip(frequencies.tolist(), probabilities.tolist()):
                tolerance = 6 * math.sqrt(expected * (1 - expected) / trials) + 1 / trials
                assert abs(observed - expected) <= tolerance, (observed, expected, tolerance)
            report["distribution"] = {"trials": trials, "expected": probabilities.tolist(),
                                      "observed": frequencies.tolist(), "passed": True}
            # Exercise compact sampling, not just its dense fallback, with an
            # analytically known truncated target distribution.
            actual = sampler.sample_tree(
                logits.clone(), metadata(1.0, 2, 0.9, seed=91), top_k=2,
            ).cpu()
            frequencies = torch.bincount(actual.long(), minlength=4).double() / trials
            compact_expected = torch.tensor([0., 0., 2 / 9, 7 / 9])
            for observed, expected in zip(frequencies.tolist(), compact_expected.tolist()):
                tolerance = 6 * math.sqrt(expected * (1 - expected) / trials) + 1 / trials
                assert abs(observed - expected) <= tolerance, (observed, expected, tolerance)
            assert frequencies[:2].sum() == 0
            report["compact_distribution"] = dict(trials=trials, expected=compact_expected.tolist(),
                                                  observed=frequencies.tolist(), passed=True)
            # Mixed fast/fallback rows must consume exactly one draw each in
            # original node order; falling back cannot advance the RNG again.
            mixed = torch.randn(17, 997, generator=rng).half().float()
            mixed[::3] = 0
            mixed = mixed.to(device)
            actual = sampler.sample_tree(mixed.clone(), metadata(1.0, 50, 0.9), top_k=50)
            serial_md = metadata(1.0, 50, 0.9)
            expected = torch.cat([sampler(row[None].clone(), serial_md).sampled_token_ids.flatten()
                                  for row in mixed])
            assert torch.equal(actual.cpu(), expected.cpu()), "mixed fallback RNG/order mismatch"
            report["mixed_fallback_seeded_match"] = True
            unseeded = sampler.sample_tree(torch.zeros(65, 4, device=device), metadata(1.0, None, None, seed=None))
            assert unseeded.shape == (65,) and unseeded.cpu().unique().numel() > 1
            report["unseeded"] = "pass"
        report["status"] = "pass"
        record()
        print(json.dumps(report))
    except Exception as error:
        report.update(status="fail", error=repr(error))
        record()
        raise


if __name__ == "__main__":
    main()
