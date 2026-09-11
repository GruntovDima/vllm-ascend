# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU checks of the compact filter against the pinned production filter."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def load_helpers():
    helper = Path(__file__).with_name("test_sampler_310.py")
    spec = importlib.util.spec_from_file_location("_compact_sampler_cpu_helpers", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = helper.resolve().parents[4] / "vllm_ascend/sample/sampler.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef)
                and n.name == "_apply_top_k_top_p_pytorch")
    ns = dict(torch=torch, get_ascend_config=lambda: SimpleNamespace(enable_reduce_sample=False))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    return module.sampler_310p, ns["_apply_top_k_top_p_pytorch"]


class TestCompactTreeSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, golden = load_helpers()
        cls.golden = staticmethod(golden)

    def check(self, logits, top_k, top_p, expect_fast=None):
        k = torch.full((len(logits),), top_k, dtype=torch.int32)
        p = None if top_p is None else torch.full((len(logits),), top_p)
        original = logits.clone()
        compact = self.module._try_compact_top_k_310p(logits, k, p, top_k)
        self.assertTrue(torch.equal(original, logits), "helper mutated input logits")
        if expect_fast is not None:
            self.assertEqual(compact is not None, expect_fast)
        if compact is None:
            return False
        values, ids, fallback_rows = compact
        safe_rows = [row for row in range(len(logits)) if row not in fallback_rows]
        expected = self.golden(logits.clone(), k, p)
        actual = torch.full_like(expected, -float("inf")).scatter(-1, ids, values)
        self.assertTrue(torch.equal(actual[safe_rows], expected[safe_rows]), "retained token set/logits differ")
        probs = torch.zeros_like(logits).scatter(-1, ids, values.softmax(-1))
        torch.testing.assert_close(probs[safe_rows], expected.softmax(-1)[safe_rows], atol=2e-7, rtol=3e-6)
        self.assertTrue((ids[:, 1:] > ids[:, :-1]).all())
        return True

    def test_random_scales_and_top_p(self):
        rng = torch.Generator().manual_seed(310)
        fast = 0
        for scale in (0.1, 1.0, 3.0, 10.0):
            logits = torch.randn(17, 997, generator=rng) * scale
            for top_k in (1, 4, 50, 128):
                for top_p in (None, 0.1, 0.5, 0.9, 1.0):
                    fast += self.check(logits, top_k, top_p)
        self.assertGreater(fast, 40)

    def test_top_p_uses_original_mass_not_renormalized_top_k(self):
        logits = torch.tensor([[0.40, 0.25, 0.20, 0.15]]).log()
        self.check(logits, 2, 0.6, expect_fast=True)
        compact = self.module._try_compact_top_k_310p(logits, torch.tensor([2]), torch.tensor([0.6]), 2)
        self.assertEqual(torch.isfinite(compact[0]).sum().item(), 2)

    def test_route_requires_exact_legacy_filter_identity(self):
        self.assertTrue(self.module._compact_filter_compatible(
            self.module._apply_top_k_top_p_pytorch
        ))
        self.assertFalse(self.module._compact_filter_compatible(lambda *args: args))

    def test_cutoff_ties_fall_back_and_internal_ties_are_retained(self):
        self.check(torch.tensor([[4., 3., 3., 3., 1., 0.]]), 2, 0.9, expect_fast=True)
        self.check(torch.tensor([[4., 3., 3., 3., 3., 3., 1., 0.]]), 2, 0.9, expect_fast=False)
        self.check(torch.tensor([[4., 3., 3., 3., 1., 0.]]), 4, 0.7, expect_fast=True)
        self.check(torch.zeros(5, 997), 50, 0.9, expect_fast=False)

    def test_numerical_top_p_boundary_uses_original_filter(self):
        logits = torch.tensor([[0.6, 0.3, 0.1]]).log()
        for p in (0.6 - 1e-6, 0.6, 0.6 + 1e-6):
            self.check(logits, 2, p, expect_fast=False)

    def test_unsupported_hints_and_dtypes_fall_back(self):
        logits = torch.randn(2, 200)
        for hint in (None, -1, 0, True, 129, 200):
            self.assertIsNone(self.module._try_compact_top_k_310p(logits, torch.tensor([50, 50]), None, hint))
        self.assertIsNone(self.module._try_compact_top_k_310p(logits.half(), torch.tensor([50, 50]), None, 50))
        self.assertIsNone(self.module._try_compact_top_k_310p(logits, None, None, 50))
        compact = self.module._try_compact_top_k_310p(logits, torch.tensor([49, 50]), None, 50)
        self.assertEqual(compact[2], [0])

    def test_only_ambiguous_rows_fall_back(self):
        logits = torch.arange(100).float().repeat(5, 1)
        logits[1] = 0
        logits[3] = 0
        compact = self.module._try_compact_top_k_310p(logits, torch.full((5,), 4), None, 4)
        self.assertEqual(compact[2], [1, 3])
        self.check(logits, 4, None, expect_fast=True)

    def test_nonfinite_probabilities_fall_back(self):
        for value in (float("inf"), float("nan"), -float("inf")):
            logits = torch.full((1, 100), value)
            self.assertIsNone(self.module._try_compact_top_k_310p(logits, torch.tensor([4]), None, 4))


if __name__ == "__main__":
    unittest.main()
