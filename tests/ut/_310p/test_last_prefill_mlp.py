"""CPU contract tests; real W8A8 row equivalence needs the NPU audit."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/ops/last_prefill_mlp.py"
TREE = ast.parse(SOURCE.read_text())
NAMESPACE = {"torch": torch}
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.Assign))],
                        type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
eligible = NAMESPACE["last_prefill_mlp_eligible"]
selected = NAMESPACE["selected_mlp_output"]
selected_norm = NAMESPACE["selected_norm_output"]
selected_attention = NAMESPACE["selected_attention_output"]


def context(**kwargs):
    fields = dict(last_prefill_mlp_allowed=True, skip_compiled=True, in_profile_run=False,
                  capturing=False, is_draft_model=False, cudagraph_runtime_mode=SimpleNamespace(name="NONE"))
    fields.update(kwargs)
    return SimpleNamespace(**fields)


class LastPrefillMLPTests(unittest.TestCase):
    def test_selected_attention_gate_projection_and_poisoned_rows(self):
        for rows in (782, 1266):
            for gated in (False, True):
                x = torch.full((rows, 8), float("nan"))
                x[-1] = torch.arange(8, dtype=torch.float32)
                gate = torch.full_like(x, float("inf")) if gated else None
                if gated:
                    gate[-1] = torch.arange(8, dtype=torch.float32) - 4
                calls = []

                def projection(value):
                    calls.append(value.shape)
                    return value[:, :4] * 2 + 1, None

                result = selected_attention(SimpleNamespace(o_proj=projection), x, gate)
                expected = x[-1:] * torch.sigmoid(gate[-1:]) if gated else x[-1:]
                self.assertEqual(calls, [torch.Size((1, 8))])
                self.assertEqual(result.shape, (rows, 4))
                self.assertTrue(torch.equal(result[-1:], expected[:, :4] * 2 + 1))
                self.assertEqual(torch.count_nonzero(result[:-1]), 0)
                self.assertTrue(torch.isnan(x[:-1]).all())
                self.assertTrue(torch.equal(x[-1], torch.arange(8)))
                if gated:
                    self.assertTrue(torch.isinf(gate[:-1]).all())
                    self.assertTrue(torch.equal(gate[-1], torch.arange(8) - 4))

    def test_only_actual_uncompiled_prefill(self):
        for rows in (128, 782, 1266, 2048):
            self.assertTrue(eligible(context(), rows, (2, 6, 10, 14, 18, 22, 26, 30), 32))
        for rows in (0, 1, 16, 32, 127):
            self.assertFalse(eligible(context(), rows, (30,), 32))

    def test_fail_closed_runtime_and_consumers(self):
        self.assertFalse(eligible(SimpleNamespace(), 782, (30,), 32))
        for key, value in (("last_prefill_mlp_allowed", False), ("skip_compiled", False),
                           ("in_profile_run", True), ("capturing", True), ("is_draft_model", True),
                           ("cudagraph_runtime_mode", SimpleNamespace(name="FULL"))):
            self.assertFalse(eligible(context(**{key: value}), 782, (30,), 32), key)
        self.assertFalse(eligible(context(), 782, (), 32))
        self.assertFalse(eligible(context(), 782, (30, 32), 32))

    def test_selected_row_exact_no_input_mutation_no_uninitialized_output(self):
        for rows in (782, 1266):
            x = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
            before = x.clone()
            calls = []

            def mlp(value):
                calls.append(value.shape)
                return value * 2 + 1

            output = selected(mlp, x)
            self.assertEqual(calls, [torch.Size((1, 8))])
            self.assertTrue(torch.equal(output[-1], before[-1] * 2 + 1))
            self.assertEqual(torch.count_nonzero(output[:-1]), 0)
            self.assertTrue(torch.equal(x, before))
            self.assertNotEqual(x.data_ptr(), output.data_ptr())

    def test_selected_norm_residual_and_poisoned_unused_rows(self):
        norm = SimpleNamespace(weight=torch.ones(8), variance_epsilon=1e-6)
        for rows in (782, 1266):
            x = torch.full((rows, 8), float("nan"))
            residual = torch.full_like(x, float("inf"))
            x[-1] = torch.arange(8)
            residual[-1] = torch.arange(8) * 2
            calls = []

            def norm_fn(value, residual, weight, eps):
                calls.append(value.shape)
                summed = value + residual
                return summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps) * weight, summed

            output, summed = selected_norm(norm, norm_fn, x, residual)
            self.assertEqual(calls, [torch.Size((1, 8))])
            expected, expected_sum = norm_fn(x[-1:], residual[-1:], norm.weight, norm.variance_epsilon)
            self.assertTrue(torch.equal(output[-1:], expected))
            self.assertTrue(torch.equal(summed[-1:], expected_sum))
            self.assertEqual(torch.count_nonzero(output[:-1]), 0)
            self.assertEqual(torch.count_nonzero(summed[:-1]), 0)
            self.assertTrue(torch.isnan(x[:-1]).all())
            self.assertTrue(torch.isinf(residual[:-1]).all())
            self.assertTrue(torch.equal(x[-1], torch.arange(8)))
            self.assertTrue(torch.equal(residual[-1], torch.arange(8) * 2))


if __name__ == "__main__":
    unittest.main()
