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


def context(**kwargs):
    fields = dict(last_prefill_mlp_allowed=True, skip_compiled=True, in_profile_run=False,
                  capturing=False, is_draft_model=False, cudagraph_runtime_mode=SimpleNamespace(name="NONE"))
    fields.update(kwargs)
    return SimpleNamespace(**fields)


class LastPrefillMLPTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
