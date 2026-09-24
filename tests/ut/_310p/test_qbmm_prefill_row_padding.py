"""CPU routing tests; actual NPU outputs/performance require the micro/model screen."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


class RowPaddingTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/quantization/methods/qbmm_custom.py"
        names = {"_kernel_supports", "_prefill_row_alignment", "_prefill_padded_rows",
                 "_qbmm_v3x", "_qbmm_v3x_fake"}
        tree = ast.parse(source.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual(len(nodes), len(names))
        self.env = SimpleNamespace(
            VLLM_ASCEND_QBMM_PREFILL_ROW_ALIGNMENTS="24576:32,12288:64",
            VLLM_ASCEND_QBMM_K_PIPELINE=False,
        )
        self.scope = {
            "torch": torch,
            "envs": self.env,
            "_MAX_DIM": 32768,
            "_ENABLE_K_PIPELINE": False,
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), self.scope)

    def weight(self, k=4096, n=24576):
        return torch.empty((k, n), dtype=torch.int8, device="meta")

    def test_guards(self):
        route = self.scope["_prefill_padded_rows"]
        x = torch.zeros((782, 4096), dtype=torch.int8)
        self.assertEqual(route(x, self.weight(), None), 800)
        self.assertEqual(route(x, self.weight(n=12288), None), 832)
        for rows in (0, 1, 16, 128, 1280, 2048):
            self.assertEqual(route(torch.empty((rows, 4096), dtype=torch.int8), self.weight(), None), 0)
        for rows, padded in ((781, 800), (783, 800), (1266, 1280)):
            self.assertEqual(route(torch.empty((rows, 4096), dtype=torch.int8), self.weight(), None), padded)
        self.assertEqual(route(x, self.weight(n=64), None), 0)
        self.assertEqual(route(torch.zeros((782, 8192), dtype=torch.int8),
                               self.weight(k=8192), None), 800)
        self.assertEqual(route(x, self.weight(), torch.ones(782)), 0)
        self.assertEqual(route(x.float(), self.weight(), None), 0)
        self.assertEqual(route(x.unsqueeze(0), self.weight(), None), 0)
        strided = torch.empty((782, 8192), dtype=torch.int8)[:, ::2]
        self.assertEqual(route(strided, self.weight(), None), 0)
        self.env.VLLM_ASCEND_QBMM_PREFILL_ROW_ALIGNMENTS = ""
        self.assertEqual(route(x, self.weight(), None), 0)

    def test_policy_validation(self):
        resolve = self.scope["_prefill_row_alignment"]
        for policy in ("bad", "24576:3", "-1:32"):
            self.env.VLLM_ASCEND_QBMM_PREFILL_ROW_ALIGNMENTS = policy
            with self.assertRaises(ValueError):
                resolve(24576)

    def test_routing_zero_tail_and_no_mutation(self):
        x = torch.randint(-128, 128, (782, 4096), dtype=torch.int8)
        before = x.clone()
        scale, bias = torch.ones(24576, dtype=torch.int64), torch.zeros(24576, dtype=torch.int32)
        for policy, expected_rows in (("", 782), ("24576:32", 800)):
            self.env.VLLM_ASCEND_QBMM_PREFILL_ROW_ALIGNMENTS = policy
            def native(value, weight, passed_scale, **kwargs):
                self.assertEqual(tuple(value.shape), (expected_rows, 4096))
                self.assertTrue(torch.equal(value[:782], before))
                if policy:
                    self.assertEqual(torch.count_nonzero(value[782:]), 0)
                self.assertIs(passed_scale, scale)
                self.assertIs(kwargs["bias"], bias)
                self.assertTrue(kwargs["transpose_x2"])
                self.assertEqual(kwargs["enable_k_pipeline"], False)
                return torch.ones((expected_rows, 24576), dtype=torch.float16)
            op = Mock(side_effect=native)
            # Mock only the native namespace, retaining real torch tensors.
            proxy = SimpleNamespace(**{n: getattr(torch, n) for n in ("Tensor", "int8", "float16")})
            proxy.ops = SimpleNamespace(_C_ascend=SimpleNamespace(quant_batch_matmul_v3_x=op))
            self.scope["torch"] = proxy
            out = self.scope["_qbmm_v3x"](x, self.weight(), scale, None, bias)
            self.assertEqual(tuple(out.shape), (782, 24576))
            self.assertTrue(out.is_contiguous())
            self.assertTrue(torch.equal(x, before))
            self.assertEqual(op.call_count, 1)

    def test_unsupported_weight_still_uses_builtin(self):
        builtin = Mock(return_value="fallback")
        self.scope["torch_npu"] = SimpleNamespace(npu_quant_matmul=builtin)
        out = self.scope["_qbmm_v3x"](torch.empty((782, 4096), dtype=torch.int8),
                                       self.weight(n=98304), torch.ones(1), None, None)
        self.assertEqual(out, "fallback")
        self.assertEqual(builtin.call_count, 1)

    def test_fake_keeps_logical_shape(self):
        x = torch.empty((782, 4096), dtype=torch.int8, device="meta")
        out = self.scope["_qbmm_v3x_fake"](x, self.weight(), torch.empty(0), None, None)
        self.assertEqual(tuple(out.shape), (782, 24576))
        self.assertEqual(out.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
