"""CPU tests for post-load guards; real NPU comparisons are separate artifacts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/ops/static_scalar_quant.py"
TREE = ast.parse(SOURCE.read_text())
NAMESPACE = {"torch": torch, "envs": SimpleNamespace(VLLM_ASCEND_STATIC_SCALAR_QUANT=True)}
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.Assign))],
                        type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
prepare = NAMESPACE["prepare_scalar_quant_params"]
select = NAMESPACE["static_quant_params"]


def layer(width=4096):
    obj = torch.nn.Module()
    obj.aclnn_input_scale = torch.full((width,), 0.03, dtype=torch.float16)
    obj.aclnn_input_scale_reciprocal = 1.0 / obj.aclnn_input_scale
    obj.aclnn_input_offset = torch.full((width,), -3, dtype=torch.float16)
    return obj


class StaticScalarQuantTests(unittest.TestCase):
    def setUp(self):
        NAMESPACE["envs"].VLLM_ASCEND_STATIC_SCALAR_QUANT = True

    def test_prepare_preserves_originals_and_exact_reciprocal(self):
        obj = layer()
        before = obj.aclnn_input_scale_reciprocal.clone()
        prepare(obj)
        self.assertTrue(obj._scalar_quant_ready)
        self.assertTrue(torch.equal(before, obj.aclnn_input_scale_reciprocal))
        self.assertTrue(torch.equal(before[:1], obj._scalar_quant_reciprocal))
        self.assertNotEqual(before.data_ptr(), obj._scalar_quant_reciprocal.data_ptr())
        self.assertEqual(obj.state_dict(), {})

    def test_decode_prefill_and_fallback(self):
        for width in (4096, 8192):
            obj = layer(width)
            prepare(obj)
            for rows in (16, 782, 1266, 2048):
                self.assertEqual([p.numel() for p in select(obj, torch.empty(rows, width, dtype=torch.float16))], [1]*3)
            for x in (torch.empty(2, width), torch.empty(1, 2, width, dtype=torch.float16),
                      torch.empty(2, 16, dtype=torch.float16)):
                self.assertIs(select(obj, x)[0], obj.aclnn_input_scale)

    def test_rejects_nonuniform_nonfinite_and_invalid_scale(self):
        for attr, value in (("aclnn_input_scale", 0.0), ("aclnn_input_scale", float('nan')),
                            ("aclnn_input_scale_reciprocal", float('inf')),
                            ("aclnn_input_offset", -2.0)):
            obj = layer()
            getattr(obj, attr)[-1] = value
            prepare(obj)
            self.assertFalse(obj._scalar_quant_ready)
        obj = layer()
        obj.aclnn_input_scale.fill_(-1)
        prepare(obj)
        self.assertFalse(obj._scalar_quant_ready)

    def test_flag_width_dtype_and_reload_guards(self):
        obj = layer()
        prepare(obj)
        NAMESPACE["envs"].VLLM_ASCEND_STATIC_SCALAR_QUANT = False
        prepare(obj)
        self.assertFalse(obj._scalar_quant_ready)
        NAMESPACE["envs"].VLLM_ASCEND_STATIC_SCALAR_QUANT = True
        for obj in (layer(12288), layer()):
            if obj.aclnn_input_scale.numel() == 4096:
                obj.aclnn_input_offset = obj.aclnn_input_offset.float()
            prepare(obj)
            self.assertFalse(obj._scalar_quant_ready)
        obj = layer()
        prepare(obj)
        obj.aclnn_input_offset.fill_(7)
        prepare(obj)
        self.assertEqual(obj._scalar_quant_offset.item(), 7)
        obj.aclnn_input_offset[-1] = 8
        prepare(obj)
        self.assertFalse(obj._scalar_quant_ready)


if __name__ == "__main__":
    unittest.main()
