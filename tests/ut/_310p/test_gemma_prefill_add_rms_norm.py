"""CPU route checks without an accelerator runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


class Tensor:
    def __init__(self, shape, dtype='half'):
        self.shape = shape
        self.dtype = dtype

    def dim(self):
        return len(self.shape)


class GemmaPrefillFusionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / 'vllm_ascend/_310p/ops/layernorm.py'
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'use_gemma_prefill_add_rms_norm']
        self.env = NS(VLLM_ASCEND_GEMMA_PREFILL_ADD_RMS_NORM=True)
        self.context = NS(cudagraph_runtime_mode='NONE')
        self.available = True
        scope = dict(envs=self.env, torch=NS(float16='half'), CUDAGraphMode=NS(NONE='NONE'),
                     _GEMMA_ADD_RMS_MIN_PREFILL_TOKENS=128, _GEMMA_ADD_RMS_HIDDEN_SIZE=4096,
                     is_forward_context_available=lambda: self.available,
                     get_forward_context=lambda: self.context)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        self.guard = scope['use_gemma_prefill_add_rms_norm']
        self.x = Tensor((1266, 4096))
        self.r = Tensor((1266, 4096))
        self.w = Tensor((4096,))

    def test_prefill_only(self):
        self.assertTrue(self.guard(self.x, self.r, self.w))
        self.env.VLLM_ASCEND_GEMMA_PREFILL_ADD_RMS_NORM = False
        self.assertFalse(self.guard(self.x, self.r, self.w))

    def test_graph_and_missing_context_stay_split(self):
        for mode in ('FULL', 'PIECEWISE'):
            self.context.cudagraph_runtime_mode = mode
            self.assertFalse(self.guard(self.x, self.r, self.w))
        self.context.cudagraph_runtime_mode = 'NONE'
        self.available = False
        self.assertFalse(self.guard(self.x, self.r, self.w))

    def test_decode_and_qk_shapes_stay_split(self):
        for shape in ((1, 4096), (16, 4096), (127, 4096), (1266, 16, 256), (1266, 2048)):
            self.assertFalse(self.guard(Tensor(shape), Tensor(shape), self.w))
        self.assertTrue(self.guard(Tensor((128, 4096)), Tensor((128, 4096)), self.w))

    def test_residual_and_weight_contract(self):
        self.assertFalse(self.guard(self.x, None, self.w))
        self.assertFalse(self.guard(self.x, Tensor((782, 4096)), self.w))
        self.assertFalse(self.guard(self.x, self.r, Tensor((2048,))))
        self.assertFalse(self.guard(self.x, Tensor(self.r.shape, 'float'), self.w))
        self.assertFalse(self.guard(self.x, self.r, Tensor(self.w.shape, 'float')))

    def test_forward_preserves_gamma_offset_and_return_order(self):
        path = Path(__file__).resolve().parents[3] / 'vllm_ascend/_310p/ops/layernorm.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'AscendGemmaRMSNorm310')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_oot')
        future = ast.parse('from __future__ import annotations').body[0]
        fused = Mock(return_value=('normalized', 'rstd', 'residual_out'))
        scope = dict(use_gemma_prefill_add_rms_norm=lambda *args: True,
                     torch_npu=NS(npu_add_rms_norm=fused))
        exec(compile(ast.Module(body=[future, method], type_ignores=[]), str(path), 'exec'), scope)
        result = scope['forward_oot'](NS(weight=0.25, variance_epsilon=1e-6), 'x', 'residual')
        self.assertEqual(result, ('normalized', 'residual_out'))
        fused.assert_called_once_with('x', 'residual', 1.25, 1e-6)


if __name__ == '__main__':
    unittest.main()
