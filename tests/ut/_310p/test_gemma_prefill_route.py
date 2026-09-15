"""Standalone CPU guard tests for the optional Gemma prefill route."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


class GemmaPrefillRouteTests(unittest.TestCase):
    def test_shape_and_enable_guards(self):
        path = Path(__file__).resolve().parents[3] / 'vllm_ascend/_310p/ops/layernorm.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'gemma_rms_norm_310')
        env = SimpleNamespace(VLLM_ASCEND_GEMMA_PREFILL_ADN=False)
        native = Mock(return_value=('native', None))
        adn = Mock(return_value='adn')
        scope = {'torch': torch, 'envs': env, '_GEMMA_ADN_MIN_PREFILL_TOKENS': 128,
                 'torch_npu': SimpleNamespace(npu_rms_norm=native),
                 'adn_rms_norm_or_fallback': adn}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        for enabled in (False, True):
            env.VLLM_ASCEND_GEMMA_PREFILL_ADN = enabled
            for shape in ((16, 4, 256), (127, 4, 256), (128, 4, 256),
                          (782, 16, 256), (782, 4096), (782, 4, 128)):
                x = torch.randn(*shape)
                gamma = torch.ones(shape[-1])
                native.reset_mock()
                adn.reset_mock()
                result = scope['gemma_rms_norm_310'](x, gamma, 1e-6)
                selected = enabled and len(shape) == 3 and shape[0] >= 128 and shape[-1] == 256
                self.assertEqual(result, 'adn' if selected else 'native')
                self.assertEqual(adn.call_count, int(selected))
                self.assertEqual(native.call_count, int(not selected))


if __name__ == '__main__':
    unittest.main()
