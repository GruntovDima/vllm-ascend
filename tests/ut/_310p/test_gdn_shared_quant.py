"""Sharing guards and exact preservation of upstream attention forward."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


class GDN:
    pass


def named(name, **values):
    return type(name, (SimpleNamespace,), {})(**values)


class SharedQuantTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[3]
        self.path = self.root / 'vllm_ascend/patch/worker/patch_gdn_shared_quant_310p.py'
        functions = [n for n in ast.parse(self.path.read_text()).body if isinstance(n, ast.FunctionDef)]
        self.env = SimpleNamespace(VLLM_ASCEND_GDN_SHARED_INPUT_QUANT=True)
        self.fallback = Mock(return_value='original')
        self.fields = ('aclnn_input_scale', 'aclnn_input_scale_reciprocal', 'aclnn_input_offset')
        self.scope = {'torch': torch, 'QwenGatedDeltaNetAttention': GDN, 'envs': self.env,
                      '_QUANT_FIELDS': self.fields,
                      '_ORIGINAL_FORWARD_CUDA': self.fallback, '_encode_layer_name': lambda x: x}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(self.path), 'exec'), self.scope)
        self.module = GDN()
        self.module.tp_size = 1
        self.module.gqa_interleaved_layout = False
        for attr in ('in_proj_qkvz', 'in_proj_ba'):
            layer = named('AscendMergedColumnParallelLinear', tp_size=1, custom_op=None, bias=None,
                          params_dtype=torch.float16,
                          quant_method=SimpleNamespace(quant_method=SimpleNamespace(
                              accepts_prequantized_input=True)))
            for field in self.fields:
                setattr(layer, field, SimpleNamespace(shape=(4096,), dtype=torch.float16,
                        device=SimpleNamespace(type='npu'), value=1, numel=lambda: 4096))
            setattr(self.module, attr, layer)
        self.model = SimpleNamespace(named_modules=lambda: [('gdn', self.module)])
        self.scope['torch'] = SimpleNamespace(float16=torch.float16,
                                             equal=lambda a, b: a.value == b.value)

    def prepare(self):
        self.scope['prepare_shared_gdn_quant'](self.model)
        return self.module._shared_input_quant_verified

    def test_equality_and_reload_invalidation(self):
        self.assertTrue(self.prepare())
        for field in self.fields:
            param = getattr(self.module.in_proj_ba, field)
            param.value = 2
            self.assertFalse(self.prepare(), field)
            param.value = 1
            self.assertTrue(self.prepare())
        self.env.VLLM_ASCEND_GDN_SHARED_INPUT_QUANT = False
        self.assertFalse(self.prepare())

    def test_structural_guards(self):
        for target, field, value in (
            (self.module, 'tp_size', 2), (self.module, 'gqa_interleaved_layout', True),
            (self.module.in_proj_ba, 'custom_op', object()),
            (self.module.in_proj_qkvz, 'tp_size', 2),
            (self.module.in_proj_qkvz, 'bias', object()),
            (self.module.in_proj_qkvz, 'params_dtype', torch.float32),
            (self.module.in_proj_ba.aclnn_input_offset, 'dtype', torch.int8),
            (self.module.in_proj_qkvz.aclnn_input_scale, 'shape', (1,)),
            (self.module.in_proj_qkvz.aclnn_input_scale, 'device', SimpleNamespace(type='cpu')),
            (self.module.in_proj_qkvz.quant_method, 'quant_method', SimpleNamespace(
                accepts_prequantized_input=False)),
            (self.module, 'in_proj_qkvz', named('LoRALinear')),
        ):
            old = getattr(target, field)
            setattr(target, field, value)
            self.assertFalse(self.prepare(), field)
            setattr(target, field, old)

    def test_shared_value_no_activation_cache_and_preserved_output_dtype(self):
        quant = Mock(side_effect=lambda x, *args: x.to(torch.int8))
        core = Mock()
        proxy = SimpleNamespace(int8=torch.int8, zeros=torch.zeros,
                                ops=SimpleNamespace(vllm=SimpleNamespace(quantize=quant, qwen_gdn_attention_core=core)))
        self.scope['torch'] = proxy
        qkvz = Mock(return_value=(torch.ones(3, 24, dtype=torch.float16), None))
        ba = Mock(return_value=(torch.ones(3, 4, dtype=torch.float16), None))
        for field in self.fields:
            setattr(qkvz, field, torch.ones(4096, dtype=torch.float16))
        module = SimpleNamespace(_shared_input_quant_verified=True, in_proj_qkvz=qkvz, in_proj_ba=ba,
                                 gqa_interleaved_layout=False, key_dim=4, value_dim=8, tp_size=1,
                                 head_v_dim=4, num_v_heads=2, split_ba=lambda x: x.chunk(2, -1),
                                 prefix='layer', _output_projection=Mock())
        fn = self.scope['shared_quant_forward_cuda']
        x, output = torch.ones(3, 4096, dtype=torch.float16), torch.empty(3, 4096, dtype=torch.float16)
        fn(module, x, output)
        self.assertEqual(quant.call_count, 1)
        self.assertIs(qkvz.call_args.args[0], ba.call_args.args[0])
        self.assertEqual(qkvz.call_args.args[0].dtype, torch.int8)
        self.assertEqual(core.call_args.args[3].dtype, torch.float16)
        fn(module, x * 2, output)
        self.assertEqual(quant.call_count, 2)
        self.assertTrue(torch.equal(qkvz.call_args.args[0], (x * 2).to(torch.int8)))
        module._shared_input_quant_verified = False
        self.assertEqual(fn(module, x, output), 'original')
        self.assertEqual(quant.call_count, 2)

    def test_forward_ast_only_changes_projection_quantization(self):
        upstream = self.root.parent / 'vllm-dflash-024/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
        if not upstream.exists():
            upstream = Path('/workspace/vllm-024/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py')
        tree = ast.parse(upstream.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenGatedDeltaNetAttention')
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_cuda')
        ours = next(n for n in ast.parse(self.path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'shared_quant_forward_cuda')
        class Normalize(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == 'projection_input':
                    node.id = 'hidden_states'
                return node
        body = [ours.body[1]] + ours.body[4:]
        body = [Normalize().visit(n) for n in body]
        self.assertEqual(ast.dump(ast.Module(body=body, type_ignores=[])),
                         ast.dump(ast.Module(body=original.body[1:], type_ignores=[])))


if __name__ == '__main__':
    unittest.main()
