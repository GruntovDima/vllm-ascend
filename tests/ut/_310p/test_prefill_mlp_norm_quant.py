"""CPU guards and startup preparation; NPU numerics are separately screened."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


class FakeTensor(SimpleNamespace):
    def __radd__(self, value):
        assert value == 1.0
        return self


def tensor(shape, dtype=torch.float16):
    return FakeTensor(shape=shape, ndim=len(shape), dtype=dtype, device=SimpleNamespace(type="npu"))


def named(name, **fields):
    return type(name, (SimpleNamespace,), {})(**fields)


class PrefillMlpNormQuantTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/ops/prefill_mlp_norm_quant.py"
        funcs = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)]
        self.env = SimpleNamespace(VLLM_ASCEND_PREFILL_MLP_NORM_QUANT=True)
        self.ctx = SimpleNamespace(cudagraph_runtime_mode="NONE")
        self.tp = 1
        self.available = True
        self.fused = Mock(return_value=("int8", None, "residual"))
        self.scope = {"torch": torch, "torch_npu": SimpleNamespace(npu_add_rms_norm_quant=self.fused),
                      "envs": self.env, "PREFILL_MIN_TOKENS": 128, "HIDDEN_SIZE": 4096,
                      "CUDAGraphMode": SimpleNamespace(NONE="NONE"),
                      "get_forward_context": lambda: self.ctx,
                      "is_forward_context_available": lambda: self.available,
                      "get_tensor_model_parallel_world_size": lambda: self.tp}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), self.scope)
        self.linear = named("MergedColumnParallelLinear", bias=None,
                            _prefill_norm_quant_scale=tensor((4096,), torch.float32),
                            _prefill_norm_quant_offset=tensor((4096,), torch.int32),
                            quant_method=SimpleNamespace(quant_method=named("AscendW8A8LinearMethod310")))
        self.norm = named("AscendGemmaRMSNorm310", weight=tensor((4096,)), bias=None, variance_epsilon=1e-6)
        self.mlp = named("Qwen2MoeMLP", gate_up_proj=self.linear, expert_gate=None)
        self.decoder = SimpleNamespace(post_attention_layernorm=self.norm, mlp=self.mlp)
        self.x, self.residual = tensor((782, 4096)), tensor((782, 4096))

    def call(self):
        return self.scope["maybe_fused_mlp_norm_quant"](self.decoder, self.x, self.residual)

    def test_selected_path_and_contract(self):
        self.assertEqual(self.call(), ("int8", "residual"))
        args, kwargs = self.fused.call_args
        self.assertIs(args[0], self.x)
        self.assertIs(args[1], self.residual)
        self.assertIs(args[3], self.linear._prefill_norm_quant_scale)
        self.assertIs(args[4], self.linear._prefill_norm_quant_offset)
        self.assertEqual(kwargs, {"epsilon": 1e-6, "div_mode": False})

    def test_decode_graph_and_configuration_guards(self):
        for target, field, value in (
            (self.env, "VLLM_ASCEND_PREFILL_MLP_NORM_QUANT", False),
            (self.ctx, "cudagraph_runtime_mode", "FULL"),
            (self, "tp", 2), (self, "available", False),
            (self.mlp, "expert_gate", object()),
            (self.norm, "bias", object()), (self.linear, "bias", object()),
            (self.linear, "_prefill_norm_quant_scale", None),
            (self, "residual", None),
        ):
            old = getattr(target, field)
            setattr(target, field, value)
            self.assertIsNone(self.call(), field)
            setattr(target, field, old)
        for rows in (0, 1, 16, 127):
            self.x, self.residual = tensor((rows, 4096)), tensor((rows, 4096))
            self.assertIsNone(self.call())
        self.assertEqual(self.fused.call_count, 0)

    def test_dtype_shape_and_scheme_guards(self):
        for target, field, value in (
            (self.x, "dtype", torch.int8), (self.residual, "dtype", torch.float32),
            (self.norm.weight, "shape", (2048,)),
            (self.x, "device", SimpleNamespace(type="cpu")),
            (self.x, "shape", (782, 2048)),
            (self.linear._prefill_norm_quant_scale, "dtype", torch.float16),
            (self.linear._prefill_norm_quant_offset, "shape", (1,)),
            (self.linear.quant_method, "quant_method", named("DynamicScheme")),
            (self.mlp, "gate_up_proj", named("LoRAMergedLinear")),
        ):
            old = getattr(target, field)
            setattr(target, field, value)
            self.assertIsNone(self.call(), field)
            setattr(target, field, old)
        self.assertEqual(self.fused.call_count, 0)

    def test_prepare_uses_fp16_reciprocal_and_nonpersistent_buffers(self):
        layer = torch.nn.Module()
        layer.prefix = "model.layers.0.mlp.gate_up_proj"
        layer.params_dtype = torch.float16
        layer.input_offset = torch.tensor([15], dtype=torch.int8)
        layer.aclnn_input_offset = torch.full((4096,), 15, dtype=torch.float16)
        scale = torch.full((4096,), 0.324462890625, dtype=torch.float16)
        layer.aclnn_input_scale_reciprocal = 1 / scale
        prepare = self.scope["prepare_mlp_norm_quant_params"]
        self.env.VLLM_ASCEND_PREFILL_MLP_NORM_QUANT = False
        prepare(layer)
        self.assertFalse(hasattr(layer, "_prefill_norm_quant_scale"))
        self.env.VLLM_ASCEND_PREFILL_MLP_NORM_QUANT = True
        prepare(layer)
        self.assertTrue(torch.equal(layer._prefill_norm_quant_scale, (1 / scale).float()))
        self.assertEqual(layer._prefill_norm_quant_offset.dtype, torch.int32)
        self.assertNotIn("_prefill_norm_quant_scale", layer.state_dict())
        self.assertNotIn("_prefill_norm_quant_offset", layer.state_dict())
        prepare(layer)  # Weight-reprocessing replaces the buffers safely.


if __name__ == "__main__":
    unittest.main()
