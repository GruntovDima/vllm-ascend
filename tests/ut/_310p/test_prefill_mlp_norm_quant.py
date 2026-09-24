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
        self.opaque = Mock(return_value=("int8", "residual"))
        torch_proxy = SimpleNamespace(**{key: getattr(torch, key) for key in
                                     ("Tensor", "float16", "float32", "int8", "int32", "qint8", "empty_like")})
        torch_proxy.ops = SimpleNamespace(vllm=SimpleNamespace(prefill_mlp_norm_quant=self.opaque))
        self.scope = {"torch": torch, "torch_npu": SimpleNamespace(npu_add_rms_norm_quant=self.fused),
                      "envs": self.env, "PREFILL_MIN_TOKENS": 128,
                      "CUDAGraphMode": SimpleNamespace(NONE="NONE"),
                      "get_forward_context": lambda: self.ctx,
                      "is_forward_context_available": lambda: self.available,
                      "get_tensor_model_parallel_world_size": lambda: self.tp}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), self.scope)
        self.scope["torch"] = torch_proxy
        self.scope["ensure_mlp_norm_quant_registered"] = Mock()
        self.linear = named("MergedColumnParallelLinear", bias=None,
                            aclnn_input_scale_reciprocal=tensor((4096,)),
                            aclnn_input_offset=tensor((4096,)),
                            _prefill_norm_quant_scale=tensor((4096,), torch.float32),
                            _prefill_norm_quant_offset=tensor((4096,), torch.int32),
                            quant_method=SimpleNamespace(quant_method=SimpleNamespace(
                                accepts_prequantized_input=True)))
        self.norm = named("AscendGemmaRMSNorm310", weight=tensor((4096,)), bias=None, variance_epsilon=1e-6)
        self.mlp = named("Qwen2MoeMLP", gate_up_proj=self.linear, expert_gate=None)
        self.decoder = SimpleNamespace(post_attention_layernorm=self.norm, mlp=self.mlp)
        self.x, self.residual = tensor((782, 4096)), tensor((782, 4096))

    def call(self):
        return self.scope["maybe_fused_mlp_norm_quant"](self.decoder, self.x, self.residual)

    def test_selected_path_and_contract(self):
        self.assertEqual(self.call(), ("int8", "residual"))
        args, kwargs = self.opaque.call_args
        self.assertIs(args[0], self.x)
        self.assertIs(args[1], self.residual)
        self.assertIs(args[3], self.linear.aclnn_input_scale_reciprocal)
        self.assertIs(args[5], self.linear._prefill_norm_quant_scale)
        self.assertIs(args[6], self.linear._prefill_norm_quant_offset)
        self.assertEqual(args[7], 1e-6)
        self.assertEqual(kwargs, {})

    def test_ascend_linear_wrapper_without_custom_parallel_op(self):
        linear = named("AscendMergedColumnParallelLinear", **vars(self.linear), tp_size=1, custom_op=None)
        self.mlp.gate_up_proj = linear
        self.assertEqual(self.call(), ("int8", "residual"))
        linear.custom_op = object()
        self.assertIsNone(self.call())
        linear.custom_op = None
        linear.tp_size = 2
        self.assertIsNone(self.call())

    def test_configuration_guards(self):
        for target, field, value in (
            (self.env, "VLLM_ASCEND_PREFILL_MLP_NORM_QUANT", False),
            (self, "tp", 2),
            (self.mlp, "expert_gate", object()),
            (self.norm, "bias", object()), (self.linear, "bias", object()),
            (self.linear, "_prefill_norm_quant_scale", None),
            (self, "residual", None),
        ):
            old = getattr(target, field)
            setattr(target, field, value)
            self.assertIsNone(self.call(), field)
            setattr(target, field, old)
        self.assertEqual(self.opaque.call_count, 0)

    def test_dispatch_is_not_folded_at_trace_time(self):
        self.ctx.cudagraph_runtime_mode = "FULL"
        self.available = False
        for rows in (1, 16, 127, 782, 1266):
            self.x, self.residual = tensor((rows, 4096)), tensor((rows, 4096))
            self.assertEqual(self.call(), ("int8", "residual"))
        self.assertEqual(self.opaque.call_count, 5)
        self.assertEqual(self.fused.call_count, 0)

    def test_runtime_prefill_and_decode_routes(self):
        norm = Mock(return_value=(torch.ones((16, 4096), dtype=torch.float16),))
        quant = Mock(return_value="split_int8")
        self.scope["torch_npu"].npu_rms_norm = norm
        self.scope["torch_npu"].npu_quantize = quant
        fn = self.scope["_mlp_norm_quant"]
        args = [torch.zeros((782, 4096), dtype=torch.float16)] * 2
        params = [torch.ones(4096)] * 5 + [1e-6]
        self.assertEqual(fn(*args, *params), ("int8", "residual"))
        self.assertEqual(self.fused.call_count, 1)
        for mode, rows, available in (("FULL", 782, True), ("NONE", 16, True), ("NONE", 782, False)):
            self.ctx.cudagraph_runtime_mode = mode
            self.available = available
            x = torch.ones((rows, 4096), dtype=torch.float16)
            out, residual = fn(x, x, *params)
            self.assertEqual(out, "split_int8")
            self.assertTrue(torch.equal(residual, x + x))
        self.assertEqual(self.fused.call_count, 1)
        self.assertEqual(norm.call_count, 3)
        self.assertEqual(quant.call_count, 3)

    def test_310p_entry_and_forward_preserve_upstream_except_norm(self):
        root = Path(__file__).resolve().parents[3]
        worker_init = ast.parse((root / "vllm_ascend/patch/worker/__init__.py").read_text())
        chip_branch = next(n for n in worker_init.body if isinstance(n, ast.If)
                           and ast.unparse(n.test) == "not is_310p()")
        imports = [alias.name for n in chip_branch.orelse if isinstance(n, ast.Import) for alias in n.names]
        self.assertIn("vllm_ascend.patch.worker.patch_qwen3_5_mlp_310p", imports)
        local = ast.parse((root / "vllm_ascend/patch/worker/patch_qwen3_5_mlp_310p.py").read_text())
        forward = next(n for n in local.body if isinstance(n, ast.FunctionDef))
        upstream_path = root.parent / "vllm-dflash-024/vllm/model_executor/models/qwen3_next.py"
        if not upstream_path.exists():
            upstream_path = Path("/workspace/vllm-024/vllm/model_executor/models/qwen3_next.py")
        upstream = ast.parse(upstream_path.read_text())
        cls = next(n for n in upstream.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3NextDecoderLayer")
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        rewritten = []
        for statement in forward.body:
            if isinstance(statement, ast.Assign):
                target = ast.unparse(statement.targets[0])
                if target in ("fused_norm", "selected_output"):
                    continue
                if target == "hidden_states" and "selected_output" in ast.unparse(statement.value):
                    rewritten.append(ast.parse("hidden_states = self.mlp(hidden_states)").body[0])
                    continue
            if isinstance(statement, ast.If) and ast.unparse(statement.test) == "fused_norm is None":
                rewritten.extend(statement.body)
            else:
                rewritten.append(statement)
        self.assertEqual(ast.dump(ast.Module(body=rewritten, type_ignores=[])),
                         ast.dump(ast.Module(body=original.body, type_ignores=[])))

    def test_dtype_shape_and_scheme_guards(self):
        for target, field, value in (
            (self.x, "dtype", torch.int8), (self.residual, "dtype", torch.float32),
            (self.norm.weight, "shape", (2048,)),
            (self.x, "device", SimpleNamespace(type="cpu")),
            (self.x, "shape", (782, 2048)),
            (self.linear._prefill_norm_quant_scale, "dtype", torch.float16),
            (self.linear._prefill_norm_quant_offset, "shape", (1,)),
            (self.linear.quant_method, "quant_method", SimpleNamespace(
                accepts_prequantized_input=False)),
            (self.mlp, "gate_up_proj", named("IncompatibleLinear",
                quant_method=SimpleNamespace(quant_method=SimpleNamespace(
                    accepts_prequantized_input=False)))),
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
