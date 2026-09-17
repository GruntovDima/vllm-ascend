"""CPU contract tests; real W8A8 row equivalence needs the NPU audit."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
import unittest
import weakref

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
maybe_selected_attention = NAMESPACE["maybe_last_prefill_attention_output"]

PATCH_SOURCE = SOURCE.parents[2] / "patch/worker/patch_qwen3_5_mlp_310p.py"
PATCH_TREE = ast.parse(PATCH_SOURCE.read_text())
PATCH_NAMESPACE = {"torch": torch, "maybe_last_prefill_attention_output": lambda *args: None}
exec(compile(ast.Module(body=[n for n in PATCH_TREE.body
                             if isinstance(n, ast.FunctionDef) and
                             n.name == "last_prefill_attention_forward"],
                        type_ignores=[]), str(PATCH_SOURCE), "exec"), PATCH_NAMESPACE)
attention_forward = PATCH_NAMESPACE["last_prefill_attention_forward"]


def context(**kwargs):
    fields = dict(last_prefill_mlp_allowed=True, skip_compiled=True, in_profile_run=False,
                  capturing=False, is_draft_model=False, cudagraph_runtime_mode=SimpleNamespace(name="NONE"),
                  last_prefill_mlp_rows=782)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


class LastPrefillMLPTests(unittest.TestCase):
    def load_310p_patch(self, attention_enabled):
        class Qwen3_5DecoderLayer:
            pass

        class Qwen3NextAttention:
            def forward(self, *args, **kwargs):
                return "upstream"

        env = SimpleNamespace(VLLM_ASCEND_PREFILL_MLP_NORM_QUANT=False,
                              VLLM_ASCEND_LAST_PREFILL_MLP=True,
                              VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT=attention_enabled)
        modules = {name: ModuleType(name) for name in (
            "vllm", "vllm.model_executor", "vllm.model_executor.models",
            "vllm.model_executor.models.qwen3_5", "vllm.model_executor.models.qwen3_next",
            "vllm_ascend", "vllm_ascend._310p", "vllm_ascend._310p.ops",
            "vllm_ascend._310p.ops.last_prefill_mlp",
            "vllm_ascend._310p.ops.prefill_mlp_norm_quant", "vllm_ascend.utils",
        )}
        modules["vllm.model_executor.models.qwen3_5"].Qwen3_5DecoderLayer = Qwen3_5DecoderLayer
        modules["vllm.model_executor.models.qwen3_next"].Qwen3NextAttention = Qwen3NextAttention
        modules["vllm_ascend"].envs = env
        modules["vllm_ascend._310p.ops.last_prefill_mlp"].maybe_last_prefill_attention_output = (
            maybe_selected_attention
        )
        modules["vllm_ascend._310p.ops.last_prefill_mlp"].maybe_last_prefill_mlp = lambda *args: None
        modules["vllm_ascend._310p.ops.prefill_mlp_norm_quant"].maybe_fused_mlp_norm_quant = (
            lambda *args: None
        )
        modules["vllm_ascend.utils"].vllm_version_is = lambda version: version == "0.24.0"
        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            spec = importlib.util.spec_from_file_location("_test_qwen3_5_mlp_310p", PATCH_SOURCE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            for name, previous in saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        return Qwen3NextAttention, module

    def test_310p_module_registration_is_flag_gated(self):
        attention, module = self.load_310p_patch(False)
        self.assertIsNot(attention.forward, module.last_prefill_attention_forward)
        attention, module = self.load_310p_patch(True)
        self.assertIs(attention.forward, module.last_prefill_attention_forward)

    def test_prepared_attention_reaches_bound_selected_dispatch(self):
        attention_type, _ = self.load_310p_patch(True)
        attention = attention_type()
        attention.qkv_proj = lambda value: (value, None)
        attention._project_qkv_gate = lambda qkv, positions: (qkv, qkv, qkv, qkv)
        attention.attn = lambda q, k, v: q + k + v
        attention.o_proj = lambda value: (value * 2, None)
        decoder = SimpleNamespace(
            layer_type="full_attention", layer_scale=False,
            mlp=type("Qwen2MoeMLP", (), {})(), self_attn=attention,
            post_attention_layernorm=SimpleNamespace(),
        )
        backbone_type = type("Qwen3_5Model", (), {})
        backbone = backbone_type()
        backbone.config = SimpleNamespace(num_hidden_layers=2)
        backbone.start_layer = 0
        backbone.end_layer = 2
        backbone.aux_hidden_state_layers = (0,)
        backbone.norm = SimpleNamespace()
        backbone.layers = [SimpleNamespace(), decoder]
        namespace = dict(NAMESPACE)
        namespace.update(weakref=weakref,
                         envs=SimpleNamespace(VLLM_ASCEND_LAST_PREFILL_MLP=True,
                                              VLLM_ASCEND_LAST_PREFILL_NORMS=False,
                                              VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT=True,
                                              VLLM_ASCEND_PREFILL_MLP_NORM_QUANT=False,
                                              VLLM_ASCEND_GEMMA_PREFILL_ADD_RMS_NORM=False))
        prepare_node = next(n for n in TREE.body
                            if isinstance(n, ast.FunctionDef) and n.name == "prepare_last_prefill_mlp")
        exec(compile(ast.Module(body=[prepare_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        namespace["prepare_last_prefill_mlp"](SimpleNamespace(modules=lambda: [backbone]))

        rows = 782
        hidden = torch.arange(rows * 8, dtype=torch.float16).reshape(rows, 8)
        before = hidden.clone()
        output = torch.empty_like(hidden)
        old_envs = NAMESPACE.get("envs")
        old_backbone = NAMESPACE.get("last_prefill_backbone")
        old_context_available = NAMESPACE.get("is_forward_context_available")
        old_get_context = NAMESPACE.get("get_forward_context")
        old_is_compiling = torch.compiler.is_compiling
        try:
            runtime_context = context(last_prefill_mlp_rows=rows)
            NAMESPACE["envs"] = SimpleNamespace(VLLM_ASCEND_LAST_PREFILL_MLP=True,
                                                VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT=True)
            NAMESPACE["is_forward_context_available"] = lambda: True
            NAMESPACE["get_forward_context"] = lambda: runtime_context
            torch.compiler.is_compiling = lambda: False

            def npu_device_shim(module, value):
                metadata = SimpleNamespace(ndim=value.ndim, dtype=value.dtype,
                                           device=SimpleNamespace(type="npu"), shape=value.shape)
                return old_backbone(module, metadata)

            NAMESPACE["last_prefill_backbone"] = npu_device_shim
            attention.forward(torch.arange(rows), output, hidden)
            expected = (hidden[-1:] * 3) * torch.sigmoid(hidden[-1:]) * 2
            self.assertTrue(torch.equal(output[-1:], expected))
            self.assertEqual(torch.count_nonzero(output[:-1]), 0)
            self.assertTrue(torch.equal(hidden, before))
            self.assertEqual(attention.last_prefill_attn_output_calls, 1)
            self.assertEqual(attention.last_prefill_attn_output_skipped_rows, rows - 1)

            for guarded in (context(is_draft_model=True, last_prefill_mlp_rows=rows),
                            context(cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
                                    last_prefill_mlp_rows=rows),
                            context(last_prefill_mlp_rows=rows - 1)):
                runtime_context = guarded
                attention.forward(torch.arange(rows), output, hidden)
                self.assertNotEqual(torch.count_nonzero(output[1:-1]), 0)
                self.assertEqual(attention.last_prefill_attn_output_calls, 1)
        finally:
            torch.compiler.is_compiling = old_is_compiling
            NAMESPACE["last_prefill_backbone"] = old_backbone
            for name, previous in (("is_forward_context_available", old_context_available),
                                   ("get_forward_context", old_get_context)):
                if previous is None:
                    NAMESPACE.pop(name, None)
                else:
                    NAMESPACE[name] = previous
            if old_envs is None:
                NAMESPACE.pop("envs", None)
            else:
                NAMESPACE["envs"] = old_envs

    def test_attention_selection_is_off_and_compile_safe(self):
        old_envs = NAMESPACE.get("envs")
        old_is_compiling = torch.compiler.is_compiling
        try:
            attention = SimpleNamespace()
            x = torch.ones((128, 8))
            NAMESPACE["envs"] = SimpleNamespace(VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT=False)
            self.assertIsNone(maybe_selected_attention(attention, x, None))
            NAMESPACE["envs"] = SimpleNamespace(VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT=True)
            torch.compiler.is_compiling = lambda: True
            self.assertIsNone(maybe_selected_attention(attention, x, None))
        finally:
            torch.compiler.is_compiling = old_is_compiling
            if old_envs is None:
                NAMESPACE.pop("envs", None)
            else:
                NAMESPACE["envs"] = old_envs

    def test_prepare_binds_actual_last_attention_module(self):
        class Qwen3_5Model:
            def __init__(self):
                self.config = SimpleNamespace(num_hidden_layers=2)
                self.start_layer = 0
                self.end_layer = 2
                self.aux_hidden_state_layers = (0,)
                self.norm = SimpleNamespace()
                decoder = SimpleNamespace(
                    layer_type="full_attention",
                    layer_scale=False,
                    mlp=type("Qwen2MoeMLP", (), {})(),
                    self_attn=SimpleNamespace(),
                    post_attention_layernorm=SimpleNamespace(),
                )
                self.layers = [SimpleNamespace(), decoder]

        backbone = Qwen3_5Model()
        model = SimpleNamespace(modules=lambda: [backbone])
        namespace = dict(NAMESPACE)
        namespace.update(weakref=weakref,
                         envs=SimpleNamespace(VLLM_ASCEND_LAST_PREFILL_MLP=True,
                                              VLLM_ASCEND_LAST_PREFILL_NORMS=False,
                                              VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT=True,
                                              VLLM_ASCEND_PREFILL_MLP_NORM_QUANT=False,
                                              VLLM_ASCEND_GEMMA_PREFILL_ADD_RMS_NORM=False))
        prepare_node = next(n for n in TREE.body
                            if isinstance(n, ast.FunctionDef) and n.name == "prepare_last_prefill_mlp")
        exec(compile(ast.Module(body=[prepare_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        namespace["prepare_last_prefill_mlp"](model)
        attention = backbone.layers[-1].self_attn
        self.assertIs(attention._last_prefill_mlp_backbone(), backbone)
        self.assertEqual(attention.last_prefill_attn_output_calls, 0)
        self.assertEqual(attention.last_prefill_attn_output_skipped_rows, 0)

    def test_actual_attention_dispatch_preserves_upstream_fallback(self):
        rows = 5
        hidden = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
        positions = torch.arange(rows)
        output = torch.empty_like(hidden)
        calls = []

        attention = SimpleNamespace(
            qkv_proj=lambda value: (value + 1, None),
            _project_qkv_gate=lambda qkv, pos: (qkv + 1, qkv + 2, qkv + 3, qkv - 1),
            attn=lambda q, k, v: q + k + v,
            o_proj=lambda value: (value * 2, None),
        )
        original_dispatch = attention_forward.__globals__["maybe_last_prefill_attention_output"]
        try:
            attention_forward.__globals__["maybe_last_prefill_attention_output"] = (
                lambda module, value, gate: calls.append((module, value.clone(), gate.clone())) or None
            )
            attention_forward(attention, positions, output, hidden)
            qkv = hidden + 1
            expected = (qkv + 1 + qkv + 2 + qkv + 3) * torch.sigmoid(qkv - 1) * 2
            self.assertTrue(torch.equal(output, expected))
            self.assertEqual(len(calls), 1)

            selected = torch.zeros_like(output)
            selected[-1] = 7
            attention.o_proj = lambda value: self.fail("o_proj fallback ran after selected dispatch")
            attention_forward.__globals__["maybe_last_prefill_attention_output"] = (
                lambda module, value, gate: selected
            )
            attention_forward(attention, positions, output, hidden)
            self.assertTrue(torch.equal(output, selected))
        finally:
            attention_forward.__globals__["maybe_last_prefill_attention_output"] = original_dispatch

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
            self.assertTrue(eligible(context(last_prefill_mlp_rows=rows), rows,
                                     (2, 6, 10, 14, 18, 22, 26, 30), 32))
        for rows in (0, 1, 16, 32, 127):
            self.assertFalse(eligible(context(last_prefill_mlp_rows=rows), rows, (30,), 32))

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
