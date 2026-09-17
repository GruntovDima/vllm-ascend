"""Dependency-free contract checks for the QBMM device runner."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = (
    ROOT
    / "csrc/qbmm/quant_batch_matmul_v3_x/tests/device/run_qbmm_k_pipeline_correctness.py"
)


class QbmmKPipelineDeviceRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text()
        cls.tree = ast.parse(cls.source)

    def test_case_matrix_contains_selected_and_fallback_boundaries(self):
        cases_node = next(
            node.value
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CASES"
                    for target in node.targets)
        )
        cases = ast.literal_eval(cases_node)
        self.assertIn(("small_m1", 1, 4096, 64, False), cases)
        self.assertIn(("small_m15", 15, 4096, 64, False), cases)
        self.assertIn(("small_m16", 16, 4096, 64, False), cases)
        self.assertIn(("selected_down_m1266", 1266, 12288, 4096, True), cases)
        self.assertIn(("fallback_down_m2048", 2048, 12288, 4096, False), cases)
        self.assertTrue(any(not expected for _, _, _, _, expected in cases))

    def test_variant_matrix_and_schema_gate_protect_old_binary(self):
        requested = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "requested_sequence"
        )
        self.assertIn("return (None,)", ast.unparse(requested))
        self.assertIn("return (False, True, False)", ast.unparse(requested))
        self.assertIn('variant == "old" and has_new_attr', self.source)
        self.assertIn('variant != "old" and not has_new_attr', self.source)
        self.assertIn('if requested is not None:', self.source)
        self.assertIn('choices=("old", "off", "on", "cycle")', self.source)

    def test_correctness_guards_cover_finite_bitwise_bias_and_immutability(self):
        self.assertIn("torch.isfinite", self.source)
        self.assertIn("for output in outputs[1:]", self.source)
        self.assertIn("for with_bias in (False, True)", self.source)
        for name in ("x", "weight_nz", "scale"):
            self.assertIn(f"torch.equal({name}, {name.replace('weight_nz', 'weight').replace('scale', 'scale')}_before)", self.source)
        self.assertIn("input_sha256", self.source)
        self.assertIn("sample_fp16_bits", self.source)

    def test_scale_encoding_and_schema_gate_are_explicit(self):
        self.assertIn("(bits & 0xFFFFFFFF) | (VDEQ16_MARKER << 32)", self.source)
        self.assertIn('has_new_attr = "enable_k_pipeline" in schema', self.source)
        self.assertIn('"transpose_x2": True', self.source)
        self.assertIn("npu_format_cast(weight_cpu.npu(), FRACTAL_NZ)", self.source)
        self.assertIn("INTEGER_GOLDEN_K_BLOCK = 32", self.source)
        self.assertIn("exact_blocked_integer_golden", self.source)
        self.assertIn('"old_custom_reference"', self.source)

    def test_padding_result_cannot_be_misreported_as_sanitizer_evidence(self):
        self.assertIn('"sanitizer_evidence": False', self.source)
        self.assertIn("this is not OOB-sanitizer evidence", self.source)
        self.assertIn('"observed_pipeline_route": None', self.source)
        self.assertNotIn("perf_counter", self.source)
        self.assertNotIn("Event(enable_timing", self.source)


if __name__ == "__main__":
    unittest.main()
