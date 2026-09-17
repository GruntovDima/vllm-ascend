"""CPU behavioral checks for the standalone QBMM device runner."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[3]
RUNNER = (
    ROOT
    / "csrc/qbmm/quant_batch_matmul_v3_x/tests/device/run_qbmm_k_pipeline_correctness.py"
)


@unittest.skipUnless(torch is not None, "PyTorch is required for behavioral checks")
class QbmmKPipelineDeviceRunnerCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub = types.ModuleType("torch_npu")
        stub.__version__ = "test-stub"
        spec = importlib.util.spec_from_file_location("qbmm_device_runner", RUNNER)
        cls.runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        with mock.patch.dict(sys.modules, {"torch_npu": stub}):
            spec.loader.exec_module(cls.runner)

    def test_requested_sequence_and_raw_call_schema_split(self):
        calls = []

        def op(*args, **kwargs):
            calls.append(kwargs)
            return None

        self.assertEqual(self.runner.requested_sequence("old"), (None,))
        self.assertEqual(self.runner.requested_sequence("cycle"), (False, True, False))
        self.runner.call_raw(op, 1, 2, 3, None, None)
        self.assertNotIn("enable_k_pipeline", calls[-1])
        self.runner.call_raw(op, 1, 2, 3, None, True)
        self.assertIs(calls[-1]["enable_k_pipeline"], True)
        self.assertIs(calls[-1]["transpose_x2"], True)

    def test_bitwise_comparison_distinguishes_signed_zero(self):
        positive = torch.tensor([0.0], dtype=torch.float16)
        negative = torch.tensor([-0.0], dtype=torch.float16)
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(self.runner.bitwise_equal(positive, negative))

    def test_exact_blocked_golden_matches_scalar_integer_oracle(self):
        x = torch.tensor([[1, -1, 1, 0]], dtype=torch.int8)
        weight = torch.tensor([[1, 1, -1, 0], [-1, 0, 1, 1]], dtype=torch.int8)
        scales = torch.tensor([0.5, 0.25], dtype=torch.float32)
        bias = torch.tensor([2, -3], dtype=torch.int32)
        actual = self.runner.exact_blocked_integer_golden(x, weight, scales, bias)
        expected = torch.tensor([[0.5, -0.75]], dtype=torch.float16)
        self.assertTrue(self.runner.bitwise_equal(actual, expected))

    def test_scale_encoding_preserves_fp32_bits_and_marker(self):
        scales = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
        encoded = self.runner.encode_vdeq16_scale(scales)
        self.assertTrue(torch.equal(
            encoded & 0xFFFFFFFF,
            scales.view(torch.int32).to(torch.int64) & 0xFFFFFFFF,
        ))
        self.assertTrue(torch.equal(
            encoded >> 32,
            torch.full((3,), self.runner.VDEQ16_MARKER, dtype=torch.int64),
        ))

    def test_reference_validation_rejects_candidate_or_incomplete_capture(self):
        case = {
            "name": "small_m1",
            "bias": False,
            "variant": "on",
            "requested_sequence": [True],
        }
        payload = {
            "status": "PASS",
            "variant": "on",
            "schema": "enable_k_pipeline",
            "seed": self.runner.SEED,
            "runner_sha256": self.runner.tensor_file_sha256(RUNNER),
            "cases": [case],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad-reference.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "variant old"):
                self.runner.load_reference(path)

    def test_reference_validation_rejects_missing_and_duplicate_cases(self):
        cases = [
            {
                "name": name,
                "bias": with_bias,
                "variant": "old",
                "requested_sequence": [None],
            }
            for name, *_ in self.runner.CASES
            for with_bias in (False, True)
        ]
        payload = {
            "status": "PASS",
            "variant": "old",
            "schema": "quant_batch_matmul_v3_x(Tensor x) -> Tensor",
            "seed": self.runner.SEED,
            "runner_sha256": self.runner.tensor_file_sha256(RUNNER),
            "cases": cases,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.json"
            path.write_text(json.dumps(payload))
            self.assertEqual(
                set(self.runner.load_reference(path)),
                self.runner.expected_result_keys(),
            )
            payload["cases"] = cases[:-1]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "case matrix differs"):
                self.runner.load_reference(path)
            payload["cases"] = cases + [cases[0]]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "duplicate reference case"):
                self.runner.load_reference(path)
            payload["cases"] = cases
            payload["seed"] += 1
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "seed differs"):
                self.runner.load_reference(path)
            payload["seed"] = self.runner.SEED
            payload["runner_sha256"] = "not-this-runner"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "exact runner"):
                self.runner.load_reference(path)


if __name__ == "__main__":
    unittest.main()
