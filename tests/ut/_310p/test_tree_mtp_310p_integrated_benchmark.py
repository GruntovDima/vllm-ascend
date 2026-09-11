# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "tree_mtp_310p_integrated_benchmark.py"
SPEC = importlib.util.spec_from_file_location("tree_mtp_310p_integrated_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def make_args(**overrides):
    values = {
        "mode": "tree",
        "depth": 3,
        "width": 4,
        "sampling": "greedy",
        "stage": "screening",
        "run_id": "screen-tree-d3-w4-r01",
        "repetitions": 1,
        "output": Path("report.json"),
        "model": BENCHMARK.DEFAULT_MODEL,
        "physical_device": 0,
        "device": 0,
        "gpu_memory_utilization": 0.85,
        "safetensors_load_strategy": "eager",
        "tree_trace": False,
        "expected_prompt_sha256": BENCHMARK.EXPECTED_PROMPT_SHA256,
        "_worker": False,
        "_worker_output": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return f"<chat>{messages[0]['content']}</chat>"

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) for character in text]


class FakeTensor:
    dtype = "float16"
    shape = (32, 64)
    device = "npu:0"
    requires_grad = False


class FakeQuantMethod:
    quant_type = "W8A8"
    in_dtype = "float16"
    out_dtype = "float16"


class FakeHead:
    def __init__(self):
        self.weight = FakeTensor()
        self.quant_method = FakeQuantMethod()


class FakeModel:
    def __init__(self):
        self.lm_head = FakeHead()


class TestPlan(unittest.TestCase):
    def test_tree_depth_five_and_six_are_blocked_without_clamping(self):
        for depth in (5, 6):
            with self.subTest(depth=depth):
                plan = BENCHMARK.build_plan(make_args(depth=depth, width=16))
                self.assertIsNotNone(plan.blocked_reason)
                self.assertEqual(plan.depth, depth)
                self.assertEqual(plan.speculative_tokens, depth * 16)

    def test_tree_engine_configuration(self):
        args = make_args(depth=4, width=8)
        plan = BENCHMARK.build_plan(args)
        engine = BENCHMARK.build_engine_args(args, plan)
        self.assertTrue(engine["enforce_eager"])
        self.assertEqual(engine["mamba_cache_mode"], "none")
        self.assertEqual(engine["speculative_config"], {"method": "mtp", "num_speculative_tokens": 32})
        self.assertEqual(
            engine["additional_config"]["tree_mtp"],
            {"enabled": True, "width": 8, "depth": 4, "trace": False},
        )
        self.assertNotIn("compilation_config", engine)

    def test_linear_graph_is_an_ordinary_full_decode_only_control(self):
        args = make_args(mode="linear-graph", depth=6, width=None)
        plan = BENCHMARK.build_plan(args)
        engine = BENCHMARK.build_engine_args(args, plan)
        self.assertFalse(engine["enforce_eager"])
        self.assertEqual(engine["mamba_cache_mode"], "none")
        self.assertEqual(engine["speculative_config"], {"method": "mtp", "num_speculative_tokens": 6})
        self.assertEqual(engine["compilation_config"], {"cudagraph_mode": "FULL_DECODE_ONLY"})
        self.assertNotIn("tree_mtp", engine["additional_config"])

    def test_linear_eager_accepts_k_two_through_six(self):
        for depth in range(2, 7):
            with self.subTest(depth=depth):
                args = make_args(mode="linear-eager", depth=depth, width=None)
                plan = BENCHMARK.build_plan(args)
                self.assertIsNone(plan.blocked_reason)
                self.assertTrue(BENCHMARK.build_engine_args(args, plan)["enforce_eager"])

    def test_invalid_topologies_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --width"):
            BENCHMARK.build_plan(make_args(width=None))
        with self.assertRaisesRegex(ValueError, "tree width"):
            BENCHMARK.build_plan(make_args(width=3))
        with self.assertRaisesRegex(ValueError, "do not accept --width"):
            BENCHMARK.build_plan(make_args(mode="linear-eager"))

    def test_smoke_is_fixed_non_performance_workload(self):
        plan = BENCHMARK.build_plan(make_args(stage="smoke"))
        self.assertEqual(plan.input_tokens, 1024)
        self.assertEqual(plan.output_tokens, 32)
        self.assertFalse(plan.performance_eligible)

    def test_sampling_presets_are_separate_and_seeded(self):
        self.assertEqual(
            BENCHMARK.sampling_parameters("greedy"),
            {"temperature": 0.0, "top_k": -1, "top_p": 1.0, "seed": 42},
        )
        self.assertEqual(
            BENCHMARK.sampling_parameters("t1"),
            {"temperature": 1.0, "top_k": 50, "top_p": 0.9, "seed": 42},
        )

    def test_environment_contract_requires_prune_variable_to_be_empty(self):
        args = make_args()
        plan = BENCHMARK.build_plan(args)
        required = {
            "ASCEND_RT_VISIBLE_DEVICES": "0",
            "VLLM_CUSTOM_QBMM": "1",
            "VLLM_ASCEND_TREE_GDN_COMPACT": "1",
        }
        with mock.patch.dict(os.environ, required, clear=True):
            self.assertEqual(BENCHMARK.environment_contract(args, plan)["status"], "PASS")
            os.environ["VLLM_LMHEAD_PRUNE_PACK"] = "0"
            contract = BENCHMARK.environment_contract(args, plan)
            self.assertEqual(contract["status"], "FAIL")
            self.assertFalse(contract["checks"]["lmhead_prune_disabled"])


class TestEvidenceAndMetrics(unittest.TestCase):
    def test_prompt_is_exact_and_deterministic(self):
        first = BENCHMARK.construct_exact_prompt(FakeTokenizer())
        second = BENCHMARK.construct_exact_prompt(FakeTokenizer())
        self.assertEqual(len(first), 1024)
        self.assertEqual(first, second)
        self.assertEqual(BENCHMARK.prompt_sha256(first), BENCHMARK.prompt_sha256(second))

    def test_runtime_metadata_uses_actual_target_and_mtp_heads(self):
        target = FakeModel()
        mtp = FakeModel()
        runner = SimpleNamespace(model=target, drafter=SimpleNamespace(model=mtp))
        worker = SimpleNamespace(model_runner=runner, vllm_config=SimpleNamespace())
        evidence = BENCHMARK._worker_runtime_metadata(worker)
        self.assertEqual(evidence["status"], "AVAILABLE")
        self.assertEqual(evidence["target_head"]["weight"]["dtype"], "float16")
        self.assertEqual(evidence["mtp_head"]["quant_method"]["quant_type"], "W8A8")
        self.assertFalse(evidence["target_and_mtp_head_same_object"])

    def test_missing_mtp_head_is_declared_unavailable(self):
        runner = SimpleNamespace(model=FakeModel(), drafter=SimpleNamespace(model=SimpleNamespace()))
        evidence = BENCHMARK._worker_runtime_metadata(SimpleNamespace(model_runner=runner))
        self.assertEqual(evidence["status"], "UNAVAILABLE")
        self.assertEqual(evidence["mtp_head"]["status"], "UNAVAILABLE")

    def test_nested_language_model_head_is_runtime_evidence(self):
        model = SimpleNamespace(language_model=SimpleNamespace(model=FakeModel()))
        metadata = BENCHMARK._head_metadata(model, "target")
        self.assertEqual(metadata["status"], "AVAILABLE")
        self.assertEqual(metadata["head_locator"], "language_model.model.lm_head")

    def test_effective_graph_config_contract_guards_all_control_fields(self):
        args = make_args(mode="linear-graph", depth=6, width=None)
        plan = BENCHMARK.build_plan(args)
        evidence = {
            "effective_config": {
                "speculative_config": {"method": "mtp", "num_speculative_tokens": 6},
                "compilation_config": {
                    "cudagraph_mode": {
                        "class": "vllm.config.CUDAGraphMode",
                        "name": "FULL_DECODE_ONLY",
                        "value": 3,
                    }
                },
                "cache_config": {"mamba_cache_mode": "none", "enable_prefix_caching": False},
            }
        }
        self.assertEqual(BENCHMARK.effective_config_contract(evidence, plan)["status"], "PASS")
        evidence["effective_config"]["cache_config"]["enable_prefix_caching"] = True
        self.assertEqual(BENCHMARK.effective_config_contract(evidence, plan)["status"], "FAIL")

    def test_counter_delta_and_decode_only_timing(self):
        before = {"counters": {BENCHMARK.VERIFICATION_COUNTER: 10, BENCHMARK.ACCEPTED_COUNTER: 15}}
        after = {
            "counters": {
                BENCHMARK.VERIFICATION_COUNTER: 110,
                BENCHMARK.ACCEPTED_COUNTER: 215,
                BENCHMARK.PROPOSED_COUNTER: 400,
            }
        }
        delta = BENCHMARK.counter_delta(before, after)
        metrics = SimpleNamespace(arrival_time=10.0, first_token_time=10.2, last_token_time=12.247)
        derived = BENCHMARK.derive_timing_and_acceptance(metrics, 300, 3.0, delta)
        self.assertAlmostEqual(derived["ttft_ms"], 200.0)
        self.assertAlmostEqual(derived["tpot_ms_excluding_prefill"], 2047.0 / 299)
        self.assertEqual(derived["verification_cycles"], 100)
        self.assertEqual(derived["proposed_draft_tokens"], 400)
        self.assertEqual(derived["accepted_draft_tokens"], 200)
        self.assertEqual(derived["accepted_drafts_per_verification"], 2.0)
        self.assertEqual(derived["emitted_tokens_per_verification_actual"], 3.0)
        self.assertEqual(derived["emitted_tokens_per_verification_counter_derived"], 3.0)

    def test_blocked_report_is_written_without_worker_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "blocked.json"
            args = make_args(output=output, depth=5, width=4)
            plan = BENCHMARK.build_plan(args)
            returncode = BENCHMARK.run_supervisor(args, plan, ["test"])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(returncode, 0)
            self.assertEqual(report["status"], "BLOCKED_UNSUPPORTED")
            self.assertEqual(report["requested_config"]["depth"], 5)
            self.assertEqual(report["requested_config"]["speculative_tokens"], 20)
            self.assertNotIn("artifacts", report)


if __name__ == "__main__":
    unittest.main()
