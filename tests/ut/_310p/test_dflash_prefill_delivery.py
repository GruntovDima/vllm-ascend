"""CPU control-flow checks without importing the accelerator runtime."""

import ast
from collections import deque
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


def load_functions():
    root = Path(__file__).resolve().parents[3]
    source = root / "vllm_ascend/patch/platform/patch_dflash_prefill_delivery.py"
    tree = ast.parse(source.read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    scope = {"wraps": wraps, "logger": Mock()}
    exec(compile(functions, str(source), "exec"), scope)
    return scope


def fixture(computed=1266, scheduled=782):
    output = NS(
        pending_structured_output_tokens=False,
        num_scheduled_tokens={"r": scheduled},
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=NS(req_ids=["r"], num_computed_tokens=[computed]),
    )
    core = NS(
        async_scheduling=True,
        is_pooling_model=False,
        vllm_config=NS(
            speculative_config=NS(method="dflash"),
            parallel_config=NS(data_parallel_size=1, pipeline_parallel_size=1),
            scheduler_config=NS(max_num_seqs=1),
        ),
        scheduler=NS(requests={"r": NS(num_prompt_tokens=2048, num_output_tokens=0,
                                      use_structured_output=False, num_computed_tokens=9999)}),
    )
    return core, output


class PrefillDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.scope = load_functions()
        self.guard = self.scope["_is_final_prompt_batch"]

    def test_cached_final_prompt_uses_snapshot_not_advanced_request(self):
        core, output = fixture()
        self.assertTrue(self.guard(core, output))

    def test_complete_new_prompt(self):
        core, output = fixture(0, 2048)
        output.scheduled_new_reqs = [NS(req_id="r", num_computed_tokens=0)]
        output.scheduled_cached_reqs = NS(req_ids=[], num_computed_tokens=[])
        self.assertTrue(self.guard(core, output))

    def test_incomplete_prompt_and_decode_do_not_drain(self):
        for computed, count in ((0, 1266), (1266, 781), (2048, 1), (2064, 16)):
            with self.subTest(computed=computed, count=count):
                core, output = fixture(computed, count)
                self.assertFalse(self.guard(core, output))

    def test_unsupported_modes_and_request_states_fall_back(self):
        mutations = (
            lambda c, o: setattr(c, "async_scheduling", False),
            lambda c, o: setattr(c, "is_pooling_model", True),
            lambda c, o: setattr(c.vllm_config, "speculative_config", None),
            lambda c, o: setattr(c.vllm_config.speculative_config, "method", "mtp"),
            lambda c, o: setattr(c.vllm_config.parallel_config, "data_parallel_size", 2),
            lambda c, o: setattr(c.vllm_config.parallel_config, "pipeline_parallel_size", 2),
            lambda c, o: setattr(c.vllm_config.scheduler_config, "max_num_seqs", 2),
            lambda c, o: setattr(o, "pending_structured_output_tokens", True),
            lambda c, o: setattr(o, "scheduled_spec_decode_tokens", {"r": [1]}),
            lambda c, o: setattr(o, "num_scheduled_tokens", {}),
            lambda c, o: c.scheduler.requests.clear(),
            lambda c, o: setattr(c.scheduler.requests["r"], "num_output_tokens", 1),
            lambda c, o: setattr(c.scheduler.requests["r"], "use_structured_output", True),
            lambda c, o: setattr(o.scheduled_cached_reqs, "req_ids", []),
        )
        for mutation in mutations:
            core, output = fixture()
            mutation(core, output)
            self.assertFalse(self.guard(core, output))

    def test_drain_preserves_abort_update_order_and_consumes_one_future(self):
        core, output = fixture()
        calls = []
        model_result = object()
        future = Mock()
        future.result.side_effect = lambda: calls.append("result") or model_result
        core.batch_queue = deque([(future, output, Mock())])
        core.log_error_detail = lambda _: nullcontext()
        core.log_iteration_details = lambda _: nullcontext()
        core._process_aborts_queue = lambda: calls.append("abort")
        core.scheduler.update_from_output = lambda o, m: calls.append("update") or {"received": m}
        actual = self.scope["_drain_first_prompt_output"](core)
        self.assertEqual(calls, ["result", "abort", "update"])
        self.assertEqual(actual, ({"received": model_result}, False))
        self.assertEqual(len(core.batch_queue), 0)
        future.result.assert_called_once_with()

    def test_model_execution_error_is_not_suppressed(self):
        core, output = fixture()
        future = Mock()
        future.result.return_value = None
        exec_future = Mock()
        exec_future.result.side_effect = ValueError("model failed")
        core.batch_queue = deque([(future, output, exec_future)])
        core.log_error_detail = lambda _: nullcontext()
        core.log_iteration_details = lambda _: nullcontext()
        with self.assertRaisesRegex(ValueError, "model failed"):
            self.scope["_drain_first_prompt_output"](core)

    def test_wrapper_is_idempotent_and_decode_calls_original(self):
        class Core:
            def step_with_batch_queue(self):
                return "upstream"

        self.scope["EngineCore"] = Core
        self.scope["_patch_engine_core"]()
        patched = Core.step_with_batch_queue
        self.scope["_patch_engine_core"]()
        self.assertIs(Core.step_with_batch_queue, patched)
        core, output = fixture(2048, 16)
        instance = Core()
        instance.__dict__.update(vars(core))
        instance.batch_queue = deque([(Mock(), output, Mock())])
        self.assertEqual(instance.step_with_batch_queue(), "upstream")
        self.assertEqual(len(instance.batch_queue), 1)

    def test_delivery_utility_reports_binding_and_rejects_inflight_switch(self):
        core = NS(batch_queue=deque(), step_fn=Mock(_ascend_dflash_prefill_delivery=True))
        stats = self.scope["_delivery_stats"]
        self.assertEqual(stats(core), {"enabled": True, "count": 0, "queue_depth": 0, "step_fn_patched": True})
        self.assertFalse(stats(core, False)["enabled"])
        core.batch_queue.append(object())
        with self.assertRaises(RuntimeError):
            stats(core, True)
        self.assertFalse(stats(core)["enabled"])
        with self.assertRaises(TypeError):
            stats(core, 1)


if __name__ == "__main__":
    unittest.main()
