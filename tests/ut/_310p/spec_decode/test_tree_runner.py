# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU tests of production tree runner methods, without initializing vLLM/NPU.

AST extraction keeps the methods under test identical to the runner source.
Only their native superclass, sampler container and model drafter are stubs.
Run directly with Python and torch; no device or checkpoint is accessed.
"""

import ast
import importlib.util
import json
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


def load_runner_methods():
    helper_spec = importlib.util.spec_from_file_location(
        "_tree_runtime_test_helpers", Path(__file__).with_name("test_tree_runtime.py")
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)
    tree, runtime = helpers.load_tree_modules()

    class NativeRunner:
        def _sample(self, *args):
            self.native_sample_args = args
            return self.native_result

        def propose_draft_token_ids(self, *args):
            self.native_propose_args = args
            return self.native_result

    source = Path(__file__).resolve().parents[4] / "vllm_ascend/_310p/model_runner_310p.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    original = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310")
    wanted = {"_prepare_tree_step", "_sample", "propose_draft_token_ids"}
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {method.name for method in methods} == wanted
    extracted = ast.ClassDef(
        name="NPUModelRunner310", bases=[ast.Name(id="NativeRunner", ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[], type_params=[],
    )
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), extracted,
    ], type_ignores=[])
    namespace = dict(
        NativeRunner=NativeRunner, torch=torch, copy=copy, json=json, logger=Mock(),
        TokenTree=tree.TokenTree, verify_target_samples=tree.verify_target_samples,
        TreeStepContext=runtime.TreeStepContext, validate_tree_sampling=runtime.validate_tree_sampling,
        SamplerOutput=SimpleNamespace,
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["NPUModelRunner310"], tree, runtime, helpers


class TestTreeRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner_type, cls.tree_module, cls.runtime, cls.helpers = load_runner_methods()

    def make_runner(self, drafts=(7, 8, 9, 10), prefix=12, prompt_tokens=4, **sampling_changes):
        runner = self.runner_type()
        runner.tree_mtp_config = self.runtime.TreeMTPConfig()
        runner._tree_step = None
        runner._tree_gdn_request_id = None
        runner._tree_gdn_indices = {}
        runner.device = torch.device("cpu")
        runner.uses_mrope = False
        runner.uses_xdrope_dim = 0
        runner.positions = torch.arange(prefix, prefix + 8)
        runner._positions_cpu_buf = runner.positions.clone()
        flat_tokens = torch.tensor([5, *drafts], dtype=torch.int32)
        tokens_cpu = torch.zeros((1, prefix + len(flat_tokens)), dtype=torch.int32)
        tokens_cpu[0, prefix:] = flat_tokens
        runner.input_batch = SimpleNamespace(
            num_reqs=1, req_ids=["request"], num_computed_tokens_cpu=[prefix],
            num_prompt_tokens=[prompt_tokens], token_ids_cpu=tokens_cpu,
            sampling_metadata=SimpleNamespace(all_greedy=sampling_changes.get("temperature", 0) == 0),
        )
        runner.sampler = SimpleNamespace(sample_tree=Mock(side_effect=lambda logits, metadata, **kw: logits.argmax(-1)))
        runner.input_ids = SimpleNamespace(gpu=flat_tokens)
        runner.num_accepted_tokens = SimpleNamespace(gpu=torch.tensor([2], dtype=torch.int32))
        runner.requests = {"request": SimpleNamespace(
            sampling_params=self.helpers.make_sampling_params(**sampling_changes),
            output_token_ids=[], mm_features=[],
        )}
        runner.scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"request": list(drafts)}, num_spec_tokens_to_schedule=4
        )
        runner._get_positions = lambda indices: runner.positions.index_select(0, indices)
        runner._copy_valid_sampled_token_count = Mock()
        runner.drafter = SimpleNamespace(_propose=Mock(return_value=torch.tensor([[1, 2, 3, 4]])))
        runner.native_result = object()
        return runner

    @staticmethod
    def logits_for(predictions):
        logits = torch.full((len(predictions), 16), -10.0)
        logits[torch.arange(len(predictions)), torch.tensor(predictions)] = 10.0
        return logits

    def prepare(self, runner):
        runner._prepare_tree_step(runner.scheduler_output, len(runner.input_ids.gpu))
        return runner._tree_step

    def test_flat_tokens_stay_physical_while_rope_uses_depth(self):
        runner = self.make_runner()
        original_ids = runner.input_ids.gpu.clone()
        context = self.prepare(runner)
        self.assertEqual(context.tree.parents, (-1, 0, 0, 1, 1))
        self.assertEqual(context.tree.depths, (0, 1, 1, 2, 2))
        self.assertEqual(runner.positions[:5].tolist(), [12, 13, 13, 14, 14])
        self.assertEqual(runner._positions_cpu_buf[:5].tolist(), [12, 13, 13, 14, 14])
        self.assertTrue(torch.equal(runner.input_ids.gpu, original_ids))
        runner.num_accepted_tokens.gpu.fill_(4)
        self.assertEqual(context.previous_accepted_tokens.tolist(), [2])

    def test_scheduler_can_truncate_topological_prefix_mid_row(self):
        for count in range(5):
            with self.subTest(count=count):
                runner = self.make_runner(drafts=(7, 8, 9, 10)[:count])
                context = self.prepare(runner)
                self.assertEqual(context.tree.parents, (-1, 0, 0, 1, 1)[:count + 1])
                self.assertEqual(context.num_nodes, count + 1)

    def test_mrope_and_xdrope_receive_same_depth_positions(self):
        for name, dimensions in (("mrope", 3), ("xdrope", 2)):
            with self.subTest(name=name):
                runner = self.make_runner()
                runner.uses_mrope = name == "mrope"
                runner.uses_xdrope_dim = dimensions if name == "xdrope" else 0
                buffers = SimpleNamespace(cpu=torch.zeros((dimensions, 8)), gpu=torch.zeros((dimensions, 8)))
                setattr(runner, f"{name}_positions", buffers)
                self.prepare(runner)
                self.assertEqual(buffers.cpu[:, :5].tolist(), [[12, 13, 13, 14, 14]] * dimensions)
                self.assertTrue(torch.equal(buffers.cpu, buffers.gpu))

    def test_prefill_and_disabled_tree_delegate(self):
        runner = self.make_runner(drafts=(), prefix=0, prompt_tokens=4)
        runner._tree_gdn_request_id = runner.input_batch.req_ids[0]
        runner._tree_gdn_indices["old_block"] = torch.tensor([99])
        self.assertIsNone(self.prepare(runner))
        self.assertEqual(runner._tree_gdn_indices, {})
        sentinel = object()
        self.assertIs(runner._sample(sentinel, None), runner.native_result)
        self.assertEqual(runner.native_sample_args, (sentinel, None))
        args = tuple(object() for _ in range(11))
        self.assertIs(runner.propose_draft_token_ids(*args), runner.native_result)
        self.assertEqual(runner.native_propose_args, args)

    def test_second_depth_sibling_commits_actual_path_not_flat_prefix(self):
        runner = self.make_runner()
        context = self.prepare(runner)
        commit = Mock()
        context.register_commit("cache", commit)
        result = runner._sample(self.logits_for([7, 10, 13, 14, 11]), object())
        self.assertEqual(result.sampled_token_ids.tolist(), [[7, 10, 11, -1, -1]])
        self.assertEqual(context.accepted_input_indices, (0, 1, 4))
        commit.assert_called_once_with((0, 1, 4))
        self.assertIsNone(result.logprobs_tensors)
        # Scheduler's existing count correction must equal the compact prefix.
        emitted_count = len(context.emitted_token_ids)
        optimistic_computed = context.prefix_length + context.num_nodes
        rejected = context.tree.num_candidates - (emitted_count - 1)
        self.assertEqual(optimistic_computed - rejected, context.prefix_length + emitted_count)

    def test_root_only_step_still_commits_one_state_and_emits_one_token(self):
        runner = self.make_runner(drafts=())
        context = self.prepare(runner)
        callback = Mock()
        context.register_commit("cache", callback)
        result = runner._sample(self.logits_for([6]), None)
        self.assertEqual(result.sampled_token_ids.tolist(), [[6]])
        callback.assert_called_once_with((0,))
        self.assertTrue(context.committed)

    def test_random_target_draws_not_argmax_determine_path(self):
        runner = self.make_runner(temperature=1.0, top_k=50, top_p=0.9, seed=42)
        context = self.prepare(runner)
        callback = Mock()
        context.register_commit("cache", callback)
        runner.sampler.sample_tree.side_effect = None
        runner.sampler.sample_tree.return_value = torch.tensor([7, 10, 13, 14, 11])
        logits = self.logits_for([8, 9, 12, 13, 14])
        output = runner._sample(logits, object())
        self.assertEqual(output.sampled_token_ids.tolist(), [[7, 10, 11, -1, -1]])
        callback.assert_called_once_with((0, 1, 4))
        runner.sampler.sample_tree.assert_called_once_with(
            logits, runner.input_batch.sampling_metadata,
            top_k=runner.requests["request"].sampling_params.top_k,
        )

    def test_random_draw_outside_children_is_emitted_without_resampling(self):
        runner = self.make_runner(temperature=1.0)
        context = self.prepare(runner)
        runner.sampler.sample_tree.side_effect = None
        runner.sampler.sample_tree.return_value = torch.tensor([15, 10, 13, 14, 11])
        output = runner._sample(self.logits_for([7, 10, 13, 14, 11]), None)
        self.assertEqual(output.sampled_token_ids.tolist(), [[15, -1, -1, -1, -1]])
        self.assertEqual(context.accepted_input_indices, (0,))

    def test_output_budget_excludes_matching_but_unprocessed_child(self):
        runner = self.make_runner(max_tokens=2)
        context = self.prepare(runner)
        result = runner._sample(self.logits_for([7, 10, 13, 14, 11]), object())
        self.assertEqual(result.sampled_token_ids.tolist(), [[7, 10, -1, -1, -1]])
        self.assertEqual(context.accepted_input_indices, (0, 1))

    def test_wrong_logits_count_and_uncommitted_proposal_fail(self):
        runner = self.make_runner()
        self.prepare(runner)
        with self.assertRaisesRegex(RuntimeError, "one row per node"):
            runner._sample(torch.zeros((4, 16)), None)
        with self.assertRaisesRegex(RuntimeError, "committed"):
            runner.propose_draft_token_ids(None, None, None, None, None, None, 5, torch.zeros((5, 3)))

    def test_prepare_rejects_batch_multimodal_and_invalid_node_count(self):
        runner = self.make_runner()
        runner.input_batch.num_reqs = 2
        with self.assertRaises(ValueError):
            self.prepare(runner)
        runner = self.make_runner()
        runner.requests["request"].mm_features = [object()]
        with self.assertRaises(ValueError):
            self.prepare(runner)
        runner = self.make_runner()
        with self.assertRaises(ValueError):
            runner._prepare_tree_step(runner.scheduler_output, 4)
        runner = self.make_runner(drafts=(7, 8, 9, 10, 11))
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.prepare(runner)

    def test_new_request_does_not_reuse_gdn_indices(self):
        runner = self.make_runner()
        runner._tree_gdn_request_id = "previous-request"
        runner._tree_gdn_indices["layer"] = [9, 10]
        context = self.prepare(runner)
        self.assertEqual(context.gdn_state_indices, {})
        self.assertIs(context.gdn_state_indices, runner._tree_gdn_indices)

    def test_drafter_reconstructs_accepted_path_in_canonical_slots(self):
        runner = self.make_runner()
        context = self.prepare(runner)
        runner._sample(self.logits_for([7, 10, 13, 14, 11]), object())
        hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        original_common = SimpleNamespace(
            slot_mapping=torch.tensor([112, 113, 114, 115, 116]),
            query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([17], dtype=torch.int32),
            num_actual_tokens=5,
        )
        metadata = SimpleNamespace(all_greedy=True)
        result = runner.propose_draft_token_ids(
            [[7, 10, 11]], metadata, runner.scheduler_output, object(), original_common,
            runner.positions, 5, hidden,
        )
        self.assertEqual(result.tolist(), [[1, 2, 3, 4]])
        kwargs = runner.drafter._propose.call_args.kwargs
        self.assertEqual(kwargs["target_token_ids"].tolist(), [5, 7, 10])
        self.assertTrue(torch.equal(kwargs["target_hidden_states"], hidden[[0, 1, 4]]))
        self.assertEqual(kwargs["target_positions"].tolist(), [12, 13, 14])
        self.assertEqual(kwargs["next_token_ids"].tolist(), [11])
        self.assertEqual(kwargs["token_indices_to_sample"].tolist(), [2])
        self.assertEqual(kwargs["num_speculative_tokens"], 4)
        self.assertEqual(kwargs["num_scheduled_tokens"], 3)
        self.assertEqual(kwargs["req_scheduled_tokens"], {"request": 3})
        self.assertIs(kwargs["scheduler_output"], runner.scheduler_output)
        self.assertIs(kwargs["sampling_metadata"], metadata)
        common = kwargs["common_attn_metadata"]
        self.assertIsNot(common, original_common)
        self.assertEqual(common.slot_mapping.tolist(), [112, 113, 114])
        self.assertEqual(common.query_start_loc_cpu.tolist(), [0, 3])
        self.assertEqual(common.query_start_loc.tolist(), [0, 3])
        self.assertEqual(common.seq_lens.tolist(), [15])
        self.assertEqual(common.seq_lens_cpu.tolist(), [15])
        self.assertEqual(common._seq_lens_cpu.tolist(), [15])
        self.assertEqual(common.seq_lens_cpu_upper_bound.tolist(), [15])
        self.assertEqual(common.num_computed_tokens_cpu.tolist(), [12])
        self.assertEqual(common._num_computed_tokens_cpu.tolist(), [12])
        self.assertEqual(common.positions.tolist(), [12, 13, 14])
        self.assertEqual(common.actual_seq_lengths_q, [3])
        self.assertEqual((common.num_actual_tokens, common.num_input_tokens, common.max_query_len), (3, 3, 3))
        self.assertEqual((common.max_seq_len, common.decode_token_per_req), (15, 3))
        self.assertEqual(original_common.query_start_loc_cpu.tolist(), [0, 5])
        self.assertEqual(original_common.seq_lens_cpu.tolist(), [17])
        self.assertEqual(original_common.slot_mapping.tolist(), [112, 113, 114, 115, 116])
        copied_ids, copied_counts = runner._copy_valid_sampled_token_count.call_args.args
        self.assertEqual(copied_ids.tolist(), [11])
        self.assertEqual(copied_counts.tolist(), [3])
        self.assertEqual(context.accepted_input_indices, (0, 1, 4))


if __name__ == "__main__":
    unittest.main()
