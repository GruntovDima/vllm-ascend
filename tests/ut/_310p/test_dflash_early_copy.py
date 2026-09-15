"""Test the actual early-output guards and ordering without initializing NPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class EarlyCopyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / 'vllm_ascend/worker/model_runner_v1.py'
        tree = ast.parse(path.read_text())
        self.method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                           and n.name == 'sample_tokens')
        self.guard = next(n for n in self.method.body if isinstance(n, ast.If)
                          and 'VLLM_ASCEND_DFLASH_EARLY_PREFILL_COPY' in ast.unparse(n.test))

    def test_copy_after_bookkeeping_before_draft(self):
        early_index = self.method.body.index(self.guard)
        before = self.method.body[:early_index]
        self.assertTrue(any(isinstance(n, ast.Assign) and '_bookkeeping_sync(' in ast.unparse(n)
                            for n in before))
        after = self.method.body[early_index + 1]
        self.assertIsInstance(after, ast.With)
        self.assertIn('propose_draft_token_ids(', ast.unparse(after))

    def test_guards_and_private_snapshot(self):
        code = compile(ast.Module(body=[self.guard], type_ignores=[]), '<actual-guard>', 'exec')
        changes = ({}, {'enabled': False}, {'async_mode': False}, {'world_size': 2},
                   {'method': 'mtp'}, {'metadata': object()}, {'requests': 2},
                   {'discarded': 1}, {'experts': True}, {'width': 16}, {'logprobs': object()})
        for change in changes:
            cfg = dict(enabled=True, async_mode=True, world_size=1, method='dflash',
                       metadata=None, requests=1, discarded=0, experts=False, width=1, logprobs=None)
            cfg.update(change)
            cloned = object()
            ids = SimpleNamespace(shape=(1, cfg['width']), clone=Mock(return_value=cloned))
            constructor = Mock(return_value='early-output')
            runner = SimpleNamespace(use_async_scheduling=cfg['async_mode'],
                                     speculative_config=SimpleNamespace(method=cfg['method']),
                                     input_batch=SimpleNamespace(num_reqs=cfg['requests'], vocab_size=248320),
                                     num_discarded_requests=cfg['discarded'],
                                     routed_experts_initialized=cfg['experts'], async_output_copy_stream=object())
            scope = dict(self=runner, ascend_envs=SimpleNamespace(VLLM_ASCEND_DFLASH_EARLY_PREFILL_COPY=cfg['enabled']),
                         pp=SimpleNamespace(world_size=cfg['world_size']), spec_decode_metadata=cfg['metadata'],
                         sampler_output=SimpleNamespace(sampled_token_ids=ids, logprobs_tensors=cfg['logprobs']),
                         model_runner_output=object(), invalid_req_indices=[],
                         AsyncGPUModelRunnerOutput=constructor, async_output=None)
            exec(code, scope)
            self.assertEqual(constructor.call_count, int(not change))
            self.assertEqual(ids.clone.call_count, int(not change))
            if not change:
                self.assertIs(constructor.call_args.kwargs['sampled_token_ids'], cloned)
                self.assertEqual(runner.early_prefill_copy_count, 1)
            else:
                self.assertIsNone(scope['async_output'])


if __name__ == '__main__':
    unittest.main()
