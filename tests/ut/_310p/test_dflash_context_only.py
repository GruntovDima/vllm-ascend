"""Host guards for cache-only incomplete prefill; no NPU initialization."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class ContextOnlyTests(unittest.TestCase):
    def test_context_commit_precedes_early_return(self):
        path = Path(__file__).resolve().parents[3] / 'vllm_ascend/spec_decode/llm_base_proposer.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'AscendSpecDecodeBaseProposer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_run_merged_draft')
        scope = {'torch': SimpleNamespace(Tensor=object, int64='int64',
                                         zeros=Mock(return_value='unused proposals'))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        obj = SimpleNamespace(method='dflash', _context_only_prefill=True, input_ids=[],
                              _get_positions=Mock(return_value=None),
                              build_model_inputs_first_pass=Mock(return_value={}),
                              skip_query_for_incomplete_prefill=Mock(return_value=True),
                              context_only_prefill_count=0, num_speculative_tokens=15,
                              device='npu', model=Mock(side_effect=AssertionError('query must be skipped')))
        result = scope['_run_merged_draft'](obj, 16, 1, None, None, None, None, 16)
        self.assertEqual(result, 'unused proposals')
        obj.build_model_inputs_first_pass.assert_called_once_with(16)
        obj.model.assert_not_called()
        self.assertEqual(obj.context_only_prefill_count, 1)

    def make_proposer(self):
        path = Path(__file__).resolve().parents[3] / 'vllm_ascend/spec_decode/dflash_proposer.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'skip_query_for_incomplete_prefill')

        class Parent:
            def _run_merged_draft(self, *args, **kwargs):
                return 'full draft'

        ctx = SimpleNamespace(cudagraph_runtime_mode=0)
        extra = SimpleNamespace(capturing=False)
        zeros = Mock(return_value='unused proposals')
        scope = {'Parent': Parent, 'get_forward_context': lambda: ctx,
                 '_EXTRA_CTX': extra, 'CUDAGraphMode': SimpleNamespace(NONE=0),
                 'torch': SimpleNamespace(zeros=zeros, int64='int64')}
        cls = ast.ClassDef(name='Proposer', bases=[ast.Name(id='Parent', ctx=ast.Load())],
                           keywords=[], body=[method], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        exec(compile(module, str(path), 'exec'), scope)
        obj = scope['Proposer']()
        obj._context_only_prefill = True
        obj.runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=1), num_discarded_requests=1)
        obj.context_only_prefill_count = 0
        obj.num_speculative_tokens = 15
        obj.device = 'npu'
        obj.build_model_inputs_first_pass = Mock()
        return obj, ctx, extra, zeros

    def test_only_incomplete_bs1_eager_skips(self):
        for case in ('eligible', 'disabled', 'complete', 'mixed', 'graph', 'capture'):
            obj, ctx, extra, zeros = self.make_proposer()
            if case == 'disabled':
                obj._context_only_prefill = False
            elif case == 'complete':
                obj.runner.num_discarded_requests = 0
            elif case == 'mixed':
                obj.runner.input_batch.num_reqs = 2
            elif case == 'graph':
                ctx.cudagraph_runtime_mode = 1
            elif case == 'capture':
                extra.capturing = True
            result = obj.skip_query_for_incomplete_prefill(1)
            if case == 'eligible':
                self.assertTrue(result)
            else:
                self.assertFalse(result)
            obj.build_model_inputs_first_pass.assert_not_called()
            zeros.assert_not_called()


if __name__ == '__main__':
    unittest.main()
