"""CPU isolation test for the real proposer method, without worker startup."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def test_pruned_dflash_routes_through_canonical_vocab_expansion():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute_draft_token_ids")
    module = ast.Module(body=[method], type_ignores=[])
    scope = {"torch": SimpleNamespace(Tensor=object), "greedy_sample": lambda logits: logits}
    exec(compile(module, str(source), "exec"), scope)
    model = SimpleNamespace(_ascend_lm_head_pruned=True, compute_logits=Mock(return_value="canonical"),
                            logits_processor=Mock(side_effect=AssertionError("compact IDs must not escape")))
    result = scope["compute_draft_token_ids"](SimpleNamespace(method="dflash", model=model), "hidden")
    assert result == "canonical"
    model.compute_logits.assert_called_once_with("hidden")
    model.logits_processor.assert_not_called()


def test_compact_dflash_returns_canonical_ids_without_expansion():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute_draft_token_ids")
    scope = {"torch": SimpleNamespace(Tensor=object)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    model = SimpleNamespace(compute_pruned_greedy_ids=Mock(return_value="canonical IDs"),
                            compute_logits=Mock(side_effect=AssertionError("must not expand")))
    result = scope["compute_draft_token_ids"](SimpleNamespace(method="dflash", model=model), "hidden")
    assert result == "canonical IDs"
    model.compute_pruned_greedy_ids.assert_called_once_with("hidden")
    model.compute_logits.assert_not_called()
