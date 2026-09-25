from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "scripts"
    / "benchmark_gsm8k_dflash.py"
)
SPEC = spec_from_file_location("benchmark_gsm8k_dflash", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gsm8k_answer_extractors():
    assert MODULE._gold_answer("work\n#### 1,234") == "1234"
    assert MODULE._predicted_answer("Therefore: \\boxed{1,234}.") == "1234"
    assert MODULE._predicted_answer("work\n#### -2.50") == "-2.5"
    assert MODULE._predicted_answer("first 2, final 17") == "17"
    assert MODULE._predicted_answer("no numeric answer") is None


def test_gsm8k_prompt_has_fewshot_examples():
    row = {"question": "target?", "answer": "#### 3"}
    examples = [
        {"question": "one?", "answer": "reason\n#### 1"},
        {"question": "two?", "answer": "reason\n#### 2"},
    ]
    assert MODULE._build_prompt(row, examples) == (
        "Question: one?\nAnswer: reason\n#### 1\n\n"
        "Question: two?\nAnswer: reason\n#### 2\n\n"
        "Question: target?\nAnswer:"
    )
