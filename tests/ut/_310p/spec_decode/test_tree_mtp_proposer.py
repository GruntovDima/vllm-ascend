# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU tensor tests with an isolated stub for the native MTP backbone.

Run directly with Python to avoid needing an installed vLLM/NPU plugin. The
numerical selection uses real torch; the stub records backbone iterations and
calls the production sampling override, without loading a model.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def load_proposer_module():
    class NativeProposer:
        def _propose(self, num_speculative_tokens, *args, **kwargs):
            self.native_calls.append((num_speculative_tokens, args, kwargs))
            self.num_speculative_tokens = num_speculative_tokens
            if hasattr(self, "native_result"):
                return self.native_result
            if num_speculative_tokens == 0:
                return torch.empty((1, 0), dtype=torch.long)
            result = []
            for index in range(num_speculative_tokens):
                if getattr(self, "fail_at", None) == index:
                    raise RuntimeError("native failure")
                if getattr(self, "bypass_sampling", False):
                    result.append(self.test_logits[index].argmax(dim=-1))
                    continue
                token_ids, _ = self._sample_draft_from_logits(
                    self.test_logits[index], kwargs.get("sampling_metadata")
                )
                result.append(token_ids)
            self.backbone_ids = torch.stack(result, dim=1)
            return self.backbone_ids

        def _sample_draft_from_logits(self, logits, sampling_metadata):
            self.native_sampling_calls += 1
            return logits.argmax(dim=-1), self.native_probs

        def _run_merged_draft(self, *args, **kwargs):
            raise NotImplementedError

    class Rotary:
        @staticmethod
        def set_rope_position_flag_310p(enabled):
            pass

    names = (
        "vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.utils", "vllm.v1.sample", "vllm.v1.sample.metadata",
        "vllm_ascend", "vllm_ascend._310p", "vllm_ascend._310p.ops",
        "vllm_ascend._310p.ops.rotary_embedding", "vllm_ascend.spec_decode",
        "vllm_ascend.spec_decode.llm_base_proposer",
    )
    stubs = {name: types.ModuleType(name) for name in names}
    for module in stubs.values():
        module.__path__ = []
    stubs["vllm.v1.attention.backends.utils"].CommonAttentionMetadata = object
    stubs["vllm.v1.sample.metadata"].SamplingMetadata = object
    stubs["vllm_ascend._310p.ops.rotary_embedding"].AscendRotaryEmbedding310 = Rotary
    stubs["vllm_ascend.spec_decode.llm_base_proposer"].AscendSpecDecodeBaseProposer = NativeProposer
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py"
    spec = importlib.util.spec_from_file_location("_standalone_tree_mtp_proposer", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module, NativeProposer


class TestTreeMTPProposer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.native_type = load_proposer_module()

    def make_proposer(self, width=2, depth=2, copied_methods=False):
        proposer_type = self.module.AscendSpecDecodeBaseProposer310
        if copied_methods:
            # Production installs 310P methods on the native base class; it
            # does not instantiate the 310P subclass. Test this binding too.
            proposer_type = type("PatchedNative", (self.native_type,), {
                "_propose": proposer_type._propose,
                "_sample_draft_from_logits": proposer_type._sample_draft_from_logits,
            })
        proposer = object.__new__(proposer_type)
        proposer.tree_mtp_config = SimpleNamespace(width=width, depth=depth)
        proposer.num_speculative_tokens = width * depth
        proposer.native_calls = []
        proposer.native_sampling_calls = 0
        proposer.native_probs = object()
        proposer.test_logits = [torch.tensor([[1.0, 5.0, 5.0, 0.0]]), torch.tensor([[7.0, 3.0, 7.0, 0.0]])]
        return proposer

    def test_comb_follows_only_primary_backbone(self):
        proposer = self.make_proposer()
        result = proposer._propose(4, sampling_metadata=SimpleNamespace(all_greedy=True))
        self.assertEqual(result.tolist(), [[1, 2, 0, 2]])
        self.assertEqual(proposer.backbone_ids.tolist(), [[1, 0]])
        self.assertEqual(proposer.native_calls[0][0], 2)
        self.assertEqual(proposer.num_speculative_tokens, 4)
        self.assertIsNone(proposer._tree_mtp_sibling_ids)
        self.assertIsNone(proposer._last_draft_probs)
        self.assertEqual(proposer.native_sampling_calls, 0)

    def test_base_instance_method_patching(self):
        proposer = self.make_proposer(copied_methods=True)
        self.assertEqual(proposer._propose(4).tolist(), [[1, 2, 0, 2]])

    def test_width_one_is_linear(self):
        proposer = self.make_proposer(width=1)
        self.assertEqual(proposer._propose(2).tolist(), [[1, 0]])

    def test_requested_sweep_preserves_primary_backbone_and_all_siblings(self):
        generator = torch.Generator().manual_seed(310)
        for width in (1, 2, 4, 8, 16):
            for depth in range(1, 5):
                with self.subTest(width=width, depth=depth):
                    proposer = self.make_proposer(width, depth, copied_methods=True)
                    proposer.test_logits = [torch.randn((1, 64), generator=generator) for _ in range(depth)]
                    expected = torch.stack([
                        torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :width]
                        for logits in proposer.test_logits
                    ], dim=1)
                    actual = proposer._propose(width * depth)
                    self.assertTrue(torch.equal(actual, expected.flatten(1)))
                    self.assertTrue(torch.equal(proposer.backbone_ids, expected[:, :, 0]))
                    self.assertEqual(proposer.num_speculative_tokens, width * depth)

    def test_disabled_propose_preserves_arguments_and_result(self):
        proposer = self.make_proposer()
        del proposer.tree_mtp_config
        proposer.native_result = torch.tensor([[7, 8, 9]])
        sentinel = object()
        result = proposer._propose(3, sentinel, another=sentinel)
        self.assertIs(result, proposer.native_result)
        self.assertEqual(proposer.native_calls, [(3, (sentinel,), {"another": sentinel})])

    def test_disabled_sampling_uses_native_sampler(self):
        proposer = self.make_proposer()
        result, probs = proposer._sample_draft_from_logits(proposer.test_logits[0], None)
        self.assertEqual(result.tolist(), [1])
        self.assertIs(probs, proposer.native_probs)
        self.assertEqual(proposer.native_sampling_calls, 1)

    def test_zero_budget_is_native_empty(self):
        proposer = self.make_proposer()
        self.assertEqual(tuple(proposer._propose(0).shape), (1, 0))
        self.assertEqual(proposer.native_calls[0][0], 0)

    def test_exception_cleans_transient_state(self):
        proposer = self.make_proposer()
        proposer.fail_at = 1
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            proposer._propose(4)
        self.assertIsNone(proposer._tree_mtp_sibling_ids)
        self.assertIsNone(proposer._last_draft_probs)
        self.assertEqual(proposer.num_speculative_tokens, 4)
        proposer.fail_at = None
        self.assertEqual(proposer._propose(4).tolist(), [[1, 2, 0, 2]])

    def test_missing_logits_fails_instead_of_returning_linear_tokens(self):
        proposer = self.make_proposer()
        proposer.bypass_sampling = True
        with self.assertRaisesRegex(RuntimeError, "full-logits"):
            proposer._propose(4)

    def test_random_target_sampling_keeps_deterministic_drafting_and_rng(self):
        proposer = self.make_proposer()
        generator = torch.Generator().manual_seed(42)
        before = generator.get_state().clone()
        metadata = SimpleNamespace(all_greedy=False, generators={0: generator}, temperature=1.0)
        result = proposer._propose(4, sampling_metadata=metadata)
        self.assertEqual(result.tolist(), [[1, 2, 0, 2]])
        self.assertTrue(torch.equal(generator.get_state(), before))

    def test_probabilistic_drafter_is_still_rejected(self):
        proposer = self.make_proposer()
        proposer._enable_probabilistic_draft_probs = True
        with self.assertRaisesRegex(ValueError, "greedy"):
            proposer._propose(4)

    def test_bad_budget_and_config_rejected(self):
        proposer = self.make_proposer()
        with self.assertRaisesRegex(ValueError, "budget"):
            proposer._propose(3)
        for width, depth in ((0, 2), (2, -1), (True, 2), (2, 1.5)):
            proposer.tree_mtp_config = SimpleNamespace(width=width, depth=depth)
            with self.assertRaisesRegex(ValueError, "positive integers"):
                proposer._propose(4)

    def test_ties_and_negative_infinity_have_distinct_stable_ids(self):
        logits = torch.tensor([[2.0, 3.0, 3.0, 3.0], [-torch.inf, -torch.inf, -torch.inf, -torch.inf]])
        original = logits.clone()
        expected = [[1, 2, 3, 0], [0, 1, 2, 3]]
        for _ in range(3):
            self.assertEqual(self.module._tree_topk_token_ids(logits, 4).tolist(), expected)
        self.assertTrue(torch.equal(logits, original))

    def test_nan_and_positive_infinity_preserve_greedy_first(self):
        logits = torch.tensor([[1.0, torch.nan, torch.nan, torch.inf], [torch.inf, 1.0, torch.inf, -torch.inf]])
        result = self.module._tree_topk_token_ids(logits, 4)
        self.assertEqual(result.tolist(), [[1, 2, 3, 0], [0, 2, 1, 3]])
        self.assertTrue(torch.equal(result[:, 0], logits.argmax(dim=-1)))

    def test_invalid_logits_or_width(self):
        for logits, width in ((torch.ones(4), 2), (torch.ones(1, 4, dtype=torch.long), 2),
                              (torch.ones(1, 4), 5), (torch.ones(1, 4), 0)):
            with self.assertRaises(ValueError):
                self.module._tree_topk_token_ids(logits, width)

    def test_boundary_ties_and_noncontiguous_inputs(self):
        rng = torch.Generator().manual_seed(310)
        for dtype in (torch.float16, torch.float32, torch.float64):
            logits = torch.randint(-4, 5, (3, 256), generator=rng).to(dtype)[:, ::2]
            before = logits.clone()
            for width in (2, 4, 8, 16, 128):
                expected = logits.argsort(dim=-1, descending=True, stable=True)[:, :width]
                actual = self.module._tree_topk_token_ids(logits, width)
                self.assertTrue(torch.equal(actual, expected))
            torch.testing.assert_close(logits, before)

    def test_adjacent_fp32_scores_are_not_perturbed_for_tie_breaking(self):
        # A numerical epsilon applied to logits could reorder these scores.
        logits = torch.arange(0x3F800000, 0x3F800080, dtype=torch.int32).view(torch.float32).flip(0)[None]
        for width in (2, 4, 8, 16):
            actual = self.module._tree_topk_token_ids(logits, width)
            self.assertEqual(actual.tolist(), [list(range(width))])

    def test_nan_payloads_and_signed_zeros(self):
        bits = torch.tensor([[0x7F800001, 0x7FC00000, 0x7F800000, 0, -2147483648, -8388608]], dtype=torch.int32)
        logits = bits.view(torch.float32)
        expected = logits.argsort(dim=-1, descending=True, stable=True)
        self.assertTrue(torch.equal(self.module._tree_topk_token_ids(logits, 6), expected))


if __name__ == "__main__":
    unittest.main()
