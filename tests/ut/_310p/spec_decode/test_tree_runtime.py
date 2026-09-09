# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Dependency-free tests of tree activation, sampling guards and cache commits."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def load_tree_modules():
    directory = Path(__file__).resolve().parents[4] / "vllm_ascend/_310p/spec_decode"
    stubs = {}
    for name in ("vllm_ascend", "vllm_ascend._310p", "vllm_ascend._310p.spec_decode"):
        stubs[name] = types.ModuleType(name)
        stubs[name].__path__ = []
    loaded = []
    with patch.dict(sys.modules, stubs):
        for filename in ("tree", "tree_runtime"):
            name = f"vllm_ascend._310p.spec_decode.{filename}"
            spec = importlib.util.spec_from_file_location(name, directory / f"{filename}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
    return loaded


def make_vllm_config():
    return SimpleNamespace(
        additional_config={"tree_mtp": {"enabled": True}},
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4, draft_sample_method="greedy"),
        model_config=SimpleNamespace(enforce_eager=True, hf_text_config=SimpleNamespace(model_type="qwen3_5_text")),
        scheduler_config=SimpleNamespace(max_num_seqs=1, async_scheduling=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode="none"),
        lora_config=None,
        kv_transfer_config=None,
    )


def make_sampling_params(**changes):
    values = dict(
        temperature=0.0, n=1, repetition_penalty=1.0, presence_penalty=0.0,
        frequency_penalty=0.0, logprobs=None, prompt_logprobs=None,
        structured_outputs=None, allowed_token_ids=None, logit_bias=None,
        bad_words=None, logits_processors=None, min_tokens=0, max_tokens=64,
        top_k=-1, top_p=1.0, min_p=0.0, thinking_token_budget=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


class TestTreeMTPConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.runtime = load_tree_modules()

    def test_disabled_does_not_require_mtp_or_model_config(self):
        for additional in (None, {}, {"tree_mtp": {}}, {"tree_mtp": {"enabled": False}}):
            with self.subTest(additional=additional):
                config = SimpleNamespace(additional_config=additional)
                self.assertIsNone(self.runtime.TreeMTPConfig.from_vllm_config(config))

    def test_enabled_defaults_and_trace(self):
        config = make_vllm_config()
        config.additional_config["tree_mtp"]["trace"] = True
        actual = self.runtime.TreeMTPConfig.from_vllm_config(config)
        self.assertEqual((actual.width, actual.depth, actual.num_candidates, actual.trace), (2, 2, 4, True))

    def test_invalid_activation_object(self):
        for raw in (True, [], "enabled", {"enabled": 1}, {"enabled": "false"}, {"unexpected": 2}):
            with self.subTest(raw=raw):
                config = make_vllm_config()
                config.additional_config["tree_mtp"] = raw
                with self.assertRaises(ValueError):
                    self.runtime.TreeMTPConfig.from_vllm_config(config)

    def test_only_validated_topology(self):
        for field, value in (("width", True), ("depth", 2.0), ("trace", 1), ("width", 3), ("depth", 5),
                             ("width", 0), ("width", 32), ("depth", 0)):
            with self.subTest(field=field, value=value):
                config = make_vllm_config()
                config.additional_config["tree_mtp"][field] = value
                with self.assertRaises(ValueError):
                    self.runtime.TreeMTPConfig.from_vllm_config(config)

    def test_requested_width_depth_grid(self):
        for width in (1, 2, 4, 8, 16):
            for depth in range(1, 5):
                with self.subTest(width=width, depth=depth):
                    config = make_vllm_config()
                    config.additional_config["tree_mtp"].update(width=width, depth=depth)
                    config.speculative_config.num_speculative_tokens = width * depth
                    actual = self.runtime.TreeMTPConfig.from_vllm_config(config)
                    self.assertEqual((actual.width, actual.depth, actual.num_candidates),
                                     (width, depth, width * depth))

    def test_execution_guards(self):
        cases = (
            ("speculative_config", "method", "eagle"),
            ("speculative_config", "num_speculative_tokens", 2),
            ("speculative_config", "draft_sample_method", "probabilistic"),
            ("model_config", "enforce_eager", False),
            ("model_config", "logits_processors", [object()]),
            ("scheduler_config", "max_num_seqs", 2),
            ("scheduler_config", "async_scheduling", True),
            ("parallel_config", "tensor_parallel_size", 2),
            ("parallel_config", "pipeline_parallel_size", 2),
            ("parallel_config", "data_parallel_size", 2),
            ("parallel_config", "decode_context_parallel_size", 2),
            ("cache_config", "enable_prefix_caching", True),
            ("cache_config", "mamba_cache_mode", "all"),
        )
        for section, field, value in cases:
            with self.subTest(section=section, field=field):
                config = make_vllm_config()
                setattr(getattr(config, section), field, value)
                with self.assertRaises(ValueError):
                    self.runtime.TreeMTPConfig.from_vllm_config(config)
        for field, value in (("lora_config", object()), ("kv_transfer_config", object()),
                             ("speculative_config", None)):
            with self.subTest(field=field):
                config = make_vllm_config()
                setattr(config, field, value)
                with self.assertRaises(ValueError):
                    self.runtime.TreeMTPConfig.from_vllm_config(config)
        config = make_vllm_config()
        config.model_config.hf_text_config.model_type = "qwen3_5_moe_text"
        with self.assertRaises(ValueError):
            self.runtime.TreeMTPConfig.from_vllm_config(config)


class TestTreeSamplingGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.runtime = load_tree_modules()

    def test_neutral_greedy_params(self):
        self.runtime.validate_tree_sampling(make_sampling_params())

    def test_temperature_top_k_top_p_and_seed_are_supported(self):
        for seed in (None, 0, 42):
            self.runtime.validate_tree_sampling(make_sampling_params(
                temperature=1.0, top_k=50, top_p=0.9, seed=seed,
            ))

    def test_all_unsupported_processors_rejected(self):
        cases = dict(
            n=2, repetition_penalty=1.1, presence_penalty=0.1,
            frequency_penalty=-0.1, logprobs=1, prompt_logprobs=1,
            structured_outputs=object(), allowed_token_ids=[1], logit_bias={1: 0.5},
            bad_words=["bad"], logits_processors=[object()], min_tokens=1,
            min_p=0.1, thinking_token_budget=0, logprob_token_ids=[1],
        )
        for field, value in cases.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.runtime.validate_tree_sampling(make_sampling_params(**{field: value}))

    def test_zero_logprobs_still_requests_logprob_output(self):
        for field in ("logprobs", "prompt_logprobs"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.runtime.validate_tree_sampling(make_sampling_params(**{field: 0}))


class TestTreeStepContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree_module, cls.runtime = load_tree_modules()

    def make_context(self):
        tree = self.tree_module.build_comb_tree(5, ((7, 8), (9, 10)))
        return self.runtime.TreeStepContext(tree, prefix_length=12)

    def test_all_cache_writers_get_actual_path_exactly_once(self):
        context = self.make_context()
        attention, gdn = Mock(), Mock()
        context.register_commit("attention", attention)
        context.register_commit("gdn", gdn)
        self.assertTrue(context.has_commit("attention"))
        result = self.tree_module.TreeVerification((7, 10, 11), (0, 1, 4))
        context.commit(result)
        attention.assert_called_once_with((0, 1, 4))
        gdn.assert_called_once_with((0, 1, 4))
        self.assertEqual(context.accepted_input_indices, (0, 1, 4))
        self.assertEqual(context.emitted_token_ids, (7, 10, 11))
        self.assertTrue(context.committed)
        self.assertFalse(context.has_commit("attention"))
        with self.assertRaises(RuntimeError):
            context.commit(result)
        with self.assertRaises(RuntimeError):
            context.register_commit("later", Mock())

    def test_invalid_paths_never_write_any_cache(self):
        for path in ((), (1,), (0, 4), (0, 1, 2), (0, 2, 4), (0, 1, 1), (0, 5)):
            with self.subTest(path=path):
                context = self.make_context()
                callback = Mock()
                context.register_commit("cache", callback)
                result = self.tree_module.TreeVerification((1,) * len(path), path)
                with self.assertRaises(ValueError):
                    context.commit(result)
                callback.assert_not_called()
                self.assertFalse(context.committed)

    def test_length_mismatch_and_duplicate_writer_rejected(self):
        context = self.make_context()
        callback = Mock()
        context.register_commit("cache", callback)
        with self.assertRaises(RuntimeError):
            context.register_commit("cache", Mock())
        with self.assertRaises(ValueError):
            context.commit(self.tree_module.TreeVerification((7,), (0, 1)))
        callback.assert_not_called()

    def test_root_only_commit_and_context_isolation(self):
        first, second = self.make_context(), self.make_context()
        callback = Mock()
        first.register_commit("cache", callback)
        first.gdn_state_indices["layer"] = [1, 2]
        first.commit(self.tree_module.TreeVerification((6,), (0,)))
        callback.assert_called_once_with((0,))
        self.assertFalse(second.committed)
        self.assertEqual(second.gdn_state_indices, {})
        self.assertFalse(second.has_commit("cache"))


if __name__ == "__main__":
    unittest.main()
