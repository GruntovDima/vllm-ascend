# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exact finite-distribution proof and CPU tests of the production RNG adapter."""

import ast
import importlib.util
import itertools
import math
import unittest
from collections import defaultdict
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def load_tree():
    spec = importlib.util.spec_from_file_location(
        "_stochastic_tree_helpers", Path(__file__).with_name("test_tree_runtime.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_tree_modules()[0]


def load_sampler_adapter():
    """Extract real methods; stub only the native sampler's forward call."""
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/_310p/sample/sampler.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    helpers = {"_prepare_cpu_generators_310p", "_generate_request_uniforms_310p"}
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in helpers]
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "AscendSampler310")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "sample_tree")

    class NativeSampler:
        def __init__(self):
            self.topk_topp_sampler = SimpleNamespace(tree_top_k=None)

        @staticmethod
        def greedy_sample(logits):
            return logits.argmax(-1)

        def forward(self, logits, metadata):
            self.seen_metadata = metadata
            self.seen_top_k = self.topk_topp_sampler.tree_top_k
            if getattr(self, "fail", False):
                raise RuntimeError("native sampler failure")
            # Return actual generated uniforms for a precise RNG sequence test.
            values = namespace["_generate_request_uniforms_310p"](
                logits.shape[0], metadata.generators, logits.device,
            )
            return SimpleNamespace(sampled_token_ids=values[:, None])

    namespace = dict(torch=torch, copy=copy, NativeSampler=NativeSampler, _CPU_GENERATOR_CACHE_310P={})
    cls = ast.ClassDef(name="TreeSampler", bases=[ast.Name(id="NativeSampler", ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[], type_params=[])
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes, cls,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


class TestTreeSamplingRNG(unittest.TestCase):
    def setUp(self):
        self.ns = load_sampler_adapter()
        self.sampler = self.ns["TreeSampler"]()
        self.real_rand = torch.rand

        def without_pinning(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return self.real_rand(*args, **kwargs)

        self.pinning_patch = patch.object(torch, "rand", side_effect=without_pinning)
        self.pinning_patch.start()
        self.addCleanup(self.pinning_patch.stop)

    @staticmethod
    def metadata(seed=42, **changes):
        values = dict(
            all_greedy=False, all_random=True, temperature=torch.tensor([1.0]),
            top_k=torch.tensor([50]), top_p=torch.tensor([0.9]),
            generators={} if seed is None else {0: torch.Generator().manual_seed(seed)},
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_metadata_broadcast_does_not_mutate_original(self):
        metadata = self.metadata()
        original_rng = metadata.generators
        self.sampler.sample_tree(torch.zeros(65, 97), metadata)
        expanded = self.sampler.seen_metadata
        self.assertIsNot(expanded, metadata)
        for name in ("temperature", "top_k", "top_p"):
            self.assertEqual(getattr(metadata, name).shape, (1,))
            self.assertEqual(getattr(expanded, name).shape, (65,))
        self.assertIs(metadata.generators, original_rng)
        self.assertEqual(set(expanded.generators), set(range(65)))
        self.assertTrue(all(rng is metadata.generators[0] for rng in expanded.generators.values()))
        self.assertEqual(set(self.ns["_CPU_GENERATOR_CACHE_310P"]), {0})

    def test_cpu_top_k_hint_is_scoped_even_on_failure(self):
        self.sampler.sample_tree(torch.zeros(5, 97), self.metadata(), top_k=50)
        self.assertEqual(self.sampler.seen_top_k, 50)
        self.assertIsNone(self.sampler.topk_topp_sampler.tree_top_k)
        self.sampler.fail = True
        with self.assertRaisesRegex(RuntimeError, "native sampler failure"):
            self.sampler.sample_tree(torch.zeros(5, 97), self.metadata(), top_k=50)
        self.assertIsNone(self.sampler.topk_topp_sampler.tree_top_k)

    def test_seeded_nodes_use_distinct_advancing_draws_even_without_prefill(self):
        metadata = self.metadata()
        observed = self.sampler.sample_tree(torch.zeros(65, 97), metadata)
        expected_rng = torch.Generator().manual_seed(42)
        expected = torch.stack([torch.rand((), generator=expected_rng) for _ in range(65)])
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(observed.unique().numel(), 65)
        following = self.sampler.sample_tree(torch.zeros(5, 97), metadata)
        expected = torch.stack([torch.rand((), generator=expected_rng) for _ in range(5)])
        self.assertTrue(torch.equal(following, expected))

    def test_prefill_tree_and_next_decode_share_one_rng_sequence(self):
        metadata = self.metadata(seed=31)
        generate = self.ns["_generate_request_uniforms_310p"]
        first = generate(1, metadata.generators, torch.device("cpu"))
        tree = self.sampler.sample_tree(torch.zeros(17, 97), metadata)
        last = generate(1, metadata.generators, torch.device("cpu"))
        expected_rng = torch.Generator().manual_seed(31)
        expected = torch.stack([torch.rand((), generator=expected_rng) for _ in range(19)])
        self.assertTrue(torch.equal(torch.cat((first, tree, last)), expected))

    def test_new_request_with_same_seed_restarts_without_leaking_previous_state(self):
        first = self.sampler.sample_tree(torch.zeros(17, 97), self.metadata())
        second = self.sampler.sample_tree(torch.zeros(17, 97), self.metadata())
        self.assertTrue(torch.equal(first, second))

    def test_unseeded_sampling_uses_default_rng_and_does_not_cache_requests(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(72)
            expected = torch.rand(17)
            torch.manual_seed(72)
            actual = self.sampler.sample_tree(torch.zeros(17, 97), self.metadata(seed=None))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(self.ns["_CPU_GENERATOR_CACHE_310P"], {})

    def test_greedy_does_not_consume_rng(self):
        metadata = self.metadata(all_greedy=True, all_random=False)
        before = torch.random.get_rng_state().clone()
        result = self.sampler.sample_tree(torch.tensor([[1.0, 3.0], [2.0, 1.0]]), metadata)
        self.assertEqual(result.tolist(), [1, 0])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(self.ns["_CPU_GENERATOR_CACHE_310P"], {})

    def test_root_only_and_disabled_filters(self):
        metadata = self.metadata(top_k=None, top_p=None)
        self.assertEqual(self.sampler.sample_tree(torch.zeros(1, 97), metadata).shape, (1,))
        self.assertIsNone(self.sampler.seen_metadata.top_k)
        self.assertIsNone(self.sampler.seen_metadata.top_p)

    def test_invalid_metadata_and_shape_fail(self):
        for logits in (torch.zeros(97), torch.zeros(0, 97)):
            with self.assertRaises(ValueError):
                self.sampler.sample_tree(logits, self.metadata())
        for changes in (dict(all_random=False), dict(top_k=torch.tensor([2, 3])),
                        dict(generators={1: torch.Generator()}),
                        dict(thinking_budget_state_holder=SimpleNamespace(has_tracked_requests=lambda: True))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.sampler.sample_tree(torch.zeros(5, 97), self.metadata(**changes))

    def test_failure_restores_request_indexed_rng_cache(self):
        self.sampler.fail = True
        with self.assertRaisesRegex(RuntimeError, "native sampler failure"):
            self.sampler.sample_tree(torch.zeros(65, 97), self.metadata())
        self.assertEqual(set(self.ns["_CPU_GENERATOR_CACHE_310P"]), {0})


class TestExactTargetDistribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree_module = load_tree()

    @staticmethod
    def probabilities(prefix):
        # History-dependent target distribution, independent of draft ranking.
        score = sum((index + 1) * (token + 1) for index, token in enumerate(prefix))
        weights = [1 + ((score + 3 * token + len(prefix)) % 7) for token in range(3)]
        return [weight / sum(weights) for weight in weights]

    def finish_ar(self, prefix, emitted, mass, horizon, distribution):
        if len(emitted) == horizon:
            distribution[tuple(emitted)] += mass
            return
        for token, probability in enumerate(self.probabilities(prefix)):
            self.finish_ar((*prefix, token), (*emitted, token), mass * probability, horizon, distribution)

    def test_every_three_token_output_has_exact_ar_probability(self):
        for candidates, horizon in (((), 3), (((0, 1), (2, 0)), 3), (((0,), (1,), (2,)), 3),
                                    (((0, 1), (2, 0)), 1)):
            with self.subTest(candidates=candidates, horizon=horizon):
                tree = self.tree_module.build_comb_tree(2, candidates)
                distributions = [self.probabilities(tuple(tree.token_ids[i] for i in tree.ancestor_indices(node)))
                                 for node in range(tree.num_nodes)]
                actual, expected = defaultdict(float), defaultdict(float)
                for draws in itertools.product(range(3), repeat=tree.num_nodes):
                    mass = math.prod(distributions[node][token] for node, token in enumerate(draws))
                    result = self.tree_module.verify_target_samples(tree, draws, max_output_tokens=horizon)
                    self.finish_ar((2, *result.emitted_token_ids), result.emitted_token_ids, mass, horizon, actual)
                self.finish_ar((2,), (), 1.0, horizon, expected)
                self.assertEqual(set(actual), set(expected))
                self.assertAlmostEqual(sum(actual.values()), 1.0, places=12)
                for sequence in expected:
                    self.assertAlmostEqual(actual[sequence], expected[sequence], places=12)

    def test_nonmatching_sample_is_not_replaced_by_an_available_candidate(self):
        tree = self.tree_module.build_comb_tree(2, ((0, 1), (2, 0)))
        result = self.tree_module.verify_target_samples(tree, [2, 0, 1, 2, 0])
        self.assertEqual(result.emitted_token_ids, (2,))
        self.assertEqual(result.accepted_input_indices, (0,))


if __name__ == "__main__":
    unittest.main()
