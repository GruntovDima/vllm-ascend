# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pure tree tests, runnable directly without importing torch or vLLM.

The package's normal __init__ loads the device plugin. Loading only tree.py
also tests that this intentionally backend-independent module needs no NPU.
"""

import importlib.util
import itertools
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


class TestTokenTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[4] / "vllm_ascend/_310p/spec_decode/tree.py"
        spec = importlib.util.spec_from_file_location("_standalone_tree_core", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.core = module

    def comb(self):
        return self.core.build_comb_tree(10, ((11, 12), (21, 22), (31, 32)))

    def test_comb_depths_differ_from_flat_indices(self):
        tree = self.comb()
        self.assertEqual(tree.token_ids, (10, 11, 12, 21, 22, 31, 32))
        self.assertEqual(tree.parents, (-1, 0, 0, 1, 1, 3, 3))
        self.assertEqual(tree.depths, (0, 1, 1, 2, 2, 3, 3))
        self.assertEqual(tree.children[2], ())
        self.assertEqual(tree.num_nodes, 7)
        self.assertEqual(tree.num_candidates, 6)

    def test_ancestor_path_excludes_siblings(self):
        self.assertEqual(self.comb().ancestor_indices(6), (0, 1, 3, 6))
        self.assertEqual(self.comb().ancestor_indices(0), (0,))

    def test_invalid_topology(self):
        cases = (
            ((), ()),
            ((1, 2), (-1,)),
            ((1,), (0,)),
            ((1, 2), (-1, -1)),
            ((1, 2), (-1, 1)),
            ((1, 2, 3), (-1, 2, 1)),
            ((1, 2), (-1, 9)),
        )
        for tokens, parents in cases:
            with self.subTest(tokens=tokens, parents=parents), self.assertRaises(ValueError):
                self.core.TokenTree(tokens, parents)

    def test_duplicate_child_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate child"):
            self.core.TokenTree((0, 1, 1), (-1, 0, 0))
        with self.assertRaisesRegex(ValueError, "Duplicate child"):
            self.core.build_comb_tree(0, ((1, 1),))

    def test_same_token_on_different_branches_is_valid(self):
        tree = self.core.TokenTree((0, 1, 2, 7, 7), (-1, 0, 0, 1, 2))
        self.assertEqual(tree.ancestor_indices(4), (0, 2, 4))

    def test_root_only(self):
        tree = self.core.build_comb_tree(10, ())
        result = self.core.greedy_verify(tree, (99,))
        self.assertEqual(result.emitted_token_ids, (99,))
        self.assertEqual(result.accepted_input_indices, (0,))

    def test_no_match_at_root(self):
        result = self.core.greedy_verify(self.comb(), (99, 21, 88, 31, 77, 55, 44))
        self.assertEqual(result.emitted_token_ids, (99,))
        self.assertEqual(result.accepted_input_indices, (0,))

    def test_second_sibling_is_accepted(self):
        result = self.core.greedy_verify(self.comb(), (12, 21, 88, 31, 77, 55, 44))
        self.assertEqual(result.emitted_token_ids, (12, 88))
        self.assertEqual(result.accepted_input_indices, (0, 2))

    def test_deep_backbone_then_leaf_bonus(self):
        result = self.core.greedy_verify(self.comb(), (11, 21, 88, 31, 77, 55, 44))
        self.assertEqual(result.emitted_token_ids, (11, 21, 31, 55))
        self.assertEqual(result.accepted_input_indices, (0, 1, 3, 5))

    def test_deep_second_sibling(self):
        result = self.core.greedy_verify(self.comb(), (11, 21, 88, 32, 77, 55, 44))
        self.assertEqual(result.emitted_token_ids, (11, 21, 32, 44))
        self.assertEqual(result.accepted_input_indices, (0, 1, 3, 6))

    def test_does_not_follow_token_found_under_a_sibling(self):
        # Node 2 is a leaf. Token 21 belongs to node 1's branch, not node 2.
        result = self.core.greedy_verify(self.comb(), (12, 21, 21, 31, 77, 55, 44))
        self.assertEqual(result.emitted_token_ids, (12, 21))
        self.assertEqual(result.accepted_input_indices, (0, 2))

    def test_root_cannot_jump_to_grandchild(self):
        result = self.core.greedy_verify(self.comb(), (21, 21, 88, 31, 77, 55, 44))
        self.assertEqual(result.emitted_token_ids, (21,))
        self.assertEqual(result.accepted_input_indices, (0,))

    def test_output_budget_keeps_terminal_token_uncached(self):
        predictions = (11, 21, 88, 31, 77, 55, 44)
        for budget, emitted, path in (
            (0, (), ()),
            (1, (11,), (0,)),
            (2, (11, 21), (0, 1)),
            (3, (11, 21, 31), (0, 1, 3)),
            (4, (11, 21, 31, 55), (0, 1, 3, 5)),
            (100, (11, 21, 31, 55), (0, 1, 3, 5)),
        ):
            with self.subTest(budget=budget):
                result = self.core.greedy_verify(self.comb(), predictions, max_output_tokens=budget)
                self.assertEqual(result.emitted_token_ids, emitted)
                self.assertEqual(result.accepted_input_indices, path)

    def test_candidate_budget_truncates_last_sibling_row(self):
        tree = self.core.build_comb_tree(10, ((11, 12), (21, 22)), max_candidates=3)
        self.assertEqual(tree.token_ids, (10, 11, 12, 21))
        self.assertEqual(tree.parents, (-1, 0, 0, 1))
        tree.validate_candidate_count(3)
        self.assertEqual(self.core.build_comb_tree(10, ((11, 12),), max_candidates=0).num_nodes, 1)
        self.assertEqual(self.core.build_comb_tree(10, ((11, 12),), max_candidates=10).num_candidates, 2)

    def test_scheduler_truncation_is_topological(self):
        tree = self.comb().truncate(3)
        self.assertEqual(tree.token_ids, (10, 11, 12, 21))
        self.assertEqual(tree.depths, (0, 1, 1, 2))
        self.assertEqual(self.comb().truncate(0).parents, (-1,))
        with self.assertRaises(ValueError):
            tree.truncate(4)
        with self.assertRaises(ValueError):
            tree.validate_candidate_count(4)

    def test_empty_candidate_row_rejected(self):
        with self.assertRaises(ValueError):
            self.core.build_comb_tree(0, ((1,), (), (2,)))

    def test_prediction_count_validated(self):
        for predictions in ((1,), tuple(range(8))):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                self.core.greedy_verify(self.comb(), predictions)

    def test_invalid_scalar_inputs(self):
        for invalid in (-1, 1.5, True, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.core.build_comb_tree(invalid, ())
                with self.assertRaises(ValueError):
                    self.core.build_comb_tree(0, ((1,),), max_candidates=invalid)
                with self.assertRaises(ValueError):
                    self.core.greedy_verify(self.comb(), tuple(range(7)), max_output_tokens=invalid)
                with self.assertRaises(ValueError):
                    self.comb().truncate(invalid)
        with self.assertRaises(ValueError):
            self.core.TokenTree((0, 1), (-1, True))
        with self.assertRaises(ValueError):
            self.comb().ancestor_indices(7)

    def test_tree_copies_mutable_input_sequences(self):
        tokens = [0, 1]
        parents = [-1, 0]
        tree = self.core.TokenTree(tokens, parents)
        tokens[1] = 99
        parents[1] = 1
        self.assertEqual(tree.token_ids, (0, 1))
        self.assertEqual(tree.parents, (-1, 0))
        with self.assertRaises(FrozenInstanceError):
            tree.token_ids = (5,)

    def test_exhaustive_small_tree_commit_invariants(self):
        tree = self.core.TokenTree((0, 1, 2, 2), (-1, 0, 0, 1))
        for predictions in itertools.product(range(3), repeat=tree.num_nodes):
            for budget in range(tree.num_nodes + 2):
                result = self.core.greedy_verify(tree, predictions, max_output_tokens=budget)
                path = result.accepted_input_indices
                emitted = result.emitted_token_ids
                self.assertEqual(len(path), len(emitted))
                self.assertLessEqual(len(emitted), budget)
                if path:
                    self.assertEqual(path[0], 0)
                    self.assertEqual(path, tree.ancestor_indices(path[-1]))
                for index, node in enumerate(path):
                    self.assertEqual(emitted[index], predictions[node])
                    if index + 1 < len(path):
                        self.assertEqual(emitted[index], tree.token_ids[path[index + 1]])


if __name__ == "__main__":
    unittest.main()
