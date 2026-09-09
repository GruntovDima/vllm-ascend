# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU parent-state/commit oracle; no NPU or vLLM imports are required.

Run this file directly with a CPU-capable PyTorch installation. The injected
operators implement one-token conv and GDN, not the old speculative fallback.
NPU operator correctness and end-to-end generation are separate system gates.
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def conv_oracle(x, weight, *, conv_states, cache_indices, bias=None, **kwargs):
    assert kwargs["num_accepted_tokens"] is None
    assert kwargs["query_start_loc"] is None
    outputs = []
    for row, slot in enumerate(cache_indices.tolist()):
        window = torch.cat((conv_states[slot], x[row : row + 1]))
        result = (window.float() * weight.float()).sum(0)
        if bias is not None:
            result += bias.float()
        if kwargs["activation_mode"]:
            result = result * torch.sigmoid(result)
        outputs.append(result.to(x.dtype))
        conv_states[slot].copy_(window[1:])
    return torch.stack(outputs)


def recurrent_oracle(*, q, k, v, g, beta, state, cu_seqlens, ssm_state_indices, **kwargs):
    assert kwargs["num_accepted_tokens"] is None
    assert torch.equal(cu_seqlens[1:] - cu_seqlens[:-1], torch.ones_like(cu_seqlens[1:]))
    q = torch.nn.functional.normalize(q.float(), dim=-1).half().float()
    k = torch.nn.functional.normalize(k.float(), dim=-1).half().float()
    outputs = []
    for row, slot in enumerate(ssm_state_indices.tolist()):
        current = state[slot].float() * g[0, row].float().exp()[:, None, None]
        key = k[0, row]
        delta = (v[0, row].float() - torch.einsum("hvk,hk->hv", current, key)) * beta[0, row, :, None].float()
        current += delta[:, :, None] * key[:, None, :]
        output = torch.einsum("hvk,hk->hv", current, q[0, row]) * key.shape[-1] ** -0.5
        outputs.append(output.to(v.dtype))
        state[slot].copy_(current.to(state.dtype))
    return torch.stack(outputs).unsqueeze(0)


def gates(_log, a, b, _bias):
    return -torch.nn.functional.softplus(a.float()).unsqueeze(0), torch.sigmoid(b.float()).half().unsqueeze(0)


def cache_update_oracle(cache, indices, updates):
    assert indices.dtype == torch.int32 and indices.ndim == 2 and indices.shape[1] == 1
    cache[indices[:, 0].long()] = updates


def split_qkv(x):
    return tuple(part.reshape(1, x.shape[0], 1, 2) for part in x.chunk(3, dim=-1))


class TestTreeGDN(unittest.TestCase):
    def test_compact_dispatch_floor(self):
        for heads in (1, 8, 16, 32):
            for nodes in range(1, 67):
                self.assertEqual(
                    self.module.use_compact_tree_gdn(nodes, heads),
                    heads == 32 and 13 <= nodes <= 65,
                )

    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[4] / "vllm_ascend/_310p/spec_decode/tree_gdn.py"
        spec = importlib.util.spec_from_file_location("_standalone_tree_gdn", source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setup_case(self, accepted=1, parents=(-1, 0, 0, 1, 1)):
        generator = torch.Generator().manual_seed(710)
        count = len(parents)
        depths = []
        for parent in parents:
            depths.append(0 if parent < 0 else depths[parent] + 1)
        context = SimpleNamespace(num_nodes=count, tree=SimpleNamespace(parents=parents, depths=tuple(depths)))
        context.commits = {}
        context.register_commit = lambda key, callback: context.commits.setdefault(key, callback)
        context.gdn_state_indices = {}
        context.previous_accepted_tokens = torch.tensor([accepted], dtype=torch.int32)
        layer = SimpleNamespace(
            prefix="model.layers.0.linear_attn",
            kv_cache=(
                torch.randn((9, 7, 6), generator=generator).half(),
                torch.randn((9, 1, 2, 2), generator=generator).half(),
            ),
            conv1d=SimpleNamespace(
                weight=(torch.randn((6, 1, 4), generator=generator) * 0.2).half(),
                bias=(torch.randn((6,), generator=generator) * 0.1).half(),
            ),
            activation=True,
            A_log=torch.zeros(1),
            dt_bias=torch.zeros(1),
            rearrange_mixed_qkv=split_qkv,
        )
        metadata = SimpleNamespace(
            num_prefills=0,
            num_decodes=0,
            num_spec_decodes=1,
            num_actual_tokens=count,
            spec_state_indices_tensor=torch.tensor([[7, 3, 5, 1, 6]], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32),
        )
        inputs = torch.randn((count, 6), generator=generator).half()
        a = torch.randn((count, 1), generator=generator).half()
        b = torch.randn((count, 1), generator=generator).half()
        return layer, metadata, context, inputs, b, a

    def evaluate(self, case, fused=False, compact=False):
        layer, metadata, context, inputs, b, a = case
        output = inputs.new_empty((context.num_nodes, 1, 2))
        calls = {"conv": [], "gdn": []}

        def conv(*args, **kwargs):
            calls["conv"].append(args[0].shape[0])
            return conv_oracle(*args, **kwargs)

        def recurrent(**kwargs):
            calls["gdn"].append(kwargs["state"].shape[0])
            return recurrent_oracle(**kwargs)

        def tree_recurrent(*, q, k, v, g, beta, initial_state, parents, out, snapshots):
            calls["gdn"].append(len(parents))
            for node, parent in enumerate(parents):
                state = (initial_state if parent < 0 else snapshots[parent : parent + 1]).clone()
                result = recurrent_oracle(
                    q=q[:, node : node + 1], k=k[:, node : node + 1], v=v[:, node : node + 1],
                    g=g[:, node : node + 1], beta=beta[:, node : node + 1], state=state,
                    cu_seqlens=torch.tensor([0, 1]), ssm_state_indices=torch.tensor([0]),
                    num_accepted_tokens=None,
                )
                out[node].copy_(result[0, 0])
                snapshots[node].copy_(state[0])

        def compact_recurrent(**kwargs):
            self.assertNotIn("snapshots", kwargs)
            initial = kwargs["initial_state"]
            snapshots = initial.new_empty((len(kwargs["parents"]), *initial.shape[1:]))
            tree_recurrent(**kwargs, snapshots=snapshots)

            def replay(path):
                calls.setdefault("replay", []).append(path)
                return snapshots[list(path)].clone()

            return replay

        self.module.forward_tree_gdn(
            layer, metadata, context, inputs, b, a, output,
            gating_fn=gates, recurrent_fn=recurrent, conv_fn=conv, cache_update_fn=cache_update_oracle,
            tree_recurrent_fn=tree_recurrent if fused and not compact else None,
            compact_recurrent_fn=compact_recurrent if compact else None,
        )
        return output, calls

    def test_fused_callback_outputs_and_commits_match_depth_reference(self):
        for parents in ((-1,), (-1, 0), (-1, 0, 0, 1, 1), (-1, 0, 1, 0, 3), (-1, 0, 1, 2, 3)):
            for accepted in (1, 3, 5):
                for terminal in range(len(parents)):
                    with self.subTest(parents=parents, accepted=accepted, terminal=terminal):
                        reference = self.setup_case(accepted, parents)
                        fused = self.setup_case(accepted, parents)
                        originals = tuple(cache.clone() for cache in fused[0].kv_cache)
                        expected, _ = self.evaluate(reference)
                        actual, calls = self.evaluate(fused, fused=True)
                        self.assertEqual(calls, {"conv": [len(parents)], "gdn": [len(parents)]})
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(originals, fused[0].kv_cache)))
                        path = []
                        node = terminal
                        while node >= 0:
                            path.append(node)
                            node = parents[node]
                        for case in (reference, fused):
                            case[2].commits[("gdn", case[0].prefix)](tuple(reversed(path)))
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference[0].kv_cache, fused[0].kv_cache)))

    def test_compact_callback_all_paths_and_no_early_mutation(self):
        for parents in ((-1,), (-1, 0), (-1, 0, 0, 1, 1), (-1, 0, 1, 0, 3), (-1, 0, 1, 2, 3)):
            for accepted in (1, 3, 5):
                for terminal in range(len(parents)):
                    with self.subTest(parents=parents, accepted=accepted, terminal=terminal):
                        reference = self.setup_case(accepted, parents)
                        compact = self.setup_case(accepted, parents)
                        before = tuple(x.clone() for x in compact[0].kv_cache)
                        expected, _ = self.evaluate(reference, fused=True)
                        actual, calls = self.evaluate(compact, compact=True)
                        self.assertTrue(torch.equal(expected, actual))
                        self.assertNotIn("replay", calls)
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(before, compact[0].kv_cache)))
                        path = []
                        node = terminal
                        while node >= 0:
                            path.append(node)
                            node = parents[node]
                        path = tuple(reversed(path))
                        compact[2].commits[("gdn", compact[0].prefix)](())
                        self.assertNotIn("replay", calls)
                        with self.assertRaises(ValueError):
                            compact[2].commits[("gdn", compact[0].prefix)]((0, 0))
                        self.assertNotIn("replay", calls)
                        for case in (reference, compact):
                            case[2].commits[("gdn", case[0].prefix)](path)
                        self.assertEqual(calls["replay"], [path])
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(reference[0].kv_cache, compact[0].kv_cache)))

    def sequential_oracle(self, case, terminal):
        layer, metadata, context, inputs, b, a = case
        path = []
        node = terminal
        while node >= 0:
            path.append(node)
            node = context.tree.parents[node]
        path.reverse()
        accepted = int(metadata.num_accepted_tokens[0])
        state_ids = metadata.spec_state_indices_tensor[0]
        history = layer.kv_cache[0][state_ids[0], accepted - 1 : accepted + 2].unsqueeze(0).clone()
        state = layer.kv_cache[1][state_ids[accepted - 1]].unsqueeze(0).clone()
        weights = layer.conv1d.weight[:, 0].transpose(0, 1)
        for node in path:
            convolved = conv_oracle(
                inputs[node : node + 1], weights, conv_states=history,
                cache_indices=torch.tensor([0]), bias=layer.conv1d.bias,
                num_accepted_tokens=None, query_start_loc=None, activation_mode=1,
            )
            q, k, v = split_qkv(convolved)
            g, beta = gates(None, a[node : node + 1], b[node : node + 1], None)
            output = recurrent_oracle(
                q=q, k=k, v=v, g=g, beta=beta, state=state,
                cu_seqlens=torch.tensor([0, 1]), ssm_state_indices=torch.tensor([0]),
                num_accepted_tokens=None,
            )
        return output[0, 0], state[0], history[0]

    def test_all_nodes_match_separate_ancestor_decodes(self):
        for accepted in (1, 2, 3, 5):
            for parents in ((-1, 0, 0, 1, 1), (-1, 0, 1, 0, 3)):
                with self.subTest(accepted=accepted, parents=parents):
                    case = self.setup_case(accepted, parents)
                    old_caches = tuple(cache.clone() for cache in case[0].kv_cache)
                    output, calls = self.evaluate(case)
                    self.assertEqual(calls, {"conv": [5], "gdn": [1, 2, 2]})
                    for node in range(5):
                        expected, _, _ = self.sequential_oracle(case, node)
                        torch.testing.assert_close(output[node], expected, rtol=0, atol=0)
                    for old, current in zip(old_caches, case[0].kv_cache):
                        self.assertTrue(torch.equal(old, current), "Verification mutated the committed cache")

    def test_commit_actual_path_and_next_native_history(self):
        for accepted in (1, 2, 3, 5):
            for path in ((0,), (0, 1), (0, 2), (0, 1, 3), (0, 1, 4)):
                with self.subTest(accepted=accepted, path=path):
                    case = self.setup_case(accepted)
                    layer, metadata, context, inputs, _, _ = case
                    originals = tuple(cache.clone() for cache in layer.kv_cache)
                    expected = [self.sequential_oracle(case, node) for node in path]
                    self.evaluate(case)
                    context.commits[("gdn", layer.prefix)](path)
                    ids = metadata.spec_state_indices_tensor[0]
                    for index, (_, state, _) in enumerate(expected):
                        self.assertTrue(torch.equal(layer.kv_cache[1][ids[index]], state))
                    window = layer.kv_cache[0][ids[0], len(path) - 1 : len(path) + 2]
                    self.assertTrue(torch.equal(window, expected[-1][2]))
                    unmodified_ssm = [slot for slot in range(9) if slot not in ids[:len(path)].tolist()]
                    self.assertTrue(torch.equal(layer.kv_cache[1][unmodified_ssm], originals[1][unmodified_ssm]))
                    unmodified_conv = [slot for slot in range(9) if slot != int(ids[0])]
                    self.assertTrue(torch.equal(layer.kv_cache[0][unmodified_conv], originals[0][unmodified_conv]))
                    tape_length = 2 + len(path)
                    self.assertEqual(torch.count_nonzero(layer.kv_cache[0][ids[0], tape_length:]).item(), 0)

    def test_conv_history_indices_exclude_siblings_and_keep_old_prefix(self):
        self.assertEqual(
            self.module._tree_conv_history_indices((-1, 0, 0, 1, 1), 3),
            ((0, 1, 2), (1, 2, 3), (1, 2, 3), (2, 3, 4), (2, 3, 4)),
        )
        self.assertEqual(
            self.module._tree_conv_history_indices((-1, 0, 1, 2, 3), 3)[-1], (4, 5, 6)
        )
        for parents in ((0,), (-1, -1), (-1, 2)):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                self.module._tree_conv_history_indices(parents, 3)

    def test_all_truncated_trees_and_deep_chain_match_ar(self):
        for parents in ((-1,), (-1, 0), (-1, 0, 0), (-1, 0, 0, 1), (-1, 0, 1, 2, 3)):
            with self.subTest(parents=parents):
                case = self.setup_case(accepted=3, parents=parents)
                observed, calls = self.evaluate(case)
                self.assertEqual(calls["conv"], [len(parents)])
                expected = torch.stack([self.sequential_oracle(case, node)[0] for node in range(len(parents))])
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_sibling_does_not_contaminate_primary_descendants(self):
        normal = self.setup_case(accepted=3)
        changed = self.setup_case(accepted=3)
        changed[3][2].add_(10)
        normal_output, _ = self.evaluate(normal)
        changed_output, _ = self.evaluate(changed)
        self.assertFalse(torch.equal(normal_output[2], changed_output[2]))
        self.assertTrue(torch.equal(normal_output[[0, 1, 3, 4]], changed_output[[0, 1, 3, 4]]))
        for case in (normal, changed):
            case[2].commits[("gdn", case[0].prefix)]((0, 1, 4))
        for old, current in zip(normal[0].kv_cache, changed[0].kv_cache):
            self.assertTrue(torch.equal(old, current))

    def test_invalid_commit_path_is_rejected_before_any_mutation(self):
        for path in ((1,), (0, 3), (0, 2, 3), (0, 0), (0, -1), (0, True)):
            with self.subTest(path=path):
                case = self.setup_case()
                old = tuple(cache.clone() for cache in case[0].kv_cache)
                self.evaluate(case)
                with self.assertRaises(ValueError):
                    case[2].commits[("gdn", case[0].prefix)](path)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, case[0].kv_cache)))

    def test_empty_commit_does_not_mutate(self):
        case = self.setup_case()
        original = tuple(cache.clone() for cache in case[0].kv_cache)
        self.evaluate(case)
        case[2].commits[("gdn", case[0].prefix)](())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, case[0].kv_cache)))

    def test_root_only_after_deep_acceptance_and_return_to_tree(self):
        first = self.setup_case(accepted=1)
        self.evaluate(first)
        layer, metadata, context = first[:3]
        context.commits[("gdn", layer.prefix)]((0, 1, 4))

        root_case = list(self.setup_case(accepted=3, parents=(-1,)))
        root_case[0] = layer
        root_metadata, root_context = root_case[1:3]
        # Use a full-table oracle before mimicking native non-spec metadata.
        expected_output, expected_state, expected_history = self.sequential_oracle(root_case, 0)
        root_context.gdn_state_indices = context.gdn_state_indices
        root_metadata.num_spec_decodes = 0
        root_metadata.num_decodes = 1
        root_metadata.non_spec_state_indices_tensor = metadata.spec_state_indices_tensor[0, :1]
        root_metadata.spec_state_indices_tensor = None
        root_metadata.num_accepted_tokens = None
        observed, calls = self.evaluate(root_case)
        self.assertTrue(torch.equal(observed[0], expected_output))
        self.assertEqual(calls, {"conv": [1], "gdn": [1]})
        root_context.commits[("gdn", layer.prefix)]((0,))
        self.assertTrue(torch.equal(layer.kv_cache[1][7], expected_state))
        self.assertTrue(torch.equal(layer.kv_cache[0][7, :3], expected_history))
        self.assertEqual(root_context.gdn_state_indices[layer.prefix].tolist(), [7])

        final_case = list(self.setup_case(accepted=1))
        final_case[0] = layer
        final_case[2].gdn_state_indices = root_context.gdn_state_indices
        expected = [self.sequential_oracle(final_case, node)[0] for node in range(5)]
        observed, _ = self.evaluate(final_case)
        self.assertTrue(torch.equal(observed, torch.stack(expected)))

    def test_truncated_table_reads_previous_full_checkpoint_map(self):
        first = self.setup_case(accepted=1)
        self.evaluate(first)
        layer, _, context = first[:3]
        context.commits[("gdn", layer.prefix)]((0, 1, 4))
        next_case = list(self.setup_case(accepted=3, parents=(-1, 0)))
        next_case[0] = layer
        next_case[2].gdn_state_indices = context.gdn_state_indices
        expected = [self.sequential_oracle(next_case, node)[0] for node in range(2)]
        next_case[1].spec_state_indices_tensor = next_case[1].spec_state_indices_tensor[:, :2]
        observed, _ = self.evaluate(next_case)
        self.assertTrue(torch.equal(observed, torch.stack(expected)))
        next_case[2].commits[("gdn", layer.prefix)]((0, 1))
        self.assertEqual(next_case[2].gdn_state_indices[layer.prefix].tolist(), [7, 3])

    def test_first_root_only_after_prefill_uses_first_checkpoint(self):
        case = self.setup_case(accepted=1, parents=(-1,))
        expected = self.sequential_oracle(case, 0)[0]
        metadata, context = case[1:3]
        metadata.num_spec_decodes = 0
        metadata.num_decodes = 1
        metadata.non_spec_state_indices_tensor = metadata.spec_state_indices_tensor[0, :1]
        metadata.spec_state_indices_tensor = None
        metadata.num_accepted_tokens = None
        # Any stale request count is ignored when no previous tree table exists.
        context.previous_accepted_tokens = torch.tensor([5], dtype=torch.int32)
        observed, _ = self.evaluate(case)
        self.assertTrue(torch.equal(observed[0], expected))

    def test_unsupported_metadata_fails_closed(self):
        cases = ("mixed_batch", "missing_count", "missing_indices", "short_cache", "fp32_state")
        for invalid in cases:
            case = self.setup_case()
            layer, metadata = case[:2]
            if invalid == "mixed_batch":
                metadata.num_decodes = 1
            elif invalid == "missing_count":
                metadata.num_accepted_tokens = None
            elif invalid == "missing_indices":
                metadata.spec_state_indices_tensor = None
            elif invalid == "short_cache":
                layer.kv_cache = (layer.kv_cache[0][:, :3], layer.kv_cache[1])
            else:
                layer.kv_cache = (layer.kv_cache[0], layer.kv_cache[1].float())
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.evaluate(case)


if __name__ == "__main__":
    unittest.main()
