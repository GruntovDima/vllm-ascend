# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only contract tests; these do not validate native splitfuse masking.

The isolated AST loader executes the real helpers/backend class while replacing
the upstream backend and device bindings. This permits running the file with
only CPU torch installed, without importing vLLM, torch_npu, or any NPU runtime.
"""

import ast
import unittest
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch


class StubBackend:
    def do_kv_cache_update(self, *args):
        self.base_cache_calls.append(args)

    def reshape_and_cache(self, query, key, value, kv_cache, metadata, output):
        self.base_cache_calls.append((query, key, value, kv_cache, metadata, output))
        return query, key, value, output


class FakeContext:
    def __init__(self, prefix=2, parents=(-1, 0, 0, 1, 1, 2)):
        self.prefix_length = prefix
        self.tree = SimpleNamespace(parents=parents)
        self.num_nodes = len(parents)
        self.callbacks = {}

    def has_commit(self, key):
        return key in self.callbacks

    def register_commit(self, key, callback):
        if key in self.callbacks:
            raise RuntimeError("Duplicate commit")
        self.callbacks[key] = callback


class TestTreeAttention310(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[4]
        source = root / "vllm_ascend/_310p/attention/attention_v1.py"
        nodes = ast.parse(source.read_text()).body
        selected = {
            "_tree_context_from_metadata", "_tree_dimensions", "build_tree_attention_mask",
            "register_tree_cache_commit", "AscendAttentionBackendImpl310",
        }
        namespace = {
            "Any": Any, "Integral": Integral, "torch": torch,
            "AscendAttentionBackendImpl": StubBackend, "AscendMetadata": SimpleNamespace,
            "AttentionType": SimpleNamespace(DECODER="decoder"),
            "AscendAttentionState": SimpleNamespace(
                PrefillNoCache=0, DecodeOnly=1, ChunkedPrefill=2, PrefillCacheHit=3, SpecDecoding=4
            ),
            "ACL_FORMAT_FRACTAL_NZ": 29,
        }
        body = [
            node for node in nodes
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected
            or isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "TREE_MASK_ALIGNMENT" for target in node.targets)
        ]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
        utils_source = root / "vllm_ascend/utils.py"
        nz_function = next(
            node for node in ast.parse(utils_source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "nd_to_nz_spec"
        )
        exec(compile(ast.Module(body=[nz_function], type_ignores=[]), str(utils_source), "exec"), namespace)
        cls.namespace = namespace

    def setUp(self):
        self.writer = MagicMock()
        self.legacy = MagicMock()
        self.compressed = MagicMock()
        self.namespace.update({
            "DeviceOperator": SimpleNamespace(reshape_and_cache=self.writer),
            "torch_npu": SimpleNamespace(
                npu_format_cast=MagicMock(side_effect=lambda tensor, _: tensor),
                _npu_paged_attention_splitfuse=self.legacy,
                _npu_paged_attention_splitfuse_v2=self.compressed,
            ),
            "get_forward_context": MagicMock(return_value=SimpleNamespace(attn_metadata=None)),
            "is_forward_context_available": MagicMock(return_value=True),
            "_EXTRA_CTX": SimpleNamespace(capturing=False),
        })
        self.context = FakeContext()
        self.impl = self.namespace["AscendAttentionBackendImpl310"].__new__(
            self.namespace["AscendAttentionBackendImpl310"]
        )
        self.impl.attn_type = "decoder"
        self.impl.sliding_window = None
        self.impl.kv_sharing_target_layer_name = None
        self.impl.base_cache_calls = []
        self.impl.support_compressed_mask = True
        self.impl.num_heads = 2
        self.impl.num_kv_heads = 1
        self.impl.scale = 0.25
        self.impl.key_cache = torch.zeros((8, 1, 4, 16), dtype=torch.float16)
        self.impl.value_cache = torch.zeros_like(self.impl.key_cache)
        self.query = torch.randn((8, 2, 16), dtype=torch.float16)
        self.output = torch.full_like(self.query, 17)
        self.key = torch.arange(6 * 16, dtype=torch.float16).reshape(6, 1, 16)
        self.value = self.key + 1000
        self.slots = torch.tensor([3, 8, 9, 10, 11, 16], dtype=torch.int32)
        self.metadata = SimpleNamespace(
            tree_mtp_context=self.context, attn_state=4, num_actual_tokens=6,
            block_tables=torch.tensor([[0, 2, 4]], dtype=torch.int32),
            seq_lens=torch.tensor([999], dtype=torch.int32), slot_mapping=self.slots,
        )

    def mask(self, context=None):
        return self.namespace["build_tree_attention_mask"](context or self.context, torch.device("cpu"))

    def register(self, context=None):
        context = context or self.context
        self.namespace["register_tree_cache_commit"](
            context, self.key, self.value, self.impl.key_cache, self.impl.value_cache, self.slots
        )
        return next(iter(context.callbacks.values()))

    def test_mask_prefix_self_and_ancestors_only(self):
        mask = self.mask()
        expected = [{0}, {0, 1}, {0, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 5}]
        self.assertEqual(mask.shape, (6, 16))
        self.assertEqual(mask.dtype, torch.float16)
        for node, ancestors in enumerate(expected):
            allowed = set(torch.where(mask[node] == 0)[0].tolist())
            self.assertEqual(allowed, {0, 1} | {2 + ancestor for ancestor in ancestors})
            self.assertTrue(torch.isneginf(mask[node, 8:]).all())

    def test_root_only_zero_prefix_and_aligned_tail(self):
        mask = self.mask(FakeContext(prefix=0, parents=(-1,)))
        self.assertEqual(mask[0, 0], 0)
        self.assertTrue(torch.isneginf(mask[0, 1:]).all())

    def test_mask_crosses_alignment_boundary(self):
        mask = self.mask(FakeContext(prefix=15))
        self.assertEqual(mask.shape, (6, 32))
        self.assertTrue((mask[:, :15] == 0).all())
        self.assertTrue(torch.isneginf(mask[:, 21:]).all())

    def test_mask_rejects_bad_context(self):
        contexts = [FakeContext(prefix=-1), FakeContext(parents=(0,)), FakeContext(parents=(-1, 2))]
        bad_count = FakeContext()
        bad_count.num_nodes += 1
        contexts.append(bad_count)
        for context in contexts:
            with self.subTest(context=vars(context)), self.assertRaises(ValueError):
                self.mask(context)

    def test_commit_preserves_snapshots_and_cross_block_slots(self):
        original_key, original_value = self.key.clone(), self.value.clone()
        callback = self.register()
        self.key.fill_(-1)
        self.value.fill_(-2)
        self.slots.fill_(-3)
        callback((0, 2, 5))
        kwargs = self.writer.call_args.kwargs
        torch.testing.assert_close(kwargs["key"], original_key[[0, 2, 5]])
        torch.testing.assert_close(kwargs["value"], original_value[[0, 2, 5]])
        torch.testing.assert_close(kwargs["slot_mapping"], torch.tensor([3, 8, 9], dtype=torch.int32))
        self.assertIs(kwargs["key_cache"], self.impl.key_cache)
        self.assertIs(kwargs["value_cache"], self.impl.value_cache)

    def test_commit_gathers_both_sources_before_writer(self):
        original_key, original_value = self.key.clone(), self.value.clone()
        callback = self.register()

        def write(**kwargs):
            self.key.fill_(-1)
            self.value.fill_(-2)
            torch.testing.assert_close(kwargs["key"], original_key[[0, 1, 3]])
            torch.testing.assert_close(kwargs["value"], original_value[[0, 1, 3]])

        self.writer.side_effect = write
        callback((0, 1, 3))
        self.writer.assert_called_once()

    def test_duplicate_writer_does_not_replace_first_snapshot(self):
        expected = self.key[0:1].clone()
        callback = self.register()
        self.key.fill_(-1)
        self.register()
        self.assertEqual(len(self.context.callbacks), 1)
        callback((0,))
        torch.testing.assert_close(self.writer.call_args.kwargs["key"], expected)

    def test_empty_commit_does_not_write(self):
        self.register()(())
        self.writer.assert_not_called()

    def test_commit_rejects_invalid_path_before_write(self):
        callback = self.register()
        for path in ((1,), (0, 1, 2), (0, 6), (0, 1, 1), (0, True), (0, 1.0)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                callback(path)
        self.writer.assert_not_called()

    def test_both_writer_entrypoints_register_once(self):
        layer = SimpleNamespace(layer_name="layer")
        self.namespace["get_forward_context"].return_value.attn_metadata = {"layer": self.metadata}
        caches = (self.impl.key_cache, self.impl.value_cache)
        self.impl.do_kv_cache_update(layer, self.key, self.value, caches, self.slots)
        self.impl.reshape_and_cache(self.query, self.key, self.value, caches, self.metadata, self.output)
        self.assertEqual(len(self.context.callbacks), 1)
        self.assertEqual(len(self.impl.base_cache_calls), 2)

    def test_sharing_consumer_never_registers_commit(self):
        self.impl.kv_sharing_target_layer_name = "producer"
        self.namespace["get_forward_context"].return_value.attn_metadata = {"layer": self.metadata}
        caches = (self.impl.key_cache, self.impl.value_cache)
        self.impl.do_kv_cache_update(SimpleNamespace(layer_name="layer"), self.key, self.value, caches, self.slots)
        self.impl.reshape_and_cache(self.query, self.key, self.value, caches, self.metadata, self.output)
        self.assertFalse(self.context.callbacks)
        self.assertEqual(len(self.impl.base_cache_calls), 1)

    def test_direct_ordinary_cache_update_needs_no_forward_context(self):
        self.namespace["is_forward_context_available"].return_value = False
        self.impl.do_kv_cache_update(
            None, self.key, self.value, (self.impl.key_cache, self.impl.value_cache), self.slots
        )
        self.namespace["get_forward_context"].assert_not_called()
        self.assertEqual(len(self.impl.base_cache_calls), 1)

    def test_tree_dispatch_bypasses_compressed_and_bounds_lengths(self):
        self.legacy.side_effect = lambda **kwargs: kwargs["out"].fill_(3)
        returned = self.impl.forward_impl(self.query, None, None, None, self.metadata, self.output)
        self.assertIs(returned, self.output)
        self.compressed.assert_not_called()
        kwargs = self.legacy.call_args.kwargs
        self.assertEqual(kwargs["query"].shape[0], 6)
        self.assertEqual(kwargs["seq_len"].device.type, "cpu")
        torch.testing.assert_close(kwargs["seq_len"], torch.tensor([6], dtype=torch.int32))
        torch.testing.assert_close(kwargs["context_lens"], torch.tensor([8], dtype=torch.int32))
        self.assertTrue((self.output[:6] == 3).all())
        self.assertTrue((self.output[6:] == 17).all())
        packed_mask = kwargs["mask"]
        dense_mask = packed_mask.permute(0, 2, 1, 3).reshape(16, 16)
        torch.testing.assert_close(dense_mask[:6], self.mask())

    def test_tree_rejects_graph_batch_or_incomplete_block_table(self):
        cases = (
            ("capturing", True), ("batch", torch.zeros((2, 3), dtype=torch.int32)),
            ("blocks", torch.zeros((1, 1), dtype=torch.int32)),
        )
        for name, value in cases:
            with self.subTest(name=name):
                self.namespace["_EXTRA_CTX"].capturing = name == "capturing"
                self.metadata.block_tables = value if name != "capturing" else torch.zeros((1, 3), dtype=torch.int32)
                with self.assertRaises((ValueError, NotImplementedError)):
                    self.impl.forward_tree_attention_310(self.query, self.metadata, self.output)
        self.legacy.assert_not_called()

    def test_ordinary_and_mock_metadata_do_not_enable_tree(self):
        for metadata in (SimpleNamespace(attn_state=4), MagicMock(attn_state=4)):
            with self.subTest(metadata=metadata):
                self.impl.forward_chunked_prefill_310 = MagicMock(return_value=self.output)
                returned = self.impl.forward_impl(self.query, None, None, None, metadata, self.output)
                self.impl.forward_chunked_prefill_310.assert_called_once_with(self.query, metadata, self.output)
                self.assertIs(returned, self.output)
        self.legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
