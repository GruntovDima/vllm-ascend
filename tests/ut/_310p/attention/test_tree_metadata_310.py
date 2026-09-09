# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the real builder limit and phase-classification methods.

Only unrelated registration and NPU mask allocation are stubbed. Method ASTs
are loaded from production sources so a missing limit hook cannot pass.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[4]


def load_builders():
    spec = importlib.util.spec_from_file_location(
        "_tree_runtime_test_helpers", ROOT / "tests/ut/_310p/spec_decode/test_tree_runtime.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    _, runtime = helper.load_tree_modules()

    class InitBase:
        def __init__(self, kv_cache_spec, *args):
            self.kv_cache_spec = kv_cache_spec

    namespace = {
        "InitBase": InitBase, "torch": torch, "F": torch.nn.functional,
        "TreeMTPConfig": runtime.TreeMTPConfig,
        "cdiv": lambda a, b: (a + b - 1) // b,
        "AscendAttentionBackend": SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128]),
        "AttentionMaskBuilder": lambda device: None,
        "is_pd_decode_recompute_scheduler_enabled": lambda: False,
    }
    body = list(ast.parse("from __future__ import annotations").body)
    utils = ast.parse((ROOT / "vllm_ascend/attention/utils.py").read_text())
    body.extend(node for node in utils.body if isinstance(node, ast.FunctionDef)
                and node.name == "split_decodes_and_prefills")
    for filename, class_name, parent, methods in (
        ("vllm_ascend/attention/attention_v1.py", "AscendAttentionMetadataBuilder", "InitBase",
         {"__init__", "_get_max_decode_threshold", "_split_decodes_and_prefills"}),
        ("vllm_ascend/_310p/attention/metadata_builder.py", "AscendAttentionMetadataBuilder310",
         "AscendAttentionMetadataBuilder", {"_get_max_decode_threshold", "_split_decodes_and_prefills"}),
    ):
        source = ast.parse((ROOT / filename).read_text())
        original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        selected = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        assert {node.name for node in selected} == methods
        body.append(ast.ClassDef(name=class_name, bases=[ast.Name(id=parent, ctx=ast.Load())],
                                 keywords=[], body=selected, decorator_list=[]))
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, "_real_tree_metadata_methods", "exec"), namespace)
    return namespace, helper


class TestTreeMetadata310(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.helper = load_builders()

    def config(self, width=16, depth=4, enabled=True):
        config = self.helper.make_vllm_config()
        config.additional_config["tree_mtp"].update(enabled=enabled, width=width, depth=depth)
        config.speculative_config.num_speculative_tokens = width * depth
        config.model_config.max_model_len = 2048
        config.scheduler_config.enable_chunked_prefill = True
        config.compilation_config = SimpleNamespace()
        return config

    def builder(self, config, specialized=True):
        name = "AscendAttentionMetadataBuilder310" if specialized else "AscendAttentionMetadataBuilder"
        return self.ns[name](None, [], config, torch.device("cpu"))

    def test_only_validated_tree_can_exceed_fia_limit(self):
        for width in (1, 2, 4, 8, 16):
            for depth in range(1, 5):
                builder = self.builder(self.config(width, depth))
                self.assertEqual(builder.decode_threshold, width * depth + 1)
                self.assertEqual(builder.reorder_batch_threshold, width * depth + 1)
        with self.assertRaises(AssertionError):
            self.builder(self.config(), specialized=False)
        with self.assertRaises(AssertionError):
            self.builder(self.config(enabled=False))
        with self.assertRaises(ValueError):
            self.builder(self.config(width=3))

    def test_ordinary_limit_is_unchanged(self):
        config = self.config(enabled=False)
        config.speculative_config.num_speculative_tokens = 15
        self.assertEqual(self.builder(config).decode_threshold, 16)
        config.speculative_config.num_speculative_tokens = 16
        with self.assertRaises(AssertionError):
            self.builder(config)

    def test_short_prefill_is_not_mistaken_for_wide_decode(self):
        builder = self.builder(self.config())
        for length, prefilling in ((1, True), (11, True), (11, False), (65, False)):
            with self.subTest(length=length, prefilling=prefilling):
                metadata = SimpleNamespace(
                    context_parallel_metadata=None, max_query_len=length,
                    num_reqs=1, num_actual_tokens=length,
                    query_start_loc_cpu=torch.tensor([0, length]),
                    is_prefilling=torch.tensor([prefilling]),
                )
                expected = (0, 1, 0, length) if prefilling else (1, 0, length, 0)
                self.assertEqual(builder._split_decodes_and_prefills(metadata), expected)


if __name__ == "__main__":
    unittest.main()
