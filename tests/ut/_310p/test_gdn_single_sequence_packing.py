"""Exact-value packing tests; run directly without importing the plugin.

For a separately monitored NPU screen, set PACKING_TEST_DEVICE=npu:0.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class SingleSequencePackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.getenv("PACKING_TEST_DEVICE", "cpu")
        if cls.device.startswith("npu"):
            import torch_npu

            torch_npu.npu.set_device(cls.device)
            # Match NPUWorker310 rather than the default JIT microtest route.
            torch_npu.npu.set_compile_mode(jit_compile=False)
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py"
        wanted = {"_ceil_div", "_pad_varlen_to_chunk", "_unpad_chunk_output"}
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted]
        assert len(nodes) == len(wanted)
        future = ast.parse("from __future__ import annotations").body[0]
        cls.env = SimpleNamespace(VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING=False)
        cls.scope = {"torch": torch, "envs": cls.env, "CHUNK_SIZE": 64}
        exec(compile(ast.Module(body=[future, *nodes], type_ignores=[]), str(source), "exec"), cls.scope)

    def pad(self, values, lengths, enabled):
        self.env.VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING = enabled
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
        return self.scope["_pad_varlen_to_chunk"](*values, cu, 64)

    def test_padding_exact_and_inputs_unchanged(self):
        for dtype in (torch.float16, torch.float32):
            for lengths in ([0], [1], [63], [64], [65], [782], [1266], [2048], [63, 65], [0, 65, 0]):
                with self.subTest(dtype=dtype, lengths=lengths):
                    total = sum(lengths)
                    # Non-contiguous input views also cover the no-padding alias path.
                    values = [torch.arange(total * 2 * 8, device=self.device, dtype=torch.float32)
                              .reshape(1, total, 2, 8).to(dtype)[..., ::2] for _ in range(3)]
                    values += [torch.ones((1, total, 2), dtype=dtype, device=self.device) for _ in range(2)]
                    before = [value.clone() for value in values]
                    old = self.pad(values, lengths, False)
                    new = self.pad(values, lengths, True)
                    for a, b in zip(old[:5], new[:5]):
                        self.assertTrue(torch.equal(a, b))
                    self.assertEqual(old[5], new[5])
                    self.assertTrue(torch.equal(old[6], new[6]))
                    for a, b in zip(values, before):
                        self.assertTrue(torch.equal(a, b))

    def test_singleton_does_not_copy_twice(self):
        values = [torch.ones((1, 64, 2, 4), device=self.device) for _ in range(3)]
        values += [torch.ones((1, 64, 2), device=self.device) for _ in range(2)]
        old, new = self.pad(values, [64], False), self.pad(values, [64], True)
        for index, value in enumerate(values):
            self.assertNotEqual(old[index].data_ptr(), value.data_ptr())
            self.assertEqual(new[index].data_ptr(), value.data_ptr())

    def test_unpad_exact_shape_and_poison_excluded(self):
        for tokens in (0, 1, 64, 65, 782, 1266, 2048):
            for tnd in (False, True):
                padded = ((tokens + 63) // 64) * 64
                out = torch.full((1, padded, 2, 4), float("nan"), device=self.device)
                out[:, :tokens] = 3.25
                args = (out, [(0, 0, tokens)], tokens, tnd, True)
                self.env.VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING = False
                old = self.scope["_unpad_chunk_output"](*args)
                self.env.VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING = True
                new = self.scope["_unpad_chunk_output"](*args)
                self.assertTrue(torch.equal(old, new))
                self.assertEqual(old.shape, new.shape)
                self.assertEqual(old.is_contiguous(), new.is_contiguous())
                if tokens:
                    self.assertEqual(out.data_ptr(), new.data_ptr())

    def test_multisequence_keeps_compaction(self):
        out = torch.full((1, 192, 2, 4), float("nan"), device=self.device)
        out[:, :63] = 1
        out[:, 64:129] = 2
        args = (out, [(0, 0, 63), (0, 63, 128)], 128, True, True)
        self.env.VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING = False
        old = self.scope["_unpad_chunk_output"](*args)
        self.env.VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING = True
        new = self.scope["_unpad_chunk_output"](*args)
        self.assertTrue(torch.equal(old, new))
        self.assertNotEqual(out.data_ptr(), new.data_ptr())


if __name__ == "__main__":
    unittest.main()
