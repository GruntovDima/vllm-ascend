"""Run directly to avoid importing unrelated NPU test fixtures on CPU."""
import importlib.util
from pathlib import Path
import unittest
import sys

import torch

DEVICE = 'cpu'
if '--npu' in sys.argv:
    import torch_npu  # noqa: F401

    sys.argv.remove('--npu')
    torch.npu.set_device(0)
    DEVICE = 'npu'

SOURCE = Path(__file__).resolve().parents[3] / 'vllm_ascend/_310p/compact_greedy.py'
spec = importlib.util.spec_from_file_location('compact_greedy', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompactGreedyTests(unittest.TestCase):
    def test_exact_canonical_greedy(self):
        for dtype in (torch.float16, torch.float32):
            for kept in ([0, 2, 4], [1, 3, 5], [0, 1, 2, 3, 4, 5]):
                inv = torch.full((6,), len(kept), dtype=torch.long)
                inv[kept] = torch.arange(len(kept))
                mapping, insertion = module.make_compact_mapping(inv, len(kept))
                torch.manual_seed(13)
                logits = torch.randn(17, len(kept), dtype=dtype)
                logits[0].fill_(0)  # Tie: lowest original ID.
                logits[1].fill_(-torch.inf)  # Missing finite-floor row must win.
                logits[2].fill_(torch.finfo(dtype).min)
                logits[3].fill_(torch.inf)
                logits[4].fill_(torch.nan)
                logits[5, -1] = torch.nan
                logits[6].fill_(-torch.inf)
                logits[6, -1] = torch.finfo(dtype).min
                logits, inv, mapping = logits.to(DEVICE), inv.to(DEVICE), mapping.to(DEVICE)
                floor = torch.full_like(logits[:, :1], torch.finfo(dtype).min)
                canonical = torch.cat((logits, floor), -1).index_select(-1, inv)
                got = module.compact_greedy_ids(logits, mapping, insertion)
                self.assertTrue(torch.equal(got, canonical.argmax(-1)), (dtype, kept))

    def test_reject_invalid_mapping(self):
        for inv in ([1, 2, 0], [0, 0, 2], [0, 4, 1]):
            with self.assertRaises(ValueError):
                module.make_compact_mapping(torch.tensor(inv), 2)


if __name__ == '__main__':
    unittest.main()
