"""CPU tests for host-slot provenance/guards, without accelerator imports."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


path = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/ops/fla/prefill_state_commit.py"
spec = importlib.util.spec_from_file_location("prefill_state_commit", path)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class HostTable:
    def __init__(self):
        self.device = NS(type="cpu")
        self.ndim = 2
        self.shape = (1, 16)
        self.dtype = "torch.int32"
        self.slot = 7
        self.reads = 0

    def __getitem__(self, key):
        assert key == (0, 0)
        self.reads += 1
        return self.slot


class PrefillStateCommitTests(unittest.TestCase):
    def setUp(self):
        self.table = HostTable()
        self.meta = NS(num_prefills=1, num_decodes=0, num_spec_decodes=0,
                       spec_sequence_masks=None)
        self.args = dict(cache_mode="none", prefix_caching=False, num_reqs=1,
                         num_reqs_padded=1, for_capture=False, pcp_size=1, dcp_size=1)

    def get_slot(self):
        return ops.prefill_host_state_slot(self.meta, self.table, **self.args)

    def test_scalar_snapshot_survives_cpu_table_reuse(self):
        slot = self.get_slot()
        self.table.slot = 42
        self.assertEqual(slot, 7)
        self.assertEqual(self.table.reads, 1)

    def test_runtime_guards_never_read_table(self):
        for key, value in (('cache_mode', 'align'), ('cache_mode', 'all'),
                           ('prefix_caching', True), ('num_reqs', 2),
                           ('num_reqs_padded', 2), ('for_capture', True),
                           ('pcp_size', 2), ('dcp_size', 2)):
            with self.subTest(key=key, value=value):
                old = self.args[key]
                self.args[key] = value
                self.assertIsNone(self.get_slot())
                self.args[key] = old
        self.assertEqual(self.table.reads, 0)

    def test_decode_mixed_and_empty_metadata_fall_back(self):
        for key, value in (('num_prefills', 0), ('num_prefills', 2),
                           ('num_decodes', 1), ('num_spec_decodes', 1),
                           ('spec_sequence_masks', object())):
            old = getattr(self.meta, key)
            setattr(self.meta, key, value)
            self.assertIsNone(self.get_slot())
            setattr(self.meta, key, old)
        self.assertEqual(self.table.reads, 0)

    def test_device_indices_are_not_copied_to_host(self):
        self.table.device.type = 'npu'
        self.assertIsNone(self.get_slot())
        self.assertEqual(self.table.reads, 0)

    def test_malformed_and_negative_host_slots(self):
        for key, value in (('ndim', 1), ('shape', (0, 16)), ('shape', (1, 0)),
                           ('dtype', 'torch.float32')):
            old = getattr(self.table, key)
            setattr(self.table, key, value)
            self.assertIsNone(self.get_slot())
            setattr(self.table, key, old)
        self.table.slot = -1
        self.assertIsNone(self.get_slot())

    def test_only_one_selected_slice_is_copied(self):
        cache = Mock(shape=(10, 2, 3, 4), dtype='half', device='npu:0')
        sink = Mock()
        cache.__getitem__ = Mock(return_value=sink)
        value = NS(shape=(1, 2, 3, 4), dtype='half', device='npu:0')
        self.assertTrue(ops.copy_single_prefill_state(cache, value, 7))
        cache.__getitem__.assert_called_once_with(slice(7, 8))
        sink.copy_.assert_called_once_with(value)

    def test_invalid_slot_or_value_never_writes(self):
        cache = Mock(shape=(10, 2, 3, 4), dtype='half', device='npu:0')
        cache.__getitem__ = Mock()
        value = NS(shape=(1, 2, 3, 4), dtype='half', device='npu:0')
        for slot in (None, True, -1, 10, 1.0):
            self.assertFalse(ops.copy_single_prefill_state(cache, value, slot))
        for key, replacement in (('shape', ()), ('shape', (2, 2, 3, 4)),
                                 ('shape', (1, 2, 3, 5)), ('dtype', 'float'),
                                 ('device', 'cpu')):
            old = getattr(value, key)
            setattr(value, key, replacement)
            self.assertFalse(ops.copy_single_prefill_state(cache, value, 7))
            setattr(value, key, old)
        cache.__getitem__.assert_not_called()


if __name__ == '__main__':
    unittest.main()
