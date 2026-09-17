"""Dependency-free host-policy and schedule checks for QBMM K pipelining."""

import ast
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
QBMM_ROOT = ROOT / "csrc/qbmm/quant_batch_matmul_v3_x"


def _simulate_weight_schedule(
    k: int, n: int, base_k: int, base_n: int, early: bool, m_tiles: int
):
    """Model ping/pong ownership and return every bounded DMA rectangle."""
    k_passes = (k + base_k - 1) // base_k
    col_batches = (n + base_n - 1) // base_n
    rectangles = []
    for m_tile in range(m_tiles):
        work = [(cb, kp) for cb in range(col_batches) for kp in range(k_passes)]
        ready = [False, False]
        free = [True, True]
        ready[0], free[0] = True, False  # Per-mTile prime loads ping.
        current = 0

        for index, (cb, kp) in enumerate(work):
            if not ready[current]:
                raise AssertionError("weight buffer read before MTE2 ready")
            ready[current] = False
            k_start = kp * base_k
            k_size = min(base_k, k - k_start)
            n_start = cb * base_n
            n_size = min(base_n, n - n_start)
            rectangles.append((m_tile, k_start, k_size, n_start, n_size))

            next_buffer = 1 - current
            has_next = index + 1 < len(work)

            # The experimental enqueue fills the other L1 before this pass's
            # L1->L0 loads/MMAD. The source and destination L1 buffers differ.
            if early and has_next:
                if not free[next_buffer] or ready[next_buffer]:
                    raise AssertionError("weight buffer refilled before consumption")
                free[next_buffer], ready[next_buffer] = False, True

            # MTE1_MTE2 is emitted only after the current L1 read.
            if free[current]:
                raise AssertionError("duplicate free token")
            free[current] = True

            # Baseline/fallback enqueue is the same transfer after current MMAD.
            if not early and has_next:
                if not free[next_buffer] or ready[next_buffer]:
                    raise AssertionError("weight buffer refilled before consumption")
                free[next_buffer], ready[next_buffer] = False, True
            current = next_buffer

        if ready != [False, False] or free != [True, True]:
            raise AssertionError("unbalanced ready/free tokens at drain")
    return rectangles


class QbmmKPipelineTests(unittest.TestCase):
    def test_default_off_is_centralized_and_routing_forwards_env_value(self):
        env_tree = ast.parse((ROOT / "vllm_ascend/envs.py").read_text())
        env_dict = next(
            node.value for node in env_tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "env_variables"
        )
        entry = next(value for key, value in zip(env_dict.keys, env_dict.values)
                     if isinstance(key, ast.Constant)
                     and key.value == "VLLM_ASCEND_QBMM_K_PIPELINE")
        getenv = next(
            node for node in ast.walk(entry)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "getenv"
        )
        self.assertEqual(ast.literal_eval(getenv.args[1]), "0")

        route_tree = ast.parse(
            (ROOT / "vllm_ascend/_310p/quantization/methods/qbmm_custom.py").read_text()
        )
        route = next(
            node for node in route_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_qbmm_v3x"
        )
        native_call = next(
            node for node in ast.walk(route)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "quant_batch_matmul_v3_x"
        )
        keyword = next(kw for kw in native_call.keywords if kw.arg == "enable_k_pipeline")
        self.assertEqual(ast.unparse(keyword.value), "_ENABLE_K_PIPELINE")
        assignment = next(
            node for node in route_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_ENABLE_K_PIPELINE"
                    for target in node.targets)
        )
        self.assertEqual(
            ast.unparse(assignment.value), "envs.VLLM_ASCEND_QBMM_K_PIPELINE"
        )

    def test_binary_metadata_attr_order_matches_public_schema(self):
        binary = json.loads((
            QBMM_ROOT / "op_host/config/ascend310p/quant_batch_matmul_v3_x_binary.json"
        ).read_text())
        expected = ["dtype", "transpose_x1", "transpose_x2", "group_size", "enable_k_pipeline"]
        self.assertEqual(len(binary["op_list"]), 8)
        for variant in binary["op_list"]:
            self.assertEqual([attr["name"] for attr in variant["attrs"]], expected)
            self.assertEqual(variant["attrs"][-1]["dtype"], "bool")
            self.assertIsNone(variant["attrs"][-1]["value"])

        schema = (ROOT / "csrc/torch_binding.cpp").read_text()
        self.assertIn("int group_size=0, ", schema)
        self.assertIn("bool enable_k_pipeline=False) -> Tensor", schema)
        meta = (ROOT / "vllm_ascend/meta_registration.py").read_text()
        self.assertIn("enable_k_pipeline: bool = False", meta)
        op_def = (QBMM_ROOT / "op_host/quant_batch_matmul_v3_x_def.cpp").read_text()
        self.assertEqual(re.findall(r'this->Attr\("([^"]+)"\)', op_def), expected)
        self.assertIn(
            'this->Attr("enable_k_pipeline").AttrType(OPTIONAL).Bool(false)',
            op_def,
        )

    def test_ping_pong_lifetimes_and_partial_tiles(self):
        for early in (False, True):
            for k_passes in (1, 2, 3, 4, 5):
                for col_batches in (1, 2, 3):
                    base_k, base_n = 64, 32
                    # K stays K0-aligned while the final baseK pass is partial;
                    # N stays N0-aligned while the final baseN tile is partial.
                    k = (k_passes - 1) * base_k + 32
                    n = (col_batches - 1) * base_n + 16
                    rectangles = _simulate_weight_schedule(k, n, base_k, base_n, early, m_tiles=3)
                    self.assertEqual(len(rectangles), 3 * k_passes * col_batches)
                    for _, k_start, k_size, n_start, n_size in rectangles:
                        self.assertGreater(k_size, 0)
                        self.assertGreater(n_size, 0)
                        self.assertLessEqual(k_start + k_size, k)
                        self.assertLessEqual(n_start + n_size, n)

    def test_changed_enqueue_is_before_mmad_and_old_fallback_remains(self):
        source = (QBMM_ROOT / "op_kernel/quant_batch_matmul_v3_x_int8_process.h").read_text()
        start = source.index("bool nxPrefetchFired = false")
        mm = source.index("Mmad(l0c", start)
        fallback = source.index("if (!nxPrefetchFired)", mm)
        early = source[start:mm]
        self.assertIn("wChunkKPasses_ > 1 || enableKPipeline_", early)
        self.assertIn("WaitFlag<HardEvent::MTE1_MTE2>", early)
        self.assertIn("PrefetchWeightToL1", early)
        self.assertIn("SetFlag<HardEvent::MTE2_MTE1>", early)
        self.assertGreater(fallback, mm)


if __name__ == "__main__":
    unittest.main()
