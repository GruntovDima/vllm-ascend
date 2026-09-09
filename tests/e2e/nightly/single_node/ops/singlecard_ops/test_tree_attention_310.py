# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Standalone NPU tree-attention component check, not a full-model test.

Run with the task-authorized ASCEND_RT_VISIBLE_DEVICES and --device as its
logical device index. Uses existing compiled attention/cache operators only.
--output records raw numerical metrics even when a component check fails.
"""

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace


def metrics(torch, expected, observed, atol, rtol):
    a = expected.detach().cpu().double().reshape(-1)
    b = observed.detach().cpu().double().reshape(-1)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = {"finite": finite, "exact_equal": bool(torch.equal(a, b)), "elements": a.numel()}
    if finite:
        error = (a - b).abs()
        denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        result.update({
            "max_abs_error": float(error.max()),
            "rms_abs_error": float(error.square().mean().sqrt()),
            "cosine": float(torch.dot(a, b) / denominator) if float(denominator) else None,
            "within_tolerance": bool(torch.all(error <= atol + rtol * a.abs())),
        })
    else:
        result["within_tolerance"] = False
    return result


def dense_reference(torch, query, key, value, mask, scale):
    # Golden stays on CPU in FP32 and starts from the actual FP16 op inputs.
    group_size = query.shape[1] // key.shape[1]
    q = query.float().transpose(0, 1)
    k = key.float().repeat_interleave(group_size, dim=1).transpose(0, 1)
    v = value.float().repeat_interleave(group_size, dim=1).transpose(0, 1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores + mask.float().unsqueeze(0)
    return torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--sweep-shapes", action="store_true")
    parser.add_argument("--long-context-shapes", action="store_true")
    args = parser.parse_args()
    if args.device < 0 or args.repeats < 1 or min(args.atol, args.rtol) < 0:
        parser.error("device/tolerances must be nonnegative and repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "started", "logical_device": args.device,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "scope": "Synthetic full-attention masks and KV commits; no MTP proposer, GDN, or model equivalence",
        "seed": args.seed, "atol": args.atol, "rtol": args.rtol, "runs": [],
    }

    def record():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    record()
    try:
        import torch
        import torch_npu
        from vllm_ascend.utils import bootstrap_custom_op_env

        # Custom OPP registration must precede the first device initialization.
        bootstrap_custom_op_env()
        torch_npu.npu.set_device(args.device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        from vllm.v1.attention.backend import AttentionType
        # Match the worker's registration order; importing DeviceOperator
        # before the ops package is initialized creates a pre-existing cycle.
        import vllm_ascend.ops  # noqa: F401

        from vllm_ascend._310p.attention.attention_v1 import (
            AscendAttentionBackend310,
            AscendAttentionBackendImpl310,
            AscendAttentionState,
            build_tree_attention_mask,
            register_tree_cache_commit,
        )
        from vllm_ascend._310p.spec_decode.tree import TokenTree, TreeVerification
        from vllm_ascend._310p.spec_decode.tree_runtime import TreeStepContext
        from vllm_ascend.device.device_op import DeviceOperator
        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op, is_310p

        enable_custom_op()
        if not is_310p():
            raise RuntimeError("This component test requires an Ascend 310P device.")
        device = torch.device(f"npu:{args.device}")
        heads_q, heads_kv, head_dim = 16, 4, 256
        logical_block_size, kernel_block_size = (640 if args.long_context_shapes else 128), 64
        logical_table = (7, 3, 6, 1, 5, 2) if args.long_context_shapes else (3, 1)
        split_factor = logical_block_size // kernel_block_size
        kernel_table = tuple(block * split_factor + part for block in logical_table for part in range(split_factor))
        num_blocks = max(kernel_table) + 1
        cache_shape = AscendAttentionBackend310.get_kv_cache_shape(
            num_blocks, kernel_block_size, heads_kv, head_dim
        )[1:]
        block_table = torch.tensor([kernel_table], dtype=torch.int32, device=device)
        report["layout"] = {
            "query_heads": heads_q, "kv_heads": heads_kv, "head_dim": head_dim,
            "logical_block_size": logical_block_size, "kernel_block_size": kernel_block_size,
            "logical_block_table": logical_table, "expanded_kernel_block_table": kernel_table,
            "cache_shape_each": cache_shape, "cache_format": ACL_FORMAT_FRACTAL_NZ,
            "note": "D256 uses kernel block64. Long-context mode tests the logical block640 selected by tree16x4.",
        }
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
        record()

        def slots(positions):
            values = [
                kernel_table[p // kernel_block_size] * kernel_block_size + p % kernel_block_size for p in positions
            ]
            return torch.tensor(values, dtype=torch.int32, device=device)

        all_checks = []
        first_outputs = {}
        cases = [
            ("root_only", 0, (-1,), (0,)),
            ("tree_boundary_127", 127, (-1, 0, 0, 1, 1), (0, 1, 4)),
        ]
        if args.sweep_shapes:
            for width in (1, 2, 4, 8, 16):
                parents = (-1,) + tuple(0 if i < width else 1 + (i // width - 1) * width for i in range(width * 4))
                wide_tree = TokenTree(tuple(range(len(parents))), parents)
                cases.append((f"width_{width}_depth_4", 127, parents, wide_tree.ancestor_indices(len(parents) - 1)))
        if args.long_context_shapes:
            width = 16
            parents = (-1,) + tuple(0 if i < width else 1 + (i // width - 1) * width for i in range(width * 4))
            wide_tree = TokenTree(tuple(range(len(parents))), parents)
            for prefix in (1023, 1279, 2559, 3071):
                cases.append((f"wide16_prefix_{prefix}", prefix, parents,
                              wide_tree.ancestor_indices(len(parents) - 1)))
        for label, prefix, parents, path in cases:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            num_nodes = len(parents)
            span = prefix + num_nodes
            tree = TokenTree(tuple(range(num_nodes)), parents)
            query_cpu = (torch.randn((num_nodes, heads_q, head_dim), generator=generator) * 0.2).half()
            key_cpu = (torch.randn((span, heads_kv, head_dim), generator=generator) * 0.2).half()
            value_cpu = torch.randn((span, heads_kv, head_dim), generator=generator).half()
            next_query_cpu = (torch.randn((1, heads_q, head_dim), generator=generator) * 0.2).half()
            next_key_cpu = (torch.randn((1, heads_kv, head_dim), generator=generator) * 0.2).half()
            next_value_cpu = torch.randn((1, heads_kv, head_dim), generator=generator).half()
            tree_mask = build_tree_attention_mask(TreeStepContext(tree, prefix), torch.device("cpu"))[:, :span]
            golden = dense_reference(torch, query_cpu, key_cpu, value_cpu, tree_mask, head_dim ** -0.5)
            for repeat in range(args.repeats):
                context = TreeStepContext(tree, prefix)
                key_cache = torch_npu.empty_with_format(
                    size=cache_shape, dtype=torch.float16, device=device, acl_format=ACL_FORMAT_FRACTAL_NZ
                )
                value_cache = torch_npu.empty_with_format(
                    size=cache_shape, dtype=torch.float16, device=device, acl_format=ACL_FORMAT_FRACTAL_NZ
                )
                key_device, value_device = key_cpu.to(device), value_cpu.to(device)
                query_device = query_cpu.to(device)
                DeviceOperator.reshape_and_cache(
                    key=key_device, value=value_device, key_cache=key_cache, value_cache=value_cache,
                    slot_mapping=slots(range(span)),
                )
                register_tree_cache_commit(
                    context, key_device[prefix:], value_device[prefix:], key_cache, value_cache,
                    slots(range(prefix, span)),
                )
                impl = AscendAttentionBackendImpl310.__new__(AscendAttentionBackendImpl310)
                impl.attn_type, impl.sliding_window = AttentionType.DECODER, None
                impl.key_cache, impl.value_cache = key_cache, value_cache
                impl.num_heads, impl.num_kv_heads, impl.scale = heads_q, heads_kv, head_dim ** -0.5
                impl.support_compressed_mask = True  # The real tree branch must still bypass compressed masks.
                metadata = SimpleNamespace(
                    tree_mtp_context=context, num_actual_tokens=num_nodes, attn_state=AscendAttentionState.SpecDecoding,
                    block_tables=block_table, seq_lens=torch.tensor([span], dtype=torch.int32, device=device),
                )
                output = torch.empty_like(query_device)
                impl.forward_impl(query_device, None, None, (key_cache, value_cache), metadata, output)
                torch_npu.npu.synchronize()
                baseline = output.cpu().clone()
                row = {
                    "case": label, "repeat": repeat + 1, "prefix_length": prefix, "parents": parents,
                    "per_node_golden": [
                        metrics(torch, golden[i], baseline[i], args.atol, args.rtol) for i in range(num_nodes)
                    ],
                }
                if label not in first_outputs:
                    first_outputs[label] = baseline
                row["repeat_vs_first"] = metrics(torch, first_outputs[label], baseline, 0, 0)
                all_checks.extend(item["within_tolerance"] for item in row["per_node_golden"])
                all_checks.append(row["repeat_vs_first"]["exact_equal"])
                if num_nodes > 1:
                    # Only node2 and its descendants may observe this poison.
                    poisoned_value = torch.full_like(value_device[prefix + 2:prefix + 3], 1000)
                    DeviceOperator.reshape_and_cache(
                        key=key_device[prefix + 2:prefix + 3], value=poisoned_value,
                        key_cache=key_cache, value_cache=value_cache, slot_mapping=slots([prefix + 2]),
                    )
                    poisoned_output = torch.empty_like(output)
                    impl.forward_impl(query_device, None, None, (key_cache, value_cache), metadata, poisoned_output)
                    torch_npu.npu.synchronize()
                    poisoned_cpu = poisoned_output.cpu()
                    changed_values = value_cpu.clone()
                    changed_values[prefix + 2] = 1000
                    poisoned_golden = dense_reference(
                        torch, query_cpu, key_cpu, changed_values, tree_mask, head_dim ** -0.5
                    )
                    unaffected = [node for node in range(num_nodes) if 2 not in tree.ancestor_indices(node)]
                    row["foreign_sibling_invariance"] = metrics(
                        torch, baseline[unaffected], poisoned_cpu[unaffected], 0, 0
                    )
                    row["poisoned_golden"] = metrics(torch, poisoned_golden, poisoned_cpu, args.atol, args.rtol)
                    row["poisoned_own_node_changed"] = not torch.equal(baseline[2], poisoned_cpu[2])
                    all_checks.extend((
                        row["foreign_sibling_invariance"]["exact_equal"],
                        row["poisoned_golden"]["within_tolerance"], row["poisoned_own_node_changed"],
                    ))

                emitted = tuple(tree.token_ids[node] for node in path[1:]) + (999,)
                context.commit(TreeVerification(emitted, path))
                committed_span = prefix + len(path)
                DeviceOperator.reshape_and_cache(
                    key=next_key_cpu.to(device), value=next_value_cpu.to(device),
                    key_cache=key_cache, value_cache=value_cache, slot_mapping=slots([committed_span]),
                )
                # Poison one excluded future slot, without ever reading NZ directly.
                DeviceOperator.reshape_and_cache(
                    key=torch.full_like(next_key_cpu, 10).to(device),
                    value=torch.full_like(next_value_cpu, 1000).to(device),
                    key_cache=key_cache, value_cache=value_cache, slot_mapping=slots([committed_span + 1]),
                )
                canonical_key = torch.cat((key_cpu[:prefix], key_cpu[prefix:][list(path)], next_key_cpu))
                canonical_value = torch.cat((value_cpu[:prefix], value_cpu[prefix:][list(path)], next_value_cpu))
                next_golden = dense_reference(
                    torch, next_query_cpu, canonical_key, canonical_value,
                    torch.zeros((1, committed_span + 1)), impl.scale,
                )
                next_output = torch.empty((1, heads_q, head_dim), dtype=torch.float16, device=device)
                torch_npu._npu_paged_attention(
                    query=next_query_cpu.to(device), key_cache=key_cache, value_cache=value_cache,
                    num_kv_heads=heads_kv, num_heads=heads_q, scale_value=impl.scale,
                    block_table=block_table,
                    context_lens=torch.tensor([committed_span + 1], dtype=torch.int32, device=device),
                    out=next_output,
                )
                torch_npu.npu.synchronize()
                row["committed_path"] = path
                row["next_decode_with_excluded_ghost"] = metrics(torch, next_golden, next_output, args.atol, args.rtol)
                all_checks.append(row["next_decode_with_excluded_ghost"]["within_tolerance"])
                report["runs"].append(row)
                record()
                print("TREE_ATTENTION_COMPONENT " + json.dumps(row, allow_nan=False), flush=True)
        report["status"] = "component_checks_pass" if all(all_checks) else "component_checks_fail"
        report["full_model_or_kernel_port_verdict"] = "not_evaluated"
        record()
        return 0 if all(all_checks) else 2
    except Exception:
        report["status"] = "runtime_error"
        report["error"] = traceback.format_exc()
        record()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
