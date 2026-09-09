# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Standalone real-kernel tree GDN component gate on an explicitly selected NPU.

Example (physical device selected by the invoking process):
  ASCEND_RT_VISIBLE_DEVICES=4 python3 test_tree_gdn_310.py --device 0 --output result.json

The reference traverses every ancestor path with independent, existing compiled
single-token conv/GDN calls. It never uses the Python speculative fallback.
This validates the tree integration, not full-model losslessness or TPOT.
"""

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def tensor_metrics(torch, reference, observed):
    reference = reference.detach().cpu().double().reshape(-1)
    observed = observed.detach().cpu().double().reshape(-1)
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(observed).all())
    delta = (reference - observed).abs()
    denominator = torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(observed)
    cosine = float(torch.dot(reference, observed) / denominator) if finite and float(denominator) else None
    return {
        "finite": finite,
        "exact_equal": bool(torch.equal(reference, observed)),
        "different_elements": int(torch.count_nonzero(delta)),
        "elements": reference.numel(),
        "max_abs_error": float(delta.max()) if finite else None,
        "cosine": min(1.0, max(-1.0, cosine)) if cosine is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True, help="Logical NPU index after ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--sweep-shapes", action="store_true")
    parser.add_argument("--tree-kernel", action="store_true", help="Use the fused tree recurrence instead of depth calls")
    parser.add_argument("--compact-kernel", action="store_true", help="Use compact verify and accepted-path NPU replay")
    args = parser.parse_args()
    if args.device < 0 or args.repeats < 3:
        parser.error("Require a nonnegative logical device and at least three repeats")
    if args.benchmark_steps < 0:
        parser.error("benchmark-steps must be nonnegative")
    report = {
        "status": "started",
        "scope": "Real compiled single-token conv/GDN tree integration; not full-model correctness or performance",
        "contract": "FP16 recurrent checkpoint after every tree edge, matching separate autoregressive calls",
        "logical_device": args.device,
        "seed": args.seed,
        "tree_kernel": args.tree_kernel,
        "compact_kernel": args.compact_kernel,
        "dimensions": {"key_heads": 16, "value_heads": 32, "key_dim": 128, "value_dim": 128, "conv_width": 4},
        "repeats": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    record()
    try:
        import torch
        import torch_npu

        from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op, is_310p

        # Custom OPP registration must precede the first NPU initialization.
        bootstrap_custom_op_env()
        if not is_310p():
            raise RuntimeError("This component gate requires an Ascend 310P")
        torch_npu.npu.set_device(args.device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        enable_custom_op()

        from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
        from vllm_ascend._310p.ops.fla.gdn_310 import (
            npu_recurrent_gated_delta_rule_310, npu_tree_gated_delta_rule_310, npu_tree_gdn_compact_310,
        )
        from vllm_ascend._310p.spec_decode.tree import TokenTree
        from vllm_ascend._310p.spec_decode.tree_gdn import forward_tree_gdn

        device = torch.device(f"npu:{args.device}")
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
        report["operators"] = {
            "conv": str(torch.ops._C_ascend.npu_causal_conv1d_310.default._schema),
            "gdn": str(torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310.default._schema),
        }
        key_heads, value_heads, key_dim, value_dim = 16, 32, 128, 128
        qk_channels = key_heads * key_dim
        value_channels = value_heads * value_dim
        channels = qk_channels * 2 + value_channels
        conv_width, num_nodes, cache_lines = 4, 5, 9
        state_ids = torch.tensor([[7, 3, 5, 1, 6]], device=device, dtype=torch.int32)
        tree = TokenTree((0, 1, 2, 3, 4), (-1, 0, 0, 1, 1))

        def random_half(shape, generator, scale=1.0):
            return (torch.randn(shape, generator=generator) * scale).half().to(device)

        def split_qkv(mixed):
            q, k, v = mixed.split((qk_channels, qk_channels, value_channels), dim=-1)
            return (
                q.reshape(1, mixed.shape[0], key_heads, key_dim),
                k.reshape(1, mixed.shape[0], key_heads, key_dim),
                v.reshape(1, mixed.shape[0], value_heads, value_dim),
            )

        def context_for(current_tree, previous_accepted, retained=None):
            context = SimpleNamespace(
                tree=current_tree,
                num_nodes=current_tree.num_nodes,
                prefix_length=128,
                gdn_state_indices={} if retained is None else retained,
                previous_accepted_tokens=torch.tensor([previous_accepted], dtype=torch.int32, device=device),
                commits={},
            )
            context.register_commit = lambda key, callback: context.commits.setdefault(key, callback)
            return context

        def metadata_for(current_tree, previous_accepted, root_only=False, state_table=None):
            table = state_ids if state_table is None else state_table
            return SimpleNamespace(
                num_prefills=0,
                num_decodes=1 if root_only else 0,
                num_spec_decodes=0 if root_only else 1,
                num_actual_tokens=current_tree.num_nodes,
                spec_state_indices_tensor=None if root_only else table,
                non_spec_state_indices_tensor=table[0, :1] if root_only else None,
                num_accepted_tokens=None if root_only else torch.tensor([previous_accepted], dtype=torch.int32, device=device),
            )

        def run_tree(layer, current_tree, inputs, b, a, previous_accepted, retained=None, root_only=False, state_table=None):
            context = context_for(current_tree, previous_accepted, retained)
            output = inputs.new_empty((current_tree.num_nodes, value_heads, value_dim))
            forward_tree_gdn(
                layer, metadata_for(current_tree, previous_accepted, root_only, state_table), context,
                inputs, b, a, output,
                gating_fn=fused_gdn_gating_pytorch,
                recurrent_fn=npu_recurrent_gated_delta_rule_310,
                tree_recurrent_fn=npu_tree_gated_delta_rule_310 if args.tree_kernel and not args.compact_kernel else None,
                compact_recurrent_fn=npu_tree_gdn_compact_310 if args.compact_kernel else None,
            )
            return context, output

        def oracle_paths(layer, current_tree, inputs, b, a, initial_state, initial_history):
            references = []
            weights = layer.conv1d.weight[:, 0].transpose(0, 1)
            local_index = torch.zeros(1, device=device, dtype=torch.int32)
            qsl = torch.arange(2, device=device, dtype=torch.int32)
            for node in range(current_tree.num_nodes):
                state = initial_state.unsqueeze(0).clone()
                history = initial_history.unsqueeze(0).clone()
                for ancestor in current_tree.ancestor_indices(node):
                    convolved = torch.ops._C_ascend.npu_causal_conv1d_310(
                        inputs[ancestor : ancestor + 1], weights,
                        bias=layer.conv1d.bias, conv_states=history,
                        query_start_loc=None, cache_indices=local_index,
                        initial_state_mode=None, num_accepted_tokens=None,
                        activation_mode=1, pad_slot_id=-1, run_mode=1,
                    )
                    q, k, v = split_qkv(convolved)
                    g, beta = fused_gdn_gating_pytorch(
                        layer.A_log, a[ancestor : ancestor + 1], b[ancestor : ancestor + 1], layer.dt_bias,
                    )
                    output = npu_recurrent_gated_delta_rule_310(
                        q=q, k=k, v=v, g=g, beta=beta, state=state,
                        cu_seqlens=qsl, ssm_state_indices=local_index,
                        num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
                    )
                references.append((output[0, 0].clone(), state[0].clone(), history[0].clone()))
            return references

        def check_commit(layer, current_tree, context, path, references):
            context.commits[("gdn", layer.prefix)](path)
            checks = {
                "path": list(path),
                "state_checkpoints": [
                    tensor_metrics(torch, references[node][1], layer.kv_cache[1][state_ids[0, slot].long()])
                    for slot, node in enumerate(path)
                ],
                "conv_next_window": tensor_metrics(
                    torch, references[path[-1]][2], layer.kv_cache[0][7, len(path) - 1 : len(path) + conv_width - 2],
                ),
            }
            return checks

        first_repeat = None
        with torch.inference_mode():
            for repeat in range(args.repeats):
                generator = torch.Generator().manual_seed(args.seed)
                layer = SimpleNamespace(
                    prefix="tree_gdn_component.linear_attn",
                    kv_cache=(
                        random_half((cache_lines, conv_width - 2 + num_nodes, channels), generator, 0.1),
                        random_half((cache_lines, value_heads, value_dim, key_dim), generator, 0.1),
                    ),
                    conv1d=SimpleNamespace(
                        weight=random_half((channels, 1, conv_width), generator, 0.2),
                        bias=random_half((channels,), generator, 0.1),
                    ),
                    activation=True,
                    A_log=torch.zeros(value_heads, device=device),
                    dt_bias=torch.zeros(value_heads, device=device),
                    rearrange_mixed_qkv=split_qkv,
                )
                inputs = random_half((num_nodes, channels), generator)
                b = random_half((num_nodes, value_heads), generator)
                a = random_half((num_nodes, value_heads), generator)
                original_conv, original_state = tuple(value.clone() for value in layer.kv_cache)
                # Previous A=3 deliberately selects nonzero checkpoint and history offset.
                references = oracle_paths(layer, tree, inputs, b, a, original_state[5], original_conv[7, 2:5])
                context, output = run_tree(layer, tree, inputs, b, a, previous_accepted=3)
                row = {
                    "repeat": repeat + 1,
                    "node_outputs": [tensor_metrics(torch, references[node][0], output[node]) for node in range(num_nodes)],
                    "cache_unchanged_before_commit": [
                        tensor_metrics(torch, expected, observed)
                        for expected, observed in zip((original_conv, original_state), layer.kv_cache)
                    ],
                    "all_path_commits": [],
                }
                # Inspect all immutable snapshots with independently reset native
                # destinations; production TreeStepContext commits only once.
                for terminal in range(num_nodes):
                    layer.kv_cache[0].copy_(original_conv)
                    layer.kv_cache[1].copy_(original_state)
                    row["all_path_commits"].append(
                        check_commit(layer, tree, context, tree.ancestor_indices(terminal), references)
                    )

                # Non-prefix commit [0,1,4] -> root-only decode -> a full tree.
                root_tree = TokenTree((10,), (-1,))
                root_inputs = random_half((1, channels), generator)
                root_b = random_half((1, value_heads), generator)
                root_a = random_half((1, value_heads), generator)
                root_reference = oracle_paths(layer, root_tree, root_inputs, root_b, root_a, references[4][1], references[4][2])
                root_context, root_output = run_tree(
                    layer, root_tree, root_inputs, root_b, root_a, previous_accepted=3,
                    retained=context.gdn_state_indices, root_only=True,
                )
                row["root_only_output"] = tensor_metrics(torch, root_reference[0][0], root_output[0])
                row["root_only_commit"] = check_commit(layer, root_tree, root_context, (0,), root_reference)
                next_inputs = random_half((num_nodes, channels), generator)
                next_b = random_half((num_nodes, value_heads), generator)
                next_a = random_half((num_nodes, value_heads), generator)
                next_reference = oracle_paths(
                    layer, tree, next_inputs, next_b, next_a, root_reference[0][1], root_reference[0][2],
                )
                next_context, next_output = run_tree(
                    layer, tree, next_inputs, next_b, next_a, previous_accepted=1, retained=root_context.gdn_state_indices,
                )
                row["tree_after_root_only_outputs"] = [
                    tensor_metrics(torch, next_reference[node][0], next_output[node]) for node in range(num_nodes)
                ]
                row["tree_after_root_only_commit"] = check_commit(layer, tree, next_context, (0, 2), next_reference)

                # Perturb only a rejected sibling; descendants of node1 must be unchanged.
                layer.kv_cache[0].copy_(original_conv)
                layer.kv_cache[1].copy_(original_state)
                changed_inputs = inputs.clone()
                changed_inputs[2].add_(0.5)
                changed_context, changed_output = run_tree(layer, tree, changed_inputs, b, a, previous_accepted=3)
                row["sibling_isolation"] = {
                    "unaffected_nodes": [tensor_metrics(torch, output[node], changed_output[node]) for node in (0, 1, 3, 4)],
                    "perturbed_node_changed": not bool(torch.equal(output[2].cpu(), changed_output[2].cpu())),
                    "accepted_primary_commit": check_commit(layer, tree, changed_context, (0, 1, 4), references),
                }
                torch_npu.npu.synchronize()
                repeat_tensors = {
                    "first_tree_output": output.cpu(),
                    "root_output": root_output.cpu(),
                    "next_tree_output": next_output.cpu(),
                    "committed_conv": layer.kv_cache[0].cpu(),
                    "committed_recurrent": layer.kv_cache[1].cpu(),
                }
                if first_repeat is None:
                    first_repeat = repeat_tensors
                row["repeat_vs_first"] = {
                    name: tensor_metrics(torch, first_repeat[name], value) for name, value in repeat_tensors.items()
                }
                row["additional_topologies"] = []
                for parents in (
                    (-1,), (-1, 0), (-1, 0, 0), (-1, 0, 0, 1),
                    (-1, 0, 1, 0, 3), (-1, 0, 1, 2, 3),
                ):
                    current_tree = TokenTree(tuple(range(len(parents))), parents)
                    layer.kv_cache[0].copy_(original_conv)
                    layer.kv_cache[1].copy_(original_state)
                    count = current_tree.num_nodes
                    current_reference = oracle_paths(
                        layer, current_tree, inputs[:count], b[:count], a[:count],
                        original_state[5], original_conv[7, 2:5],
                    )
                    current_context, current_output = run_tree(
                        layer, current_tree, inputs[:count], b[:count], a[:count], previous_accepted=3,
                    )
                    row["additional_topologies"].append({
                        "parents": list(parents),
                        "node_outputs": [
                            tensor_metrics(torch, current_reference[node][0], current_output[node])
                            for node in range(count)
                        ],
                        "terminal_commit": check_commit(
                            layer, current_tree, current_context,
                            current_tree.ancestor_indices(count - 1), current_reference,
                        ),
                    })
                if args.sweep_shapes:
                    row["wide_topologies"] = []
                    for width in (2, 4, 8, 16):
                        count = 1 + width * 4
                        parents = (-1,) + tuple(
                            0 if i < width else 1 + (i // width - 1) * width for i in range(count - 1)
                        )
                        wide_tree = TokenTree(tuple(range(count)), parents)
                        wide_layer = SimpleNamespace(**vars(layer))
                        wide_layer.kv_cache = (
                            random_half((count + 4, count + 2, channels), generator, 0.1),
                            random_half((count + 4, value_heads, value_dim, key_dim), generator, 0.1),
                        )
                        table = [7, 3, 5, 1, 6]
                        table += [i for i in range(count + 4) if i not in table][: count - len(table)]
                        wide_ids = torch.tensor([table], device=device, dtype=torch.int32)
                        wide_inputs = random_half((count, channels), generator)
                        wide_a = random_half((count, value_heads), generator)
                        wide_b = random_half((count, value_heads), generator)
                        wide_reference = oracle_paths(
                            wide_layer, wide_tree, wide_inputs, wide_b, wide_a,
                            wide_layer.kv_cache[1][5], wide_layer.kv_cache[0][7, 2:5],
                        )
                        wide_context, wide_output = run_tree(
                            wide_layer, wide_tree, wide_inputs, wide_b, wide_a,
                            previous_accepted=3, state_table=wide_ids,
                        )
                        # Accepted path has at most five nodes; its canonical
                        # destination IDs deliberately match the original table.
                        row["wide_topologies"].append({
                            "width": width, "depth": 4, "num_nodes": count,
                            "node_outputs": [
                                tensor_metrics(torch, wide_reference[node][0], wide_output[node])
                                for node in range(count)
                            ],
                            "terminal_commit": check_commit(
                                wide_layer, wide_tree, wide_context,
                                wide_tree.ancestor_indices(count - 1), wide_reference,
                            ),
                        })
                if args.benchmark_steps:
                    timings = []
                    for iteration in range(3 + args.benchmark_steps):
                        torch_npu.npu.synchronize()
                        started = time.perf_counter_ns()
                        benchmark_context, _ = run_tree(layer, tree, inputs, b, a, previous_accepted=3)
                        benchmark_context.commits[("gdn", layer.prefix)]((0, 1, 4))
                        torch_npu.npu.synchronize()
                        elapsed_us = (time.perf_counter_ns() - started) / 1000
                        if iteration >= 3:
                            timings.append(elapsed_us)
                    row["benchmark"] = {
                        "scope": "one layer: tree conv/GDN, snapshots and accepted-path commit, synchronized wall time",
                        "warmup_steps": 3,
                        "step_us": timings,
                        "median_us": statistics.median(timings),
                    }
                report["repeats"].append(row)
                record()
                print(f"TREE_GDN_COMPONENT_REPEAT {repeat + 1} recorded", flush=True)

        def all_pass(value):
            if isinstance(value, dict):
                if "exact_equal" in value:
                    return value["finite"] and value["exact_equal"]
                return all(all_pass(item) for item in value.values())
            if isinstance(value, list):
                return all(all_pass(item) for item in value)
            if isinstance(value, bool):
                return value
            return True

        passed = all_pass(report["repeats"])
        report["status"] = "passed" if passed else "failed"
        report["gate"] = "All reported tensors finite and bitexact to independent compiled AR paths; sibling positive control changed"
        record()
        print("TREE_GDN_COMPONENT_RESULT " + report["status"], flush=True)
        return 0 if passed else 2
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        record()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
