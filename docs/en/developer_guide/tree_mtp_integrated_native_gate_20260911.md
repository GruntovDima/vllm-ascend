# Tree-MTP integrated native prerequisite audit (2026-09-11)

This note records a read-only audit of the integrated 310P tree-MTP series. It
does not change native kernels, their ABI, synchronization, tiling policy, or
the authorized performance gate.

- Tree-series source HEAD: `02c0731f5643f157edc6b28194b4dec75b95e3fe`.
- Integrated checkout HEAD at audit time:
  `e38045b076c0191bf7e9a4c5e0ad1af8830b3072`.
- Intended environment: aarch64 Ascend310P3, CANN 9.1.0 beta.1, vLLM 0.27.1;
  the source validation used vLLM commit
  `ba07e4a48fc951300d97eb506217dd530583dea3`.

## Current requested-grid status

The comb tree has `nodes = 1 + width * depth`. For requested depths 2 through
6 and widths 1, 2, 4, 8 and 16, 15 of 25 configurations are supported by the
unchanged source and 10 are blocked before they can be treated as valid native
tests.

| Depth | Nodes at width 1 / 2 / 4 / 8 / 16 | Current status |
| ---: | --- | --- |
| 2 | 3 / 5 / 9 / 17 / 33 | supported |
| 3 | 4 / 7 / 13 / 25 / 49 | supported |
| 4 | 5 / 9 / 17 / 33 / 65 | supported |
| 5 | 6 / 11 / 21 / 41 / 81 | blocked by depth; width 16 also exceeds node cap |
| 6 | 7 / 13 / 25 / 49 / 97 | blocked by depth; width 16 also exceeds node cap |

The first guard is the Python configuration check in
[`tree_runtime.py`](../../../vllm_ascend/_310p/spec_decode/tree_runtime.py#L44-L45).
The native contract independently fixes `MAX_NODES=65` and `MAX_DEPTH=4` in
[`tree_gated_delta_rule_v310_tiling_data.h`](../../../csrc/attention/tree_gated_delta_rule_v310/op_kernel/tree_gated_delta_rule_v310_tiling_data.h#L6-L7),
and validates them in
[`tree_gdn_plan.h`](../../../csrc/attention/tree_gated_delta_rule_v310/op_host/tree_gdn_plan.h#L13-L48).
Compact replay is additionally limited to a five-node accepted path by
[`tree_gdn_compact_310_torch_adpt.h`](../../../csrc/attention/tree_gated_delta_rule_v310/tree_gdn_compact_310_torch_adpt.h#L47-L74)
and
[`aclnn_tree_gdn_compact_replay_v310.cpp`](../../../csrc/attention/tree_gated_delta_rule_v310/op_host/op_api/aclnn_tree_gdn_compact_replay_v310.cpp#L49-L57).

Depth 5 or 6 rejection is therefore an expected capacity result, not evidence
that the integrated depth-2-through-4 implementation is incorrect.

## Historical performance gate versus current testing

Historical evidence was inspected read-only under
`tree-mtp-work/csrc/attention/tree_gated_delta_rule_v310/workbench/`; that
untracked workbench and its generated datasets were intentionally not copied
into this checkout. In particular:

- `notes/timing_variance_policy.yaml` authorizes three runs and at most 1%
  absolute deviation from the mean for S, V, M, MTE1, MTE2, MTE3, total cycles
  and duration.
- `candidates/compact_checkpoint/final_performance_v1/summary.json` records
  `experimental_T1_timing_gate_failed`; all nine profiled forms failed at least
  one timing metric.
- The isolated unchanged full-snapshot root control also failed: duration
  deviation was 6.748%, total-cycle deviation 7.080%, and MTE2 deviation
  23.049%.
- The same evidence reports bit-exact outputs and minimum cosine
  `0.9999999999999998`.

Consequently, the historical result is a formal blocker for T1 performance
acceptance, cost-model calibration, and promotion of a new native extension
under the unchanged policy. It is not a correctness failure, a compiler
blocker, or proof of a compact-specific race: the unchanged control failed the
same repeatability rule. Historical passes also do not substitute for tests of
this integration. The 15 currently supported configurations may be tested now;
their results must be reported separately from the historical gate. This audit
does not relax or reinterpret the 1% policy.

## Required coordinated change before depth 5/6

Supporting depth 6 and 97 nodes requires one coherent native-and-Python change,
not a Python-only guard relaxation:

1. Raise the shared native constants to 97 nodes and six draft levels, and the
   accepted replay path to seven nodes.
2. Update the hardcoded limits in both Torch adapters, the compact ACLNN replay
   API, native test runner, Python configuration, and compact dispatch predicate
   in [`tree_gdn.py`](../../../vllm_ascend/_310p/spec_decode/tree_gdn.py#L20-L22).
3. Rebuild and reinstall host tiling, all three device kernels, ACLNN op-api,
   and the `_C_ascend` Torch extension together. The embedded `order[]` and
   `depth[]` arrays make this an ABI change: the base tiling payload grows from
   648 to 904 bytes and the replay plan from 672 to 936 bytes.
4. Add depth-5/6 and N81/N97 coverage to host topology, ICPU, native
   functional/out, compact verify/replay, accepted-path, attention, sampler,
   sanitizer, graph-replay, and full-model tests. Existing committed sweeps stop
   at depth 4 or 65 rows.
5. Re-run the unchanged numerical, memory, synchronization and 1% timing gates
   before claiming native acceptance.

For Ascend310P3's recorded 262,144-byte UB, the current planner formula predicts
that N97/depth6/Hv32 does not fit R64 (325,600 bytes), but fits R32 (185,568
bytes) and R16 (115,552 bytes). This is only a host-side prerequisite estimate;
the CANN tiling dump and device gates must confirm it after a coherent change.
Full snapshots would also retain 97 MiB per GDN layer, whereas compact records
are about 1.71 MiB per layer before accepted-path replay, so the compact
dispatch limit must not be overlooked.

## Tests available without extending native limits

After rebuilding the integrated package with `COMPILE_CUSTOM_KERNELS=1`, the
self-contained component gates are:

```bash
python3 -P tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_gdn_310.py \
  --device 0 --tree-kernel --sweep-shapes --output results/gdn-full.json
python3 -P tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_gdn_310.py \
  --device 0 --compact-kernel --sweep-shapes --output results/gdn-compact.json
python3 -P tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_attention_310.py \
  --device 0 --sweep-shapes --long-context-shapes --output results/attention.json
python3 -P tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_sampling_310.py \
  --device 0 --output results/sampling.json
```

These commands require an explicitly selected physical NPU before Python is
started. The GDN `--sweep-shapes` implementation does not itself cover the full
15-case model grid, so the supported depth/width combinations still require the
standalone model benchmark described in
[`experimental_tree_mtp_310p.md`](experimental_tree_mtp_310p.md#reproduce-the-model-benchmark).
There is no valid depth-5/6 model invocation with the unchanged source.
