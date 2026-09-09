# Experimental Qwen3.5 tree-MTP on Ascend 310P

This branch packages the measured batch-one implementation, including native
tree GDN kernels. It is opt-in and experimental: component correctness does
not establish full-model lossless equivalence, and the measured TPOT benefit
is small and not a robust production speedup claim.

## Compatibility and supported configuration

- vllm-ascend base: `f69831343a1850a363f4892b1f5d5cfe8e4a051d`.
- Compatible vLLM pin: `ba07e4a48fc951300d97eb506217dd530583dea3`.
- Tested environment: aarch64, Ascend310P3, CANN 9.1.0 beta.1.
- Qwen3.5-9B ModelSlim W8A8 base with an **FP16 MTP head** in the same
  checkpoint directory. This is not a fully W8A8-quantized MTP head.
- FP16 activations and recurrent state; text only; batch one; TP/PP/DP/DCP one.
- Eager, synchronous execution. Prefix caching is disabled and
  `mamba_cache_mode="none"`; ordinary KV caching and live GDN state remain active.
- Greedy sampling only; no penalties, logprobs, constrained decoding,
  minimum-token constraint, LoRA or KV transfer.
- Comb-tree width 1, 2, 4, 8 or 16; depth 1 through 4.
  `num_speculative_tokens = width * depth`.

Do not update vLLM or move this branch onto current upstream main as part of
installation. Use the existing compatible environment. From this checkout,
with its CANN environment loaded, rebuild the editable package without
dependency resolution:

```bash
export COMPILE_CUSTOM_KERNELS=1
python3 -m pip install -e . --no-deps --no-build-isolation
```

Native sources are built through the normal custom-op build. Do not substitute
a wheel built before the tree operator registrations. Packaging this Git series
did not reinstall dependencies or rebuild the server installation.

## Reading the commit series

The series branch is `codex/tree-mtp-310p3-series`. Its compatibility base is
intentionally older than the fork's main branch; existing remote history is
not overwritten.

1. `cdf3f30c`: topology, greedy verification, activation and cache-commit contracts.
2. `b7a986d7`: native full-snapshot and compact verify/replay kernels, tiling,
   ACLNN/torch bindings and operator test harnesses.
3. `0e3b5e15`: batched masked attention, metadata and accepted-path paged KV commit.
4. `f319cb72`: tree GDN/convolution execution and accepted-state commit.
5. `c6add8c7`: MTP proposer, runner and opt-in configuration.
6. The documentation/benchmark commit: this guide and a standalone model probe.

Tests live with the implementation they cover. Generated libraries, tensors,
profiler dumps, private server configuration and local operator-porting
workbench policies are not part of the series. They remain in the test
workspace; this document summarizes their historical results.

## Execution and cache semantics

The MTP head follows the primary candidate for `depth` steps and keeps the top
`width` siblings at each step. Only the primary sibling is expanded further.
This is neither an exponential tree nor `width` complete independent chains.
For width two and depth two:

```text
root (already sampled input)
├── a
│   ├── c
│   └── d
└── b
```

Scratch order is `[root, a, b, c, d]`, parents are `[-1, 0, 0, 1, 1]`,
and relative RoPE positions are `[0, 1, 1, 2, 2]`.
There are at most 65 nodes, and one verification emits at most `depth + 1`
tokens, regardless of width.

All nodes enter one target-model forward. Full attention uses one NPU legacy
splitfuse call per layer with an ancestor-only additive mask. Attention does
not run on CPU; mask/topology construction and part of candidate selection
still involve CPU work. A dedicated device-side mask builder is not included.

Greedy verification follows direct children. For an accepted path such as
`[0, 1, 4]`, the target KV entries are gathered before writes and compacted into
the canonical chronological slots. Rejected scratch entries need not be
zero-filled: restored sequence lengths and mappings exclude them, and future
writes replace them. Tests poison stale slots and check the next normal decode.

The final emitted bonus token is the next input and is not committed yet.
GDN state and convolution history are committed along the actual accepted
path, including non-primary siblings. The drafter's cache only describes its
primary backbone; its next first pass reconstructs the accepted path from
target hidden states and shifted tokens, rather than reusing target-tree
indices against draft KV.

### Native GDN backends

The full-snapshot kernel traverses parent-indexed tree states on NPU and
exports each node's checkpoint. The compact backend exports
FP32 `[nodes, value_heads, 144]` records, retains the immutable initial state
and keys, and replays only the accepted path on NPU at commit time.

Both retain an FP16 checkpoint at every tree edge. The grouped linear-MTP
recurrence, which retains FP32 within its group, is not a bitwise oracle for
that contract. Convolution histories are prepared from ancestor inputs and
processed in a batched compiled convolution call.

`VLLM_ASCEND_TREE_GDN_COMPACT=1` enables compact dispatch for supported
`value_heads=32`, 13 through 65-node trees; other shapes use full snapshots.
The default is `0`. Tiling selects R64/R32/R16 against platform-reported UB
capacity; the deepest wide trees can require R32. No `compute_wy` or chunk-GDN
implementation was changed.

## Reproduce the model benchmark

Run from the repository root in the compatible container. Select an authorized
physical device before starting Python; the example uses device 4. The
benchmark's `--device` is the physical visibility value, whereas the component
tests below accept logical device 0.

```bash
source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

python3 examples/tree_mtp_310p_benchmark.py \
  --model /home/models/Qwen3.5-9B-w8a8-mtp/ --device 4 \
  --mtp 4 --input-tokens 1024 --max-tokens 2048 --max-model-len 4096 \
  --repeats 3 --spec-metrics --no-tree-trace \
  --gpu-memory-utilization 0.85 --safetensors-load-strategy eager \
  --output results/linear-k4.json

VLLM_ASCEND_TREE_GDN_COMPACT=1 python3 examples/tree_mtp_310p_benchmark.py \
  --model /home/models/Qwen3.5-9B-w8a8-mtp/ --device 4 \
  --mtp 16 --tree --tree-width 4 --tree-depth 4 \
  --input-tokens 1024 --max-tokens 2048 --max-model-len 4096 \
  --repeats 3 --spec-metrics --no-tree-trace \
  --gpu-memory-utilization 0.85 --safetensors-load-strategy eager \
  --output results/tree-w4-k4.json
```

Use new output filenames: the probe writes its report as it progresses.
For the original grid, run depth `k=1..4` and width `1,2,4,8,16` with
`--mtp k*width`; the linear controls use `--mtp k` without `--tree`.
The script is self-contained; no parent-directory probe or private SSH wrapper
is needed. Its prompt construction and engine arguments are preserved from
the measured probe.

The probe fixes input to 1024 tokenizer IDs, ignores EOS to produce 2048
tokens, warms up with eight output tokens, and saves token IDs, prompt hashes,
request timestamps and speculative counter deltas. For the measured model,
the prompt SHA-256 is
`51c0e67b7cad2db9cb9c41ae0e1f2eb1b0f77da5f1b22bed3d1cdbd2db439b74`.

Decode TPOT is `(last_token_timestamp - first_token_timestamp) / 2047`;
prefill is excluded. Wall time per output token including prefill is a separate
field. This is an offline LLM benchmark, not an HTTP serving benchmark.
FULL_DECODE_ONLY graph capture is not supported for the current tree path.

## Latest historical model measurements

Batch one, input 1024, output 2048, eager, same checkpoint/prompt and dependency
pins. Compact dispatch was enabled with full-snapshot fallback. Values are
TPOT in milliseconds, lower is better.

| Depth k | Linear MTP | Width 1 | Width 2 | Width 4 | Width 8 | Width 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 87.40 | 91.27 | 89.39 | 87.61 | 88.74 | 91.49 |
| 2 | 75.68 | 78.36 | 73.04 | 70.81 | 76.08 | 79.62 |
| 3 | 70.31 | 75.62 | 73.19 | 70.01 | 70.48 | 80.41 |
| 4 | 67.98 | 76.72 | 69.88 | 66.45 | 75.39 | 84.05 |

Screening has one sample per cell except linear k3, linear k4 and tree w4/k4,
which have three samples and are shown as medians:

- Linear k3: 67.94, 71.91, 70.31 ms.
- Linear k4: 67.98, 66.31, 71.31 ms.
- Tree w4/k4: 64.33, 67.71, 66.45 ms.

The best tree median is 2.25% below linear k4. The ranges overlap, and output
IDs differ even between repeated linear runs. This is not evidence of a
stable speedup on arbitrary workloads or full-model AR equivalence.

At k4, the counter-derived emitted tokens per verification rise from 3.133
(linear) to 3.585 (tree w4), while average draft/verify/commit cycle time rises
from 212.77 to 239.41 ms. This metric is
`1 + accepted_tokens / verification_steps`, not acceptance divided by every
candidate in the tree.

Do not attribute the entire improvement to compact records. Matched R64
kernel measurements were 511.70 vs 548.71 microseconds at k3/w4 (compact
slower), and 704.39 vs 697.33 at k4/w4 (about 1% faster). R32-to-R64 tiling
was a separate important improvement. Byte-count reductions are not measured
bandwidth or end-to-end TPOT speedups.

## Validation and open limits

Publication-time CPU checks passed 72 spec-decode tests and 18 attention tests.
The standalone C++ host test also passed 26 valid and seven invalid
topologies with DFS frame and UB-layout checks. Runtime source hashes matched
the tested server copy after line-ending normalization. The full repository
format check could not run because `pre-commit` was absent; dependencies were
not installed just for publication.

```bash
python3 -B -m unittest discover -s tests/ut/_310p/spec_decode -p 'test_tree*.py'
python3 -B -m unittest discover -s tests/ut/_310p/attention -p 'test_tree*.py'
g++ -std=c++17 -O2 \
  csrc/attention/tree_gated_delta_rule_v310/tests/ut/test_tree_plan.cpp \
  -o csrc/attention/tree_gated_delta_rule_v310/tests/ut/test_tree_plan
csrc/attention/tree_gated_delta_rule_v310/tests/ut/test_tree_plan
```

Native component tests require explicitly selected NPU visibility and the
built package. `-P` avoids shadowing the runtime's triton package with a test
directory. Run full and compact GDN separately:

```bash
python3 -P tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_gdn_310.py \
  --device 0 --tree-kernel --sweep-shapes --output results/gdn-full.json
python3 -P tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_gdn_310.py \
  --device 0 --compact-kernel --sweep-shapes --output results/gdn-compact.json
python3 -P tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_attention_310.py \
  --device 0 --sweep-shapes --long-context-shapes --output results/attention.json
```

Previous kernel/component validation covered nine forms and 148 accepted
paths with 866 bit-exact clean/poison comparisons. Sanitizer checks covered
all eight cores individually for memcheck and all blocks for synccheck:
81 invocations, 486 kernel executions, no reported errors. These are
historical checks, not newly rerun NPU tests during Git packaging.

The per-pipe timing repeatability gate with a one-percent tolerance did
**not** pass on the nine forms, including the unchanged full-snapshot control;
the cause remains unresolved. Correctness checks do not override that failed
performance gate.

Full-model numerical differences from AR and repeated-run token differences
remain open. Prefix-state reuse, stochastic tree sampling, asynchronous or
graph execution, and device-side mask/selection construction are not
implemented. This branch should remain opt-in until those contracts and
performance behavior are addressed.
