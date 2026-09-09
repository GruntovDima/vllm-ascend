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
- Greedy or stochastic target sampling with temperature, top-k and top-p;
  seeded and unseeded requests. Proposal construction remains deterministic.
- No penalties, logprobs, constrained/custom logits processors, minimum-token
  constraint, thinking-token budget, LoRA or KV transfer. Keep `min_p=0`.
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
6. `8b9e7503`: this guide and a standalone model probe.
7. `ece179a7`: stochastic target verification, RNG handling and distribution tests.
8. `c0622f7e`: sampling controls and stochastic validation results.
9. `ca664cff`: shared per-step attention inputs and cache-lifetime checks.
10. `94cbc178`: deterministic rank-key draft top-k and CPU/NPU selector tests.

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
splitfuse call per layer with an ancestor-only additive mask. The prepared
FP16/NZ mask and physical lengths are shared across attention layers within
one `TreeStepContext`, keyed by device, prefix length and topology. They are
released at commit, never shared across requests; layer-local KV tensors and
block tables are not cached here. Attention and numerical draft selection run
on NPU. Building topology and the initial mask still involves CPU work; a
dedicated device-side mask builder is not included.

Verification follows the sampled target token through direct children. For an
accepted path such as `[0, 1, 4]`, the target KV entries are gathered before
writes and compacted into the canonical chronological slots. Rejected scratch
entries need not be zero-filled: restored sequence lengths and mappings exclude them, and future
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

### Stochastic target verification

The target model samples one token for every node using the existing
`AscendSampler310`, with request temperature/top-k/top-p broadcast over the
node rows. It samples from the full processed vocabulary, not just from the
children proposed by MTP. If the sampled token is a child, traversal continues
there; otherwise that exact sample becomes the bonus token and traversal stops.
Cache commit is identical to the greedy path, including non-primary siblings.

This is direct target sampling, not draft-probability rejection sampling.
Candidate construction does not consume target random numbers. With independent
draws per node, reaching a node depends only on ancestor draws, so the draw at
that node still has its target conditional distribution. CPU tests exhaustively
enumerate every outcome of small history-dependent models and compare the
resulting output-sequence probabilities with serial AR, including fallback
outside the tree and output-budget truncation. That algebraic check does not
resolve the existing numerical differences between tree and AR model forwards.

Seeded requests share one advancing RNG across all node rows; each row gets
a distinct draw. The same seed is not restarted at every node. The existing
310P sampler creates scalar uniforms on CPU; filtering, softmax and inverse-CDF
selection remain on NPU. There is no vocabulary-sized CPU sampling transfer.
Unvisited nodes also consume random draws. Fixed seeds are repeatable for fixed
logits and topology, but do not promise token-for-token identity with serial AR
or another tree width. Greedy mode does not consume these random draws.

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
the measured probe; the new sampling arguments default to the original greedy
settings. For stochastic runs add the same arguments to both tree and linear
commands:

```bash
--temperature 1.0 --top-k 50 --top-p 0.9 --seed 42
```

Use `--seed -1` for unseeded request sampling. Reports record the exact sampling
parameters. The historical TPOT table below is greedy, not a measurement of
the newly added stochastic mode.

The probe fixes input to 1024 tokenizer IDs, ignores EOS to produce 2048
tokens, warms up with eight output tokens, and saves token IDs, prompt hashes,
request timestamps and speculative counter deltas. For the measured model,
the prompt SHA-256 is
`51c0e67b7cad2db9cb9c41ae0e1f2eb1b0f77da5f1b22bed3d1cdbd2db439b74`.

Decode TPOT is `(last_token_timestamp - first_token_timestamp) / 2047`;
prefill is excluded. Wall time per output token including prefill is a separate
field. This is an offline LLM benchmark, not an HTTP serving benchmark.
FULL_DECODE_ONLY graph capture is not supported for the current tree path.

## Stochastic model validation, 2026-09-09

Both Qwen3.5-9B runs completed 1,024 input and 2,048 output tokens on physical
NPU 4, sequentially, with the compatible pins above. Sampling was temperature
1, top-k 50, top-p 0.9, request seed 42, no penalties and ignored EOS. The
prompt hashes matched. Tree used depth 4, width 4 and compact GDN; the control
used ordinary linear MTP with four speculative tokens. Both used eager,
synchronous batch-one execution with prefix caching disabled. Each engine had
an eight-token warmup followed by **one** measured request.

| Mode | Decode TPOT (ms) | Tokens per verification | Verification steps | Accepted draft tokens |
| --- | ---: | ---: | ---: | ---: |
| Linear MTP k4 | 82.0728 | 2.68283 | 763 | 1284 |
| Tree MTP k4/w4 | 77.8549 | 3.37232 | 607 | 1440 |

Tree TPOT was 5.14% lower and tokens per verification were 25.70% higher in
this pair. Counter-derived average cycle time was 262.55 ms for tree versus
220.19 ms for linear: additional coverage still comes with additional work.
This single-prompt, single-run comparison is a functionality/performance
screen, not a stable speedup claim or an accuracy evaluation. Generated
sequences differ; the shared seed does not force the same RNG consumption.

Raw reports are retained in the test workspace under
`stochastic_validation_20260909_0944/{component,tree,linear}.json`.
The sampling extension did not alter native kernels, `compute_wy`, model
weights or dependencies. The previous native timing-repeatability limitation
below remains open. Full-model distributional equivalence to AR has not been
established by these runs.

## Runtime top-k and shared-mask optimization, 2026-09-09

Draft widths greater than one use two selections and score ranking instead of
one full-vocabulary reduction loop per sibling. The first top-k supplies score
boundaries; `searchsorted(right=True)` assigns equal scores equal ranks. A
second top-k selects the unique keys `rank * vocab_size - token_id`. Distinct
selected score groups have ranks at least one apart, and a vocabulary-wide
token-ID difference cannot reverse that ordering. This also repairs ties that
cross the selection boundary, without adding an epsilon to logits or consuming
RNG. The fast path checks that `(width + 1) * vocab_size <= 2**24`, keeping all
FP32 key arithmetic exact; unusual dtypes/sizes retain the reference algorithm.
There is no new per-level host/device synchronization.

NaN detection uses IEEE magnitude bits: the pinned 310P floating `isnan` and
self-inequality paths also classified infinities as NaNs in the probe.
`nan_to_num` was unsupported. Widths greater than one sort NaNs before +inf
and break ties by token ID. Width one deliberately keeps native `argmax`;
its existing nonfinite behavior is not CPU-argmax-equivalent. The NPU test
records those differences explicitly, separately from the stable-sort golden
for the new algorithm. CANN and torch_npu were not modified.

Same-device component times, milliseconds per draft level, vocabulary 248320,
FP32, median of three groups of 15 calls after warmup:

| Width | Previous reductions | Ranked top-k |
| --- | ---: | ---: |
| 1 | 0.0514 | 0.0508 |
| 2 | 1.6394 | 1.9376 |
| 4 | 3.1717 | 1.9600 |
| 8 | 6.2240 | 2.0084 |
| 16 | 12.5341 | 2.0238 |

Width two regresses in this composition; this is not an improvement at every
width. FP16 was also tested: width four changed from 3.1056 to 2.1358 ms.
Two exact int64-key alternatives were screened but were slower than the
ranked approach. These are compositions of existing device operations, not a
new fused AscendC top-k kernel.

For a synthetic eight-attention-call step at width/depth 4 and prefix 2048,
rebuilding inputs before every call took 9.0670 ms; one build plus seven shared
uses took 2.1135 ms, with bit-identical outputs. This reuses the same synthetic
KV tensors across the eight calls and is a component comparison, not a model
layer-time decomposition. Wall timings include dispatch and device completion;
isolated savings cannot simply be added to predict end-to-end TPOT.

Full-model screen: batch one, input 1024/output 2048, eager, temperature 1,
top-k 50/top-p 0.9/seed 42, same physical NPU and prompt, eight-token warmup,
one measured request per mode. The tree configuration is depth 4/width 4.

| Mode | TPOT (ms) | Tokens/verification | Verification steps |
| --- | ---: | ---: | ---: |
| No MTP | 105.9421 | 1 | n/a |
| Tree before | 82.8759 | 3.10152 | 660 |
| Tree after top-k + shared inputs | 81.0306 | 3.09682 | 661 |

TPOT decreased 2.23% in this pair; the counter-derived mean cycle changed
from 257.04 to 250.94 ms. The new tree run was 23.51% below the no-MTP control.
Generated sequences differ, despite the same seed; these single requests do
not establish a stable speedup or lossless full-model equivalence. Do not mix
this comparison with the older stochastic linear-MTP pair or greedy grid.
Neither the sampler nor GDN/`compute_wy`, dependencies or weights changed.

Validation: 117 CPU tests; 94 NPU top-k cases with three repetitions (74 use
CPU stable-sort golden, 20 preserve native argmax); 33 attention cases covering
shared-input identity, unchanged output, sibling poison and next decode after
KV commit; 13 additional mask-reuse equivalence checks. `git diff --check`
passed. Full `format.sh ci` was unavailable because `pre-commit` is not
installed; dependencies were not installed to bypass that limitation.

The standalone NPU selector check is
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_topk_310.py`.
It accepts `--device`, a new `--output` JSON path, and optionally
`--baseline-source` pointing to the previous proposer file for paired timings.
Raw reports remain in the test workspace under
`runtime_opt_stage1_20260909/{before_tree,no_mtp,after_tree,topk_v2,attention_v2,attention_perf}.json`.

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

The stochastic extension passed 110 CPU tests (87 spec-decode, 18 attention,
five existing sampler tests). On physical NPU 4, all 20 native sampler cases
passed: 1/5/17/65 node rows across greedy and four stochastic configurations,
including temperature 1, top-k 50, top-p 0.9. Every case matched serial calls
to the same sampler on fixed logits and reproduced its seeded result.
An 8,192-draw check of target probabilities `[0, 0.1, 0.2, 0.7]` observed
`[0, 0.1015625, 0.1982421875, 0.7001953125]`; the zero-mass token was never
sampled. Unseeded sampling also passed. These component checks do not prove
full-model distributional equivalence to AR.

```bash
python3 -B -m unittest discover -s tests/ut/_310p/spec_decode -p 'test_tree*.py'
python3 -B -m unittest discover -s tests/ut/_310p/attention -p 'test_tree*.py'
python3 -B tests/ut/_310p/sample/test_sampler_310.py
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
python3 -P tests/e2e/nightly/single_node/ops/singlecard_ops/test_tree_sampling_310.py \
  --device 0 --output results/sampling.json
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
remain open. Prefix-state reuse, asynchronous or
graph execution, and device-side mask/selection construction are not
implemented. This branch should remain opt-in until those contracts and
performance behavior are addressed.
