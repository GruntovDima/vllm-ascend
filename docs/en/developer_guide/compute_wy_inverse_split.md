# Compute-WY inverse/split and energy gate on Ascend 310P

## HQ_TEST backport, 2026-10-19

The current kernel is backported from GDN commit
`8887fe30870b4c7ecba9651a1973925e1172d399` without merging its newer main base.
HQ_TEST keeps its existing vLLM 0.24 API, operator names, tiling ABI, DFlash,
Custom QBMM and Python integration. The compatible native-bf16 compiler guard
is also applied to the three 310P GDN operators.

The fast path retains the original `max(abs(A).sum(row)) < 0.75` admission.
Originally rejected chunks may additionally use it when the finite FP32
`sum(A*A) < 7.502183356513852`. This empirical threshold was frozen on grouped
training layers with a 20% Frobenius-norm margin, then checked on reserved
layers and 27B. It is not a universal accuracy bound. The check reuses dead
Gram scratch for vector multiplication/reduction, without another matmul or
additional UB allocation. Existing FP32 fallback/retry remains unchanged.

Packaged source SHA256 (Windows CRLF):
`600d5b7c5d1eecebf83b931d111c5247612785da6fa4931cba408ce2c9a3ef13`.
Canonical Git/LF source SHA256:
`2a4eb273ac47e3fef1fbc03ca126516845dce2c58cd46960123016e41508260d`.
The committed kernel is byte-identical to the GDN Git blob; its LF-normalized
contents match the packaged, tested source.

The exact GDN kernel was freshly built: 15 existing and 8 new native cases
passed, with five repeats. On 116 captured calls, new strict-reference
failures were zero; known baseline failure-ID sets remained 15 for 9B, 77 for
4B and zero for 27B. These per-chunk limitations are not an all-outputs PASS.
HQ_TEST full-model BS1/2/4 results are separate from those standalone checks.

The HQ_TEST backport was freshly rebuilt on CANN 9.1.0-beta.1: 76 native
regression cases passed (23 WY, 12 FwdH and 41 FwdO). All 467 Python source
files and non-GDN runtime binaries remained unchanged. Scoped Ruff, formatting,
mypy and source-contract checks passed; this is not a full CI result.

The private HQ_TEST runtime then completed a 9B W8A8 timing matrix with an
FP16 LM head, TP1, DFlash15 and Custom QBMM. Chunked prefill and prefix caching
were disabled; model length and token budget were held at 8192. Each case
used three measured waves after excluded warmups, with 2048 output tokens per
request (`ignore_eos=True`). Inputs were meaningful English continuations
with thinking disabled and exact supplied token IDs, not random IDs.

| BS | Input/request | TTFT mean, ms | TPOT mean/weighted, ms | Output tok/s | Draft acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 1178.332 | 35.346 | 27.852 | 17.27% |
| 2 | 1024 | 1344.156 | 44.404 | 43.079 | 15.32% |
| 4 | 512 | 1432.957 | 54.923 | 66.882 | 15.54% |

All 21 requests completed with exactly 2048 outputs. Token IDs were stable
across repeats of each request. Actual target and draft graph-replay counters
increased in every measured wave; WY and Custom QBMM dispatch were observed
only during excluded warmup. Observation wrappers were removed before timing.

These are offline `engine.step` cumulative-event measurements with bundled
DFlash output, not HTTP/AIS timing or a quality/equivalence test. There is no
matched baseline for an isolated GDN speedup claim. Existing integer ArgSort
warnings report AiCPU execution for multi-request cases. Other devices on the
shared host were active, but the selected NPU7 had only the experiment worker.
The private runtime did not replace the existing global installation.

The sections below retain historical evidence for the earlier inverse/split
kernel. Its old hash and measurements do not describe the current energy gate.

## Scope

Local transfer of the `ir-inverse-split` standalone candidate, 2026-10-09.
Only the compute-WY kernel header is replaced. FwdH/FwdO, bindings, tiling,
workspace sizes, build configuration and Python dispatch are not changed by
this transfer. No optimization environment variables are introduced.

The large-row-sum path for K=V=128 computes the triangular inverse once using
FP32 vector forward substitution. It then applies that inverse to W and U
using the existing hi/lo FP16 split matmul with FP32 accumulation (three
products, omitting lo*lo). FP32 substitution is column-vectorized while
preserving the update order for each destination row. Other dimensions and
the contractive path keep their existing dispatch. Finite/range guard failure
drains the candidate and recomputes the complete task through the exact NPU
fallback; this is not a CPU fallback.

The source is copied exactly, including inactive experimental helper methods.
Removing these helpers is a separate, unvalidated change.

Kernel SHA256:
`87ccce57354a30dbec2cf304db0382ca23f0b80e990696322ce76ef01f5e30ef`.

## Existing standalone evidence

The identical kernel was natively built and screened in the isolated
HQ_TEST-derived experiment, not installed into HQ_TEST itself. Environment:
CANN 9.1.0-beta.1, vLLM 0.24, torch 2.10 CPU, torch_npu 2.10.post2, one NPU 3.

- Three independent processes, 123 fixed inputs each: **369/369 passed**.
  Inputs include 11 synthetic, 64 captured 27B and 48 captured 9B inputs.
- Initial output plus five repeat checks per input; all five outputs finite.
- CPU-reference cosine >=0.999999; minimum 0.9999999126229021.
- Per-chunk W/U cosine against saved native baseline >=0.999999;
  minimum 0.9999999500177071.
- Repeated outputs were bitwise stable within each process: 369/369 cases.
  Cross-process numerical metrics agree, but raw output hashes were not saved
  for a cross-process bitwise claim. Bitwise equality to the old kernel is not
  required and was achieved for only 12/369 cases.

Mean of per-case three-process timing medians, compared with cached identical
inputs from the previous best column-vectorized kernel:

| Input group | Previous, ms | Candidate, ms | Time reduction |
| --- | ---: | ---: | ---: |
| Real 9B, T1280, 24 cases | 5.441 | 3.735 | 31.35% |
| Real 9B, T832, 24 cases | 3.444 | 2.377 | 30.99% |
| Selected real 27B, 5 timed cases | 1.955 | 1.324 | 32.26% |

The sum over 48 captured 9B calls is 146.688 ms, versus 213.236 ms for the
previous best and 364.094 ms for the original kernel. This is operator
screening, **not TTFT**. NPU-event series can include host-dispatch gaps.
Four short synthetic timed cases regress by 11.81–20.40%; universal speedup
and production acceptance are not established.

Evidence workspace: `Qwen-mtp-optimize/compute-wy-tuning-20261008/`;
raw metrics in `results/ir-inverse-split/{0,1,2}/`, paired timing report in
`results/ir-inverse-split/comparison.json`, and retained failure logs in
`evidence-refinement-final/`. These private fixtures are not bundled in this
branch. They are not required for the synthetic regression below.

## Local regression commands

CPU source-contract checks (no CANN or vLLM imports; uses the project's
existing `regex` dependency):

Local transfer verification: 7/7 source-contract checks and 34/34 existing
experiment scope/schedule checks passed. Both new Python tests parse under
Python 3.10 syntax. Kernel SHA256 matches the validated standalone source.

The WY-only checks completed on 2026-10-09: Ruff, Ruff formatting, mypy with
Python 3.10 as its target, forbidden-import and boolean-context checks,
markdownlint, typos, codespell and a Gitleaks content scan all passed. Seven
source-contract tests passed. Two additional CPU-only helper tests checked
all 15 synthetic input/reference combinations and negative controls for the
cosine scorer (shape mismatch, non-finite values and a corrupted chunk).
These helper checks do not execute or substitute the native operator.
The checks used an isolated local Linux environment; model/container
dependencies were not changed. This is not a full repository CI run.

```bash
python -m unittest discover -s tests/ut/_310p -p test_compute_wy_inverse_split_contract.py -v
```

After rebuilding/installing this branch in an appropriate 310P environment:

```bash
python -m pytest tests/e2e/nightly/310p/single_node/ops/singlecard_ops/test_compute_wy_inverse_split_310.py -v
```

The new native regression uses an independent FP64 CPU triangular solve,
per-chunk cosine >=0.999999, finite outputs and five bitwise-stable repeats.
It covers grouped heads, B1/B4, T64/128/832/1280/2048 and three decay regimes.
It must be run against the newly rebuilt operator; running an old installed
wheel is not validation of the transferred source.

## Remaining validation

This local transfer has not yet been rebuilt as the complete GDN branch or
tested end-to-end. The new native regression is added but not claimed as
executed during the local transfer. Existing standalone results are not
full-model autoregressive equivalence, simulator/mssanitizer validation or a
formal pipeline synchronization proof. No changes have been pushed.
