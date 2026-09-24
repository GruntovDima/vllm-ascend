# Standalone QBM optimization branch, before unfinished tiling tuning

Branch: `codex/opt-qbm`. Extracted on 2026-09-24.

Common historical base: `2ee50f3847e4873e61d280911b2a24d85f5278d6`,
before the local 310P DFlash series. This is not a rebase to current upstream.
No dependency on the sibling GDN branch is introduced.

## Included source commits

| Source | Scope |
| --- | --- |
| `bb717885f` | QuantBatchMatmulV3X, build integration, bindings, W8A8 routing |
| `b0f41d00a` | Reject unsupported odd K tails and use builtin fallback |
| `3e3eab019` | Required adapter include in Torch bindings |
| `0e284fd28` (QBM files only) | Explicit post-load registration and errors |
| `4d5823b75` | Optional padding of two measured prefill projection shapes |
| `1b3efab25` | Default-off single-pass weight-prefetch K-pipeline experiment |
| `c06388df9` | Device numerical-screen runner and its CPU/contract tests |

Original authors and source hashes are retained in signed-off cherry-picks.
The mixed `0e284fd28` commit is represented by a separately attributed
extraction containing only the QBM registration fix and its tests.
Conflicting build lists/bindings/environment entries retain only QBM additions.

## Options and scope

- `VLLM_CUSTOM_QBMM=1`: existing opt-in custom-QBM routing.
- `VLLM_ASCEND_QBMM_PREFILL_ROW_PADDING=1`: model-side zero padding of
  `(M,K,N)=(782,4096,24576)` to M800 and `(782,4096,12288)` to M832,
  followed by slicing back to the logical output. No scheduler chunk change.
- `VLLM_ASCEND_QBMM_K_PIPELINE=1`: experimental guarded single-pass prefetch;
  default zero. Inclusion in this branch does not promote it to the selected
  production configuration.

The latter two options default to zero. Weight arithmetic and unsupported
shape fallbacks are retained. Native ABI components must be rebuilt together:
the K-pipeline series adds an optional operator attribute and tiling field.

## Explicitly excluded tiling experiment

The uncommitted 2026-09-18 changes in `qbmm-tiling-ascend` are NOT included:
no calibrated 7840-MAC cost, no `UsePrefillTileScale` policy, and no new
full-channel scale-buffer disabling allowlist. The entire `csrc/qbmm` tree
and QBM Python wrapper match `c06388df9` from the pre-tuning integration.
Existing original tiling and the K-pipeline guard are intentionally retained.

## Verification and limitations

- 19 dependency-light Python CPU/contract tests pass.
- Both C++ CPU tests pass: K-pipeline policy and production-picker reachability.
- Python source syntax and diff whitespace checks pass.
- Registration tests are preserved, but require the plugin test environment;
  local vLLM/torch_npu/pytest dependencies are unavailable.
- No new CANN build, NPU execution, sanitizer run or end-to-end benchmark was
  performed for this extracted branch. Existing integration measurements do
  not certify the standalone build or its latency.

Excluded: DFlash, GDN, lm-head pruning/routing, scalar-quant parameter changes,
MLP/norm fusion, last-layer prefill selection and early delivery.
