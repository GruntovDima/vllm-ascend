# DFlash sampling with FULL_DECODE_ONLY on 310P

The integration supports `enforce_eager=False` with
`compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
"cudagraph_capture_sizes": [16]}` for the tested batch-1 DFlash block-16
configuration (`num_speculative_tokens=15`). Prefill remains eager; target
and draft uniform decode use separate FULL ACL graphs. Verify runtime replay
counter deltas, not just successful capture, before reporting graph timings.

## Single-request stochastic sampler indexing

The Qwen3.5-9B W8A8 / FP16 DFlash test on CANN 9.1.0-beta.1 initially
captured both graphs but failed after replay in the stochastic sampler.
CANN exception reports identified two built-in int64 Add kernels:

- `[1,1] + [1,15]`: the first request's zero cumulative offset plus token
  positions. `_make_rejection_token_indices` returns positions directly for
  one request; multiple requests retain cumulative-offset addition.
- `[15] + 1`: a static maximum-output bound. Compute `max_spec_len + 1` on
  the host before copying the bound to the device.

Both changes remove redundant device work without changing acceptance
probabilities, RNG draws, rejected-token recovery or the bonus-token bound.
They do not modify the CANN toolkit, native operators, compute_wy, model
weights, quantization or graph input contracts.

## Validation and limits

Focused CPU validation:

```bash
python -m pytest -q \
  tests/ut/sample/test_rejection_token_indices.py \
  tests/ut/sample/test_rejection_sampler.py \
  tests/ut/_310p/sample/test_rejection_sampler_310.py
```

The 62 tests include exact cumulative-offset grids, single-request output
prefixes, first/middle/final rejection, all-accepted bonus handling, unchanged
RNG state, and rejection of the redundant tiny Add dispatches.

Performance must be measured separately from DEBUG mode, which synchronizes
after each graph replay. Keep prompt IDs, lengths, sampling, weights and
optimization settings identical to the eager control. Record accepted draft
tokens and emitted tokens per verification separately. A fixed-seed timing
comparison does not establish arbitrary-prompt sampling-distribution or
full-model AR equivalence. No new native simulator/sanitizer qualification is
claimed by this Python caller fix.
