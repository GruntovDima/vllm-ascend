# Requested tree-GDN operator

Tree-indexed gated delta recurrence. A separate vector-only inference operator; convolution, gating preparation, normalization and compute_wy are not modified.

- q: fp16 ND [N,Hk,K].
- k: fp16 ND [N,Hk,K].
- v: fp16 ND [N,Hv,V].
- beta: fp16 ND [N,Hv].
- initial_state: fp16 ND [Hv,V,K].
- g: fp32 ND [N,Hv].
- out: FP16 ND [N,Hv,V].
- snapshots: FP16 ND [N,Hv,V,K].

- hkey = floor(h / (Hv/Hk)); P = initial_state[h] if parents[n] == -1 else snapshots[parents[n], h]
- D[v,k] = fp32(P[v,k]) * exp_fp32(g[n,h])
- r[v] = ReduceSum(D[v,k] * fp32(k[n,hkey,k]), axis=K); pair k and k+64, then WholeReduceSum64
- delta[v] = (fp32(v[n,h,v]) - r[v]) * fp32(beta[n,h])
- S[v,k] = fma(delta[v], fp32(k[n,hkey,k]), D[v,k])
- o[v] = ReduceSum(S[v,k] * (fp32(q[n,hkey,k]) * scale), axis=K); same reduction order
- out[n,h,v] = cast_fp16_CAST_NONE(o[v]); snapshots[n,h,v,k] = cast_fp16_CAST_NONE(S[v,k])

- "1 <= N <= 65"
- "K == 128 and V == 128"
- "1 <= Hk <= Hv <= 32 and Hv % Hk == 0"
- "parents has N entries; parents[0] == -1; 0 <= parents[n] < n for n > 0"
- "max tree depth <= 4; root depth is 0"
- "q and k are already L2-normalized; scale is finite"
- "All tensor inputs contiguous ND on the same NPU; output buffers are disjoint from inputs"

Keep old per-node implementation as golden. Do not modify compute_wy or original linear native operator. Support comb width16 depth4 (65 nodes) and graph capture without NPU->CPU reads. This records the user's tree-MTP request and existing numerical behavior, not a new approximation.
