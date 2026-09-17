# QBMM K-pipeline device correctness screen

This runner is for coordinator-authorized Ascend 310P validation of commit
`1b3efab25a370ae8f35f79b84ca3f3271dc0362e` plus the runner commit that
contains this file. It performs correctness checks only; it records no timing.

Before running, rebuild and install all coupled artifacts from the same commit:

- QBMM kernel and op-host package;
- generated ACLNN/op-api layer, because `enable_k_pipeline` is a new attribute;
- `_C_ascend` torch extension, because its schema also has the new attribute.

Do not combine any pre-change generated binary with the updated tiling ABI.

Produce the preserved old-custom-op reference before installing the candidate:

```bash
python csrc/qbmm/quant_batch_matmul_v3_x/tests/device/run_qbmm_k_pipeline_correctness.py \
  --variant old --json-output /tmp/qbmm-old.json
```

The runner verifies that the old schema does not expose the new boolean, so it
never passes an unsupported argument to the old binary. After installing a
complete candidate build, run both explicit new-schema variants:

```bash
python csrc/qbmm/quant_batch_matmul_v3_x/tests/device/run_qbmm_k_pipeline_correctness.py \
  --variant off --reference-json /tmp/qbmm-old.json \
  --json-output /tmp/qbmm-new-off.json
python csrc/qbmm/quant_batch_matmul_v3_x/tests/device/run_qbmm_k_pipeline_correctness.py \
  --variant on --reference-json /tmp/qbmm-old.json \
  --json-output /tmp/qbmm-new-on.json
python csrc/qbmm/quant_batch_matmul_v3_x/tests/device/run_qbmm_k_pipeline_correctness.py \
  --variant cycle --reference-json /tmp/qbmm-old.json \
  --json-output /tmp/qbmm-new-cycle.json
```

The runner invokes the raw custom op with `enable_k_pipeline=False`, then
`True`, then `False` for deterministic INT8 operands, encoded real FP32 channel
scales, and both absent and INT32 bias. It requires finite FP16 results and
bitwise equality across OFF/ON/OFF in `cycle` mode. Separate `off` and `on`
variants must match the old custom-op output hashes and deterministic input
hashes for every case. The reference is accepted only when it is a passing
old-schema capture from the exact same runner and complete case matrix.
It covers M=1/15/16 fallbacks, the actual selected `(1266,12288,4096)` down
projection, and production-size fallback cases including M=2048.

The small N=64 cases additionally use exact K0=32-blocked integer CPU
accumulation followed by exactly representable power-of-two channel scales.
Large cases deliberately use the preserved old custom op as their oracle; the
builtin is not used as an oracle for shapes where its behavior is unsuitable.

It also compares logical M=1/15 results with prefixes from zero-padded M=16
inputs. That is only a numerical padding-invariance check. It is not an OOB or
memory-sanitizer check; such safety requires a separately authorized sanitizer
run. Runtime routing is reported as `observed_pipeline_route: null` because the
current runtime exposes no supported tiling-field introspection API. Every case
records input hashes, output hashes, sample FP16 bit patterns, runner hash, and
PyTorch/torch-npu versions, and verifies that the raw op did not mutate its inputs. Cases are freed sequentially and the NPU
allocator cache is released between cases; no timing region exists.
