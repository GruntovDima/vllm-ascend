#!/usr/bin/env bash
set -euo pipefail

# Runs a reproducible 310P target + DFlash workload. Model artefacts are inputs,
# not repository assumptions; every performance setting is also overridable.
# Example:
#   MODEL=/models/target DRAFT_MODEL=/models/dflash ./run_dflash_e2e.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/e2e-$(date +%Y%m%d-%H%M%S)}"
: "${MODEL:?Set MODEL to the target model path or identifier}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the DFlash model path or identifier}"
PROMPT_JSON="${PROMPT_JSON:-}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# A matching vLLM checkout can be supplied explicitly when the environment
# also contains another installed vLLM version.
export PYTHONPATH="${ASCEND_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${VLLM_CHECKOUT:-}" ]]; then
  export PYTHONPATH="${VLLM_CHECKOUT}:$PYTHONPATH"
fi
export VLLM_CUSTOM_QBMM="${VLLM_CUSTOM_QBMM:-1}"
if [[ -n "${LMHEAD_PRUNE_PACK:-}" ]]; then
  export VLLM_LMHEAD_PRUNE_PACK="$LMHEAD_PRUNE_PACK"
fi
export VLLM_ASCEND_DFLASH_COMPACT_GREEDY="${VLLM_ASCEND_DFLASH_COMPACT_GREEDY:-1}"
export VLLM_ASCEND_DFLASH_CONTEXT_ONLY_PREFILL="${VLLM_ASCEND_DFLASH_CONTEXT_ONLY_PREFILL:-1}"
export VLLM_ASCEND_GDN_PREFILL_HOST_COMMIT="${VLLM_ASCEND_GDN_PREFILL_HOST_COMMIT:-1}"
export VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING="${VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING:-1}"
export VLLM_ASCEND_GDN_SHARED_INPUT_QUANT="${VLLM_ASCEND_GDN_SHARED_INPUT_QUANT:-0}"
export VLLM_ASCEND_STATIC_SCALAR_QUANT="${VLLM_ASCEND_STATIC_SCALAR_QUANT:-1}"
export VLLM_ASCEND_LAST_PREFILL_MLP="${VLLM_ASCEND_LAST_PREFILL_MLP:-1}"
export VLLM_ASCEND_LAST_PREFILL_NORMS="${VLLM_ASCEND_LAST_PREFILL_NORMS:-1}"
export VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT="${VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT:-0}"
export VLLM_ASCEND_QBMM_PREFILL_ROW_ALIGNMENTS="${VLLM_ASCEND_QBMM_PREFILL_ROW_ALIGNMENTS:-24576:32,12288:64}"
export VLLM_ASCEND_PREFILL_MLP_NORM_QUANT="${VLLM_ASCEND_PREFILL_MLP_NORM_QUANT:-0}"
export VLLM_ASCEND_GEMMA_PREFILL_ADD_RMS_NORM="${VLLM_ASCEND_GEMMA_PREFILL_ADD_RMS_NORM:-0}"
export VLLM_ASCEND_GEMMA_PREFILL_ADN="${VLLM_ASCEND_GEMMA_PREFILL_ADN:-0}"
export VLLM_ASCEND_DFLASH_EARLY_PREFILL_COPY="${VLLM_ASCEND_DFLASH_EARLY_PREFILL_COPY:-0}"
export VLLM_ASCEND_DFLASH_PREFILL_DELIVERY="${VLLM_ASCEND_DFLASH_PREFILL_DELIVERY:-0}"

args=(
  --model "$MODEL"
  --draft "$DRAFT_MODEL"
  --input-len "${INPUT_LEN:-2048}"
  --output-len "${OUTPUT_LEN:-1024}"
  --spec-tokens "${SPEC_TOKENS:-15}"
  --repeats "${REPEATS:-3}"
  --warmup-output-len "${WARMUP_OUTPUT_LEN:-1}"
  --temperature "${TEMPERATURE:-0}"
  --max-batched-tokens "${MAX_BATCHED_TOKENS:-1280}"
  --max-model-len "${MAX_MODEL_LEN:-4096}"
  --graph "${GRAPH:-FULL_DECODE_ONLY}"
  --process-cpus "${PROCESS_CPUS:-}"
  --result-dir "$RESULT_DIR"
)
if [[ -f "$PROMPT_JSON" ]]; then
  args+=(--prompt-json "$PROMPT_JSON")
fi

echo "Result directory: $RESULT_DIR"
python3 "$SCRIPT_DIR/benchmark_dflash_e2e.py" "${args[@]}"
