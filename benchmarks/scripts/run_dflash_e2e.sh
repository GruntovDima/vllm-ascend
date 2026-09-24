#!/usr/bin/env bash
set -euo pipefail

# Reproduces the optimized 310P Qwen3.5-9B + DFlash workload. Every setting can
# be overridden from the environment, for example REPEATS=1 INPUT_LEN=1024.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-/work/dflash/e2e-$(date +%Y%m%d-%H%M%S)}"
MODEL="${MODEL:-/home/models/Qwen3.5-9B-w8a8-lmhead-mtp}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/models/Qwen3.5-9B-DFlash}"
PROMPT_JSON="${PROMPT_JSON:-/work/dflash/prompt.json}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# The benchmark image also contains a newer system vLLM.  Put the matching
# integration checkout first so a manual run cannot silently benchmark 0.27.
export PYTHONPATH="/workspace/vllm-024:/workspace/vllm-ascend:/work/dflash${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_CUSTOM_QBMM="${VLLM_CUSTOM_QBMM:-1}"
export VLLM_LMHEAD_PRUNE_PACK="${VLLM_LMHEAD_PRUNE_PACK:-/home/models/lmhead_prune_v3_int8.pt}"
export VLLM_ASCEND_DFLASH_COMPACT_GREEDY="${VLLM_ASCEND_DFLASH_COMPACT_GREEDY:-1}"
export VLLM_ASCEND_DFLASH_CONTEXT_ONLY_PREFILL="${VLLM_ASCEND_DFLASH_CONTEXT_ONLY_PREFILL:-1}"
export VLLM_ASCEND_GDN_PREFILL_HOST_COMMIT="${VLLM_ASCEND_GDN_PREFILL_HOST_COMMIT:-1}"
export VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING="${VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING:-1}"
export VLLM_ASCEND_GDN_SHARED_INPUT_QUANT="${VLLM_ASCEND_GDN_SHARED_INPUT_QUANT:-0}"
export VLLM_ASCEND_STATIC_SCALAR_QUANT="${VLLM_ASCEND_STATIC_SCALAR_QUANT:-1}"
export VLLM_ASCEND_LAST_PREFILL_MLP="${VLLM_ASCEND_LAST_PREFILL_MLP:-1}"
export VLLM_ASCEND_LAST_PREFILL_NORMS="${VLLM_ASCEND_LAST_PREFILL_NORMS:-1}"
export VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT="${VLLM_ASCEND_LAST_PREFILL_ATTN_OUTPUT:-0}"
export VLLM_ASCEND_QBMM_PREFILL_ROW_PADDING="${VLLM_ASCEND_QBMM_PREFILL_ROW_PADDING:-1}"
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
  --process-cpus "${PROCESS_CPUS:-144-191}"
  --result-dir "$RESULT_DIR"
)
if [[ -f "$PROMPT_JSON" ]]; then
  args+=(--prompt-json "$PROMPT_JSON")
fi

echo "Result directory: $RESULT_DIR"
python3 "$SCRIPT_DIR/benchmark_dflash_e2e.py" "${args[@]}"
