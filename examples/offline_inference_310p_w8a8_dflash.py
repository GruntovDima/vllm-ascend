#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""End-to-end 310P example: W8A8 target + DFlash speculative decoding.

Exercises the merged 310P feature set in one run:

- Qwen3.5 W8A8 target with the custom ``quant_batch_matmul_v3`` (QBM) op
  and its ZN weight layout (set ``ASCEND_310P_USE_CUSTOM_QBM=0`` to fall back
  to ``torch_npu.npu_quant_matmul``;
  ``ASCEND_310P_USE_CUSTOM_QBM=1`` is the default).
- Qwen3.5 hybrid GDN prefill through the native ``compute_wy`` /
  ``chunk_gated_delta_rule_fwd_h_vllm`` / ``chunk_fwd_o_vllm`` kernels.
- A Qwen3.5-DFlash draft model with N speculative tokens on
  ``FULL_DECODE_ONLY`` graph capture (required for DFlash on 310P).

Run (on a host with the NPU stack and both checkpoints present):

    python examples/offline_inference_310p_w8a8_dflash.py \
        --model  /home/models/Qwen3.5-9B-w8a8-mtp \
        --draft  /home/models/Qwen3.5-9B-DFlash \
        --num-speculative-tokens 15

Greedy decoding with a fixed seed keeps runs comparable between the custom
QBM path and the CANN fallback.
"""

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/models/Qwen3.5-9B-w8a8-mtp", help="Target checkpoint (W8A8).")
    parser.add_argument("--draft", default="/home/models/Qwen3.5-9B-DFlash", help="DFlash draft checkpoint.")
    parser.add_argument("--num-speculative-tokens", type=int, default=15)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from vllm import LLM, SamplingParams

    capture_sizes = [args.num_speculative_tokens + 1, 8 * (args.num_speculative_tokens + 1)]
    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=128,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
        seed=42,
        speculative_config={
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        },
    )

    prompts = [
        "Explain why the sky is blue in three short sentences.",
        "Compute 17 times 23 and explain the calculation.",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)

    # Warm-up once, then time one measured pass.
    llm.generate(prompts, sampling_params)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    for out in outputs:
        text = out.outputs[0].text.strip().replace("\n", " ")
        spec = getattr(out.outputs[0], "spec_token_ids", None)
        print(f"prompt  : {out.prompt!r}")
        print(f"output  : {text!r}")
        if spec is not None:
            print(f"drafted : {len(spec)} speculative tokens")

    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    print(f"\nmeasured pass: {total_tokens} output tokens in {elapsed:.3f}s ({total_tokens / elapsed:.1f} tok/s)")
    print("Set ASCEND_310P_USE_CUSTOM_QBM=0 to compare against the CANN npu_quant_matmul fallback.")


if __name__ == "__main__":
    main()
