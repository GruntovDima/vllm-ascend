/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QUANT_BATCH_MATMUL_V3_TORCH_ADPT_H
#define QUANT_BATCH_MATMUL_V3_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor quant_batch_matmul_v3(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &scale,
    const c10::optional<at::Tensor> &offset,
    const c10::optional<at::Tensor> &pertoken_scale,
    const c10::optional<at::Tensor> &bias,
    bool transpose_x1,
    bool transpose_x2,
    int64_t group_size)
{
    TORCH_CHECK(x1.dim() >= 2 && x2.dim() >= 2,
                "quant_batch_matmul_v3: x1 and x2 must have rank >= 2");
    // Output shape matches the op's InferShape: batch dims from x1 + [M, N].
    const int64_t m = transpose_x1 ? x1.size(-1) : x1.size(-2);
    const int64_t n = transpose_x2 ? x2.size(-2) : x2.size(-1);
    std::vector<int64_t> out_shape(x1.sizes().begin(), x1.sizes().end() - 2);
    out_shape.push_back(m);
    out_shape.push_back(n);
    // Kernel output is FP16 (op_def declares DT_FLOAT16 for all variants;
    // the aclnn layer reads the output dtype from `out`).
    at::Tensor out = at::empty(out_shape, x1.options().dtype(at::kHalf));

    const bool x2_is_nz = x2.is_privateuseone() &&
        NPUBridge::GetNpuStorageImplDesc(x2).npu_format_ ==
            ACL_FORMAT_FRACTAL_NZ;
    // Reserved slots of the WeightNz ABI (yScale / x1Offset / yOffset) are
    // default-constructed optionals so the aclnn layer receives nullptr.
    const c10::optional<at::Tensor> unused;
    if (x2_is_nz) {
        // WeightNz ABI remapping: x1Scale -> pertokenScale,
        // x2Scale -> scale, x2Offset -> offset.
        EXEC_NPU_CMD(aclnnQuantMatmulWeightNz,
                     x1,
                     x2,
                     pertoken_scale,
                     scale,
                     unused,
                     unused,
                     offset,
                     unused,
                     bias,
                     transpose_x1,
                     transpose_x2,
                     group_size,
                     out);
    } else {
        EXEC_NPU_CMD(aclnnQuantMatmulV4,
                     x1,
                     x2,
                     scale,
                     offset,
                     pertoken_scale,
                     bias,
                     transpose_x1,
                     transpose_x2,
                     out);
    }
    return out;
}

}
#endif
