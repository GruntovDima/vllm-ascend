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
#ifndef SWIGLU_QUANT_V310_TORCH_ADPT_H
#define SWIGLU_QUANT_V310_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_swiglu_quant_v310(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& offset)
{
    std::vector<int64_t> outShape(x.sizes().begin(), x.sizes().end() - 1);
    outShape.push_back(x.size(-1) / 2);
    at::Tensor output = at::empty(outShape, x.options().dtype(at::kChar));
    EXEC_NPU_CMD(aclnnSwigluQuantV310, x, scale, offset, output);
    return output;
}

}
#endif
