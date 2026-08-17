/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quant_matmul_weight_nz.cpp
 * \brief aclnn adapter for QuantBatchMatmulV3 when x2 arrives in FRACTAL_NZ format.
 * \author Feodor Pisnitchenko
 *
 * Dispatches to the same l0op::QuantBatchMatmulV3 kernel as
 * aclnnQuantMatmulV4, with x2 re-tagged as FRACTAL_NZ and a K-major
 * storage shape [..., K1, N1, N0=16, K0=32] that the kernel expects.
 *
 * Parameter re-mapping (WeightNz ABI -> kernel):
 *     x1Scale    -> pertokenScale
 *     x2Scale    -> scale
 *     x2Offset   -> offset
 *     yScale / x1Offset / yOffset / groupSize -> accepted but unused.
 */

#include "aclnn_quant_matmul_weight_nz.h"
#include "quant_matmul_v3.h"
#include "op_api_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
using qbmv3_opapi::SetTensorToNDFormat;
using qbmv3_opapi::MakeContiguous;
using qbmv3_opapi::MakeEmptyIfNull;

namespace {

// Block sizes for int8 FRACTAL_NZ, matching our kernel + our converter.
constexpr int64_t NZ_N0 = 16;  // penultimate dim of storage shape
constexpr int64_t NZ_K0 = 32;  // last dim of storage shape

// Mark x2 as FRACTAL_NZ with K-major storage shape [..., K1, N1, 16, 32].
//
// Caller has already laid out the physical bytes K-major (via
// torch_npu.empty_with_format + copy_memory_). We unconditionally override
// the storage-shape metadata so the kernel interprets the bytes the way
// it expects, regardless of any default that empty_with_format stamped in.
static const aclTensor *SetTensorToNZFormat(const aclTensor *input, aclOpExecutor *executor)
{
    if (input == nullptr) return nullptr;

    auto viewShape = input->GetViewShape();
    size_t viewDim = viewShape.GetDimNum();
    int64_t K = viewShape.GetDim(viewDim - 2);
    int64_t N = viewShape.GetDim(viewDim - 1);
    int64_t K1 = (K + NZ_K0 - 1) / NZ_K0;
    int64_t N1 = (N + NZ_N0 - 1) / NZ_N0;

    Shape nzShape;
    for (size_t i = 0; i + 2 < viewDim; ++i) {
        nzShape.AppendDim(viewShape.GetDim(i));
    }
    nzShape.AppendDim(K1);
    nzShape.AppendDim(N1);
    nzShape.AppendDim(NZ_N0);
    nzShape.AppendDim(NZ_K0);

    auto formatTensor = executor->CreateView(input, nzShape, input->GetViewOffset());
    CHECK_RET(formatTensor != nullptr, nullptr);
    formatTensor->SetStorageFormat(Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(Format::FORMAT_ND);
    formatTensor->SetStorageShape(nzShape);
    formatTensor->SetViewShape(viewShape);
    return formatTensor;
}

}  // namespace

// ============================================================
// aclnnQuantMatmulWeightNz -- FRACTAL_NZ entry point
// ============================================================

aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2,
    const aclTensor *x1Scale, const aclTensor *x2Scale,
    const aclTensor *yScale, const aclTensor *x1Offset,
    const aclTensor *x2Offset, const aclTensor *yOffset,
    const aclTensor *bias, bool transposeX1, bool transposeX2,
    int64_t groupSize, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulWeightNz,
                    DFX_IN(x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, bias,
                           transposeX1, transposeX2, groupSize),
                    DFX_OUT(out));

    // yScale / x1Offset / yOffset are reserved parameters -- not supported here.
    (void)yScale;
    (void)x1Offset;
    (void)yOffset;

    CHECK_RET(x1 != nullptr && x2 != nullptr && x2Scale != nullptr && out != nullptr &&
              workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    auto *execPtr = uniqueExecutor.get();

    // 1. x1: plain ND, make contiguous.
    x1 = SetTensorToNDFormat(x1);
    CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    x1 = l0op::Contiguous(x1, execPtr);
    CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 2. x2: mark as FRACTAL_NZ with K-major storage shape.
    //    DO NOT call Contiguous on it -- that would try to convert format.
    x2 = SetTensorToNZFormat(x2, execPtr);
    CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 3. Scalar/vector inputs: make contiguous in ND.
    aclnnStatus ret;
    ret = MakeContiguous(x2Scale, execPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MakeContiguous(x2Offset, execPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MakeContiguous(x1Scale, execPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MakeContiguous(bias, execPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 4. Replace null OPTIONAL inputs with zero-element placeholders so the
    //    runtime does not shift the remaining positional arguments. Dtypes
    //    match what op_def declares for each slot.
    x2Offset = MakeEmptyIfNull(x2Offset, DataType::DT_FLOAT, execPtr);
    bias = MakeEmptyIfNull(bias, DataType::DT_INT32, execPtr);
    x1Scale = MakeEmptyIfNull(x1Scale, DataType::DT_FLOAT, execPtr);

    // 5. Dispatch to kernel. Parameter re-mapping:
    //       x2Scale  -> scale          (kernel arg 3)
    //       x2Offset -> offset         (kernel arg 4)
    //       x1Scale  -> pertokenScale  (kernel arg 6)
    int64_t dtype = static_cast<int64_t>(out->GetDataType());
    const aclTensor *matmulRet = l0op::QuantBatchMatmulV3(
        x1, x2, x2Scale, x2Offset, bias, x1Scale,
        dtype, transposeX1, transposeX2, groupSize, execPtr);
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 6. ViewCopy result to user-provided out.
    auto viewCopyRet = l0op::ViewCopy(matmulRet, out, execPtr);
    CHECK_RET(viewCopyRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize,
                                      aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
