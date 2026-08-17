/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quant_matmul_v4.cpp
 * \brief aclnnQuantMatmulV4: aclnn entry point for QuantBatchMatmulV3 when x2 is in ND format.
 * \author Feodor Pisnitchenko
 *
 * Null OPTIONAL inputs are replaced with zero-element placeholders
 * before dispatch to keep parameter positions stable across the
 * runtime; downstream dispatch is identical to the FRACTAL_NZ path.
 */

#include "aclnn_quant_matmul_v4.h"
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

// ============================================================
// aclnnQuantMatmulV4 -- the only entry point from pytorch
// ============================================================

aclnnStatus aclnnQuantMatmulV4GetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
    const aclTensor *offset, const aclTensor *pertokenScaleOptional,
    const aclTensor *bias,
    bool transposeX1, bool transposeX2,
    const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulV4,
                    DFX_IN(x1, x2, scale, offset, pertokenScaleOptional, bias),
                    DFX_OUT(out));

    CHECK_RET(x1 != nullptr && x2 != nullptr && scale != nullptr && out != nullptr &&
              workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    auto *execPtr = uniqueExecutor.get();

    // 1. Make all non-null tensors contiguous (ND pass-through, no format conversion)
    x1 = SetTensorToNDFormat(x1);
    x2 = SetTensorToNDFormat(x2);
    x1 = l0op::Contiguous(x1, execPtr);
    CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    x2 = l0op::Contiguous(x2, execPtr);
    CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclnnStatus ret;
    ret = MakeContiguous(scale, execPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MakeContiguous(offset, execPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MakeContiguous(pertokenScaleOptional, execPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MakeContiguous(bias, execPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 2. Create zero-element placeholders for any null OPTIONAL input -- the
    //    runtime positionally packs inputs and shifts following arguments
    //    when an absent input is passed as nullptr.
    offset = MakeEmptyIfNull(offset, DataType::DT_FLOAT, execPtr);
    bias = MakeEmptyIfNull(bias, DataType::DT_INT32, execPtr);
    pertokenScaleOptional = MakeEmptyIfNull(pertokenScaleOptional, DataType::DT_FLOAT, execPtr);

    // 3. Dispatch to kernel
    int64_t dtype = static_cast<int64_t>(out->GetDataType());
    const aclTensor *matmulRet = l0op::QuantBatchMatmulV3(
        x1, x2, scale, offset, bias, pertokenScaleOptional,
        dtype, transposeX1, transposeX2, 0, execPtr);
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 4. ViewCopy result to output
    auto viewCopyRet = l0op::ViewCopy(matmulRet, out, execPtr);
    CHECK_RET(viewCopyRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV4(void *workspace, uint64_t workspaceSize,
                                aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
