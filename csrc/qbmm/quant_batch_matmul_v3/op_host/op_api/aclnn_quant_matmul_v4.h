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
 * \file aclnn_quant_matmul_v4.h
 * \brief aclnnQuantMatmulV4 declarations: aclnn entry point when x2 arrives in ND format.
 * \author Feodor Pisnitchenko
 */

#ifndef OP_API_INC_QUANT_MATMUL_V4
#define OP_API_INC_QUANT_MATMUL_V4

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnQuantMatmulV4GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                         const aclTensor* scale, const aclTensor* offset,
                                                         const aclTensor* pertokenScaleOptional, const aclTensor* bias,
                                                         bool transposeX1, bool transposeX2, const aclTensor* out,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnQuantMatmulV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_V4
