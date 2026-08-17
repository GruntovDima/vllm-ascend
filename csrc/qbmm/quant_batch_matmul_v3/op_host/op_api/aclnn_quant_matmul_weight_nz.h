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
 * \file aclnn_quant_matmul_weight_nz.h
 * \brief aclnnQuantMatmulWeightNz adapter declarations: entry point when x2 arrives in FRACTAL_NZ format.
 * \author Feodor Pisnitchenko
 */
#ifndef OP_API_INC_QUANT_MATMUL_WEIGHT_NZ
#define OP_API_INC_QUANT_MATMUL_WEIGHT_NZ

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Stage-one entry: compute workspace size and build the executor.
 *
 *   Dispatched by the public runtime when x2 arrives in FRACTAL_NZ
 *   format. Parameter names are fixed by the public symbol and are
 *   remapped to the kernel contract:
 *     x1Scale    -> kernel pertoken_scale
 *     x2Scale    -> kernel scale
 *     x2Offset   -> kernel offset
 *   yScale / x1Offset / yOffset / groupSize are accepted but unused.
 */
ACLNN_API aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                               const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                               const aclTensor *yScale, const aclTensor *x1Offset,
                                                               const aclTensor *x2Offset, const aclTensor *yOffset,
                                                               const aclTensor *bias, bool transposeX1,
                                                               bool transposeX2, int64_t groupSize, aclTensor *out,
                                                               uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Stage-two entry: execute the kernel on the given stream.
 */
ACLNN_API aclnnStatus aclnnQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_WEIGHT_NZ
