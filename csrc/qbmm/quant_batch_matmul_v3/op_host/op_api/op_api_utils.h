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
 * \file op_api_utils.h
 * \brief Shared aclnn adapter helpers (ND tagging, empty-placeholder, contiguous wrap) -- declarations.
 * \author Feodor Pisnitchenko
 */
#ifndef QBMV3_OP_API_UTILS_H
#define QBMV3_OP_API_UTILS_H

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"
#include "opdev/data_type_utils.h"

namespace qbmv3_opapi {

// Stamp the tensor's view and original format as ND. Returns the input pointer.
const aclTensor *SetTensorToNDFormat(const aclTensor *t);

// Promote `tensor` to a contiguous ND tensor in-place. Null input is a no-op.
aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor);

// If `tensor` is null, allocate a zero-element placeholder with the given
// dtype so the runtime does not shift following positional arguments.
const aclTensor *MakeEmptyIfNull(const aclTensor *tensor, op::DataType dtype,
                                 aclOpExecutor *executor);

}  // namespace qbmv3_opapi

#endif  // QBMV3_OP_API_UTILS_H
