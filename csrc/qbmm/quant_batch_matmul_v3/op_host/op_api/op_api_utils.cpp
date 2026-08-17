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
 * \file op_api_utils.cpp
 * \brief Shared aclnn adapter helpers (ND tagging, empty-placeholder, contiguous wrap).
 * \author Feodor Pisnitchenko
 */
#include "op_api_utils.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

namespace qbmv3_opapi {

const aclTensor *SetTensorToNDFormat(const aclTensor *t)
{
    if (t == nullptr) return nullptr;
    const_cast<aclTensor *>(t)->SetViewFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(t)->SetOriginalFormat(Format::FORMAT_ND);
    return t;
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) return ACLNN_SUCCESS;
    tensor = SetTensorToNDFormat(tensor);
    if (tensor == nullptr) return ACLNN_ERR_INNER_NULLPTR;
    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) return ACLNN_ERR_INNER_NULLPTR;
    return ACLNN_SUCCESS;
}

const aclTensor *MakeEmptyIfNull(const aclTensor *tensor, DataType dtype,
                                 aclOpExecutor *executor)
{
    if (tensor != nullptr) return tensor;
    auto *empty = executor->AllocTensor(dtype, Format::FORMAT_ND, Format::FORMAT_ND);
    if (empty != nullptr) {
        const_cast<aclTensor *>(empty)->SetViewShape(Shape({0}));
        const_cast<aclTensor *>(empty)->SetStorageShape(Shape({0}));
    }
    return empty;
}

}  // namespace qbmv3_opapi
