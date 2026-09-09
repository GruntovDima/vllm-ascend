// Copyright (c) 2026. Licensed under the repository LICENSE.
#pragma once
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) aclnnStatus aclnnTreeGdnCompactVerifyV310GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *beta, const aclTensor *initialState, const aclTensor *g,
    const aclIntArray *parents, float scale, int64_t vTile, aclTensor *out,
    aclTensor *records, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default"))) aclnnStatus aclnnTreeGdnCompactVerifyV310(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
