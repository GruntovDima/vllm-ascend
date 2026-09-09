// Copyright (c) 2026. Licensed under the repository LICENSE.
#pragma once
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) aclnnStatus aclnnTreeGdnCompactReplayV310GetWorkspaceSize(
    const aclTensor *key, const aclTensor *initialState, const aclTensor *records,
    const aclIntArray *parents, const aclIntArray *path, int64_t vTile,
    aclTensor *acceptedStates, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default"))) aclnnStatus aclnnTreeGdnCompactReplayV310(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
