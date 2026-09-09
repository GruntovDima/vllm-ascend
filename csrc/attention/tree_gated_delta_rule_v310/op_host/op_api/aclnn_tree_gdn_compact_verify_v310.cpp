// Copyright (c) 2026. Licensed under the repository LICENSE.
#include "aclnn_tree_gdn_compact_verify_v310.h"
#include "../tree_gdn_plan.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <initializer_list>
#include <limits>

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(TreeGdnCompactVerifyV310);

static aclnnStatus Launch(const aclTensor *query, const aclTensor *key,
    const aclTensor *value, const aclTensor *beta, const aclTensor *initialState,
    const aclTensor *g, const aclIntArray *parents, float scale, int64_t vTile,
    aclTensor *out, aclTensor *records, aclOpExecutor *executor)
{
    // Direct outputs: no intermediate full-state buffer and no ViewCopy of records.
    return ADD_TO_LAUNCHER_LIST_AICORE(TreeGdnCompactVerifyV310,
        OP_INPUT(query, key, value, beta, initialState, g),
        OP_OUTPUT(out, records), OP_ATTR(parents, scale, vTile));
}
} // namespace l0op

namespace {
bool ShapeIs(const aclTensor *tensor, std::initializer_list<int64_t> dims)
{
    const auto &shape = tensor->GetViewShape();
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (auto dim : dims) {
        if (shape.GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}
} // namespace

extern "C" aclnnStatus aclnnTreeGdnCompactVerifyV310GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *beta, const aclTensor *initialState, const aclTensor *g,
    const aclIntArray *parents, float scale, int64_t vTile, aclTensor *out,
    aclTensor *records, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // Validate memory ownership before L2_DFX_PHASE_1: a warm executor cache
    // can return early for identical shapes with different tensor addresses.
    // In particular, an out= input alias must not reuse a non-aliasing plan.
    if (workspaceSize == nullptr || executor == nullptr || parents == nullptr ||
        !std::isfinite(scale) || (vTile != 0 && vTile != 16 && vTile != 32 && vTile != 64)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *tensors[] = {query, key, value, beta, initialState, g, out, records};
    for (size_t i = 0; i < 8; ++i) {
        const auto *tensor = tensors[i];
        if (tensor == nullptr || tensor->GetData() == nullptr || !IsContiguous(tensor) ||
            tensor->GetStorageFormat() != Format::FORMAT_ND ||
            tensor->GetDataType() != ((i == 5 || i == 7) ? DataType::DT_FLOAT : DataType::DT_FLOAT16)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (query->GetViewShape().GetDimNum() != 3 || value->GetViewShape().GetDimNum() != 3) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t n = query->GetViewShape().GetDim(0);
    const int64_t hk = query->GetViewShape().GetDim(1);
    const int64_t hv = value->GetViewShape().GetDim(1);
    if (n < 1 || n > TreeGDN310::MAX_NODES || hk < 1 || hv < hk || hv > 32 || hv % hk ||
        parents->Size() != static_cast<uint64_t>(n) ||
        !ShapeIs(query, {n, hk, 128}) || !ShapeIs(key, {n, hk, 128}) ||
        !ShapeIs(value, {n, hv, 128}) || !ShapeIs(beta, {n, hv}) ||
        !ShapeIs(g, {n, hv}) || !ShapeIs(initialState, {hv, 128, 128}) ||
        !ShapeIs(out, {n, hv, 128}) || !ShapeIs(records, {n, hv, 144})) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Shape-bounded byte counts; reject shared storage AND partial overlap.
    uintptr_t starts[8], ends[8];
    for (size_t i = 0; i < 8; ++i) {
        starts[i] = reinterpret_cast<uintptr_t>(tensors[i]->GetData());
        const uint64_t bytes = static_cast<uint64_t>(tensors[i]->Numel()) * ((i == 5 || i == 7) ? 4U : 2U);
        if (bytes > std::numeric_limits<uintptr_t>::max() - starts[i]) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        ends[i] = starts[i] + bytes;
    }
    for (size_t output = 6; output < 8; ++output) {
        for (size_t other = 0; other < output; ++other) {
            if (tensors[output]->GetStorageAddr() == tensors[other]->GetStorageAddr() ||
                (starts[output] < ends[other] && starts[other] < ends[output])) {
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    TreeGDN310::TreeGatedDeltaRuleV310TilingData td{};
    std::string error;
    if (!TreeGDN310::PlanTopology(
            std::vector<int64_t>(parents->GetData(), parents->GetData() + parents->Size()), td, error)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    L2_DFX_PHASE_1(aclnnTreeGdnCompactVerifyV310,
        DFX_IN(query, key, value, beta, initialState, g, parents, scale, vTile),
        DFX_OUT(out, records));
    for (const auto *tensor : tensors) {
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const auto status = l0op::Launch(query, key, value, beta, initialState, g, parents,
                                     scale, vTile, out, records, uniqueExecutor.get());
    CHECK_RET(status == ACLNN_SUCCESS, status);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnTreeGdnCompactVerifyV310(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTreeGdnCompactVerifyV310);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
