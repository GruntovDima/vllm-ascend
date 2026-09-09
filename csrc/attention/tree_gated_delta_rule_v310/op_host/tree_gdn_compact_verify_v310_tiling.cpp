// Copyright (c) 2026. Licensed under the repository LICENSE.
// Rendered from tiling.cpp.tmpl. Exact UB accounting replaces generic Cube fields.
#include "tree_gated_delta_rule_v310_tiling.h"
#include <cstring>
#include <initializer_list>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/broadcast/broadcast_tiling.h"
#include "tiling_base/error_log.h"

namespace optiling {
using namespace TreeGDN310;

static bool ShapeIs(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != dims.size()) {
        return false;
    }
    size_t i = 0;
    for (auto dim : dims) {
        if (shape->GetStorageShape().GetDim(i++) != dim) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus TilingTreeGDN(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr ||
        context->GetAttrs() == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &q = context->GetInputShape(0)->GetStorageShape();
    const auto &v = context->GetInputShape(2)->GetStorageShape();
    if (q.GetDimNum() != 3 || v.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const auto n = q.GetDim(0);
    const auto hk = q.GetDim(1);
    const auto hv = v.GetDim(1);
    if (n < 1 || n > MAX_NODES || hk < 1 || hv < hk || hv > 32 || hv % hk != 0) {
        OP_LOGE(context->GetNodeName(), "Unsupported nodes or head dimensions");
        return ge::GRAPH_FAILED;
    }
    if (!ShapeIs(context->GetInputShape(0), {n, hk, KEY_DIM}) ||
        !ShapeIs(context->GetInputShape(1), {n, hk, KEY_DIM}) ||
        !ShapeIs(context->GetInputShape(2), {n, hv, VALUE_DIM}) ||
        !ShapeIs(context->GetInputShape(3), {n, hv}) ||
        !ShapeIs(context->GetInputShape(4), {hv, VALUE_DIM, KEY_DIM}) ||
        !ShapeIs(context->GetInputShape(5), {n, hv}) ||
        !ShapeIs(context->GetOutputShape(0), {n, hv, VALUE_DIM}) ||
        !ShapeIs(context->GetOutputShape(1), {n, hv, VALUE_DIM + 16})) {
        OP_LOGE(context->GetNodeName(), "Tree GDN shape contract failed");
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < 6; ++i) {
        const auto *desc = context->GetInputDesc(i);
        if (desc == nullptr || desc->GetStorageFormat() != ge::FORMAT_ND ||
            desc->GetDataType() != (i == 5 ? ge::DT_FLOAT : ge::DT_FLOAT16)) {
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t i = 0; i < 2; ++i) {
        const auto *desc = context->GetOutputDesc(i);
        if (desc == nullptr || desc->GetStorageFormat() != ge::FORMAT_ND ||
            desc->GetDataType() != (i == 1 ? ge::DT_FLOAT : ge::DT_FLOAT16)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *parents = context->GetAttrs()->GetListInt(0);
    const auto *scale = context->GetAttrs()->GetAttrPointer<float>(1);
    const auto *requestedTile = context->GetAttrs()->GetAttrPointer<int64_t>(2);
    if (parents == nullptr || parents->GetSize() != static_cast<size_t>(n) ||
        scale == nullptr || !std::isfinite(*scale) || requestedTile == nullptr ||
        (*requestedTile != 0 && *requestedTile != 16 && *requestedTile != 32 && *requestedTile != 64)) {
        OP_LOGE(context->GetNodeName(), "Invalid tree attributes");
        return ge::GRAPH_FAILED;
    }
    TreeGatedDeltaRuleV310TilingData td{};
    std::string error;
    std::vector<int64_t> parentVector(parents->GetData(), parents->GetData() + parents->GetSize());
    if (!PlanTopology(parentVector, td, error)) {
        OP_LOGE(context->GetNodeName(), "%s", error.c_str());
        return ge::GRAPH_FAILED;
    }
    td.keyHeads = hk;
    td.valueHeads = hv;
    td.scale = *scale;
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const auto cores = platform.GetCoreNumAiv();
    if (ub == 0 || cores == 0) {
        return ge::GRAPH_FAILED;
    }
    // Qwen batch-one measurements favor R64. Preserve queried UB capacity:
    // deep/wide trees that do not fit automatically fall back to R32/R16.
    bool fits = false;
    const uint32_t candidates[] = {64, 32, 16};
    for (uint32_t candidate : candidates) {
        td.rows = *requestedTile == 0 ? candidate : static_cast<uint32_t>(*requestedTile);
        uint32_t maxTmp = 0;
        uint32_t minTmp = 0;
        AscendC::GetBroadCastMaxMinTmpSize(platform, ge::Shape({td.rows, 1}),
                                         ge::Shape({td.rows, KEY_DIM}), sizeof(float),
                                         false, maxTmp, minTmp);
        if (PlanUb(td, maxTmp) <= ub) {
            fits = true;
            break;
        }
        if (*requestedTile != 0) {
            break;
        }
    }
    if (!fits) {
        OP_LOGE(context->GetNodeName(), "Tree stack and Broadcast temporary exceed queried UB");
        return ge::GRAPH_FAILED;
    }
    const uint32_t tasks = td.valueHeads * (VALUE_DIM / td.rows);
    td.coreCount = std::min(static_cast<uint32_t>(cores), tasks);
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (raw == nullptr || raw->GetCapacity() < sizeof(td) || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &td, sizeof(td));
    raw->SetDataSize(sizeof(td));
    context->SetTilingKey(10000);
    context->SetBlockDim(td.coreCount);
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrepareTreeGDN(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TreeGdnCompactVerifyV310)
    .Tiling(TilingTreeGDN)
    .TilingParse<TreeGatedDeltaRuleV310CompileInfo>(PrepareTreeGDN);
} // namespace optiling
