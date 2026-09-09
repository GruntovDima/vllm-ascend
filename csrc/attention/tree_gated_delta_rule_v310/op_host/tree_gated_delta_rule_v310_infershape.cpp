// Copyright (c) 2026. Licensed under the repository LICENSE.
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferTreeGDN(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(2) == nullptr ||
        context->GetInputShape(4) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *value = context->GetInputShape(2);
    const auto *state = context->GetInputShape(4);
    if (value->GetDimNum() != 3 || state->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *value;
    auto *snapshots = context->GetOutputShape(1);
    snapshots->SetDimNum(4);
    snapshots->SetDim(0, value->GetDim(0));
    for (size_t i = 0; i < 3; ++i) {
        snapshots->SetDim(i + 1, state->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferTreeGDNType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    context->SetOutputDataType(1, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(TreeGatedDeltaRuleV310)
    .InferShape(InferTreeGDN)
    .InferDataType(InferTreeGDNType);
} // namespace ops
