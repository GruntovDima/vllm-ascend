#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_0 = 0;

static ge::graphStatus InferShapeSwigluQuantV310(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;
    const size_t rank = yShape->GetDimNum();
    if (rank == 0) {
        return GRAPH_FAILED;
    }
    yShape->SetDim(rank - 1, xShape->GetDim(rank - 1) / 2);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluQuantV310).InferShape(InferShapeSwigluQuantV310);
}  // namespace ops
