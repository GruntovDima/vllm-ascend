#include "swiglu_quant_v310_tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr uint32_t K_TILE_ALIGN = 32;   // half 32B = 16 elems; int8 32B = 32 elems
constexpr uint32_t MAX_COL_TILE = 2048;

uint32_t MaxColTile()
{
    const char* env = std::getenv("QSW_COL_TILE");
    if (env != nullptr) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0 && v <= static_cast<long>(MAX_COL_TILE)) {
            return static_cast<uint32_t>(v);
        }
    }
    return MAX_COL_TILE;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* xShape = context->GetInputShape(0);
    const auto* sShape = context->GetInputShape(1);
    const auto* oShape = context->GetInputShape(2);
    if (xShape == nullptr || sShape == nullptr || oShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorageShape = xShape->GetStorageShape();
    const size_t rank = xStorageShape.GetDimNum();
    if (rank == 0) {
        OPS_LOG_E(context, "x must have at least one dimension");
        return ge::GRAPH_FAILED;
    }
    const int64_t xLast = xStorageShape.GetDim(rank - 1);
    if (xLast < 2 * static_cast<int64_t>(K_TILE_ALIGN) || xLast % (2 * K_TILE_ALIGN) != 0) {
        OPS_LOG_E(context, "x last dim must be a multiple of 32, got %ld", xLast);
        return ge::GRAPH_FAILED;
    }
    const int64_t hidden64 = xLast / 2;
    if (hidden64 % K_TILE_ALIGN != 0) {
        OPS_LOG_E(context, "hidden must be a multiple of 16, got %ld", hidden64);
        return ge::GRAPH_FAILED;
    }
    const int64_t elementCount = xStorageShape.GetShapeSize();
    if (elementCount <= 0 || elementCount % xLast != 0) {
        OPS_LOG_E(context, "invalid x shape: elements=%ld last=%ld", elementCount, xLast);
        return ge::GRAPH_FAILED;
    }
    if (sShape->GetStorageShape().GetShapeSize() != hidden64 ||
        oShape->GetStorageShape().GetShapeSize() != hidden64) {
        OPS_LOG_E(context, "scale/offset element count must equal hidden %ld", hidden64);
        return ge::GRAPH_FAILED;
    }

    uint32_t coreNum = 1;
    const auto* compileInfo = reinterpret_cast<const SwigluQuantV310CompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr && compileInfo->totalCoreNum > 0) {
        coreNum = compileInfo->totalCoreNum;
    } else {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        coreNum = std::max<uint32_t>(platform.GetCoreNum(), 1);
    }

    const uint64_t numRows = static_cast<uint64_t>(elementCount / xLast);
    const uint32_t usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(coreNum, numRows));
    const uint64_t baseRowsPerCore = numRows / usedCoreNum;
    const uint32_t extraRowCoreCount = static_cast<uint32_t>(numRows % usedCoreNum);

    uint32_t colTile = static_cast<uint32_t>(std::min<int64_t>(hidden64, MaxColTile()));
    colTile = static_cast<uint32_t>(colTile / K_TILE_ALIGN * K_TILE_ALIGN);
    if (colTile == 0) {
        colTile = K_TILE_ALIGN;
    }

    SwigluQuantV310TilingData tiling;
    tiling.set_numRows(numRows);
    tiling.set_hidden(static_cast<uint32_t>(hidden64));
    tiling.set_colTile(colTile);
    tiling.set_baseRowsPerCore(baseRowsPerCore);
    tiling.set_extraRowCoreCount(extraRowCoreCount);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4SwigluQuantV310(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<SwigluQuantV310CompileInfo>();
    if (compileInfo == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->totalCoreNum = platform.GetCoreNum();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluQuantV310)
    .Tiling(TilingFunc)
    .TilingParse<SwigluQuantV310CompileInfo>(TilingPrepare4SwigluQuantV310);

}  // namespace optiling
