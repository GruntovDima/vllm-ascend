#ifndef SWIGLU_QUANT_V310_TILING_H
#define SWIGLU_QUANT_V310_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SwigluQuantV310TilingData)
    TILING_DATA_FIELD_DEF(uint64_t, numRows);
    TILING_DATA_FIELD_DEF(uint32_t, hidden);
    TILING_DATA_FIELD_DEF(uint32_t, colTile);
    TILING_DATA_FIELD_DEF(uint64_t, baseRowsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, extraRowCoreCount);
END_TILING_DATA_DEF;

struct SwigluQuantV310CompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

REGISTER_TILING_DATA_CLASS(SwigluQuantV310, SwigluQuantV310TilingData)

}  // namespace optiling

#endif  // SWIGLU_QUANT_V310_TILING_H
