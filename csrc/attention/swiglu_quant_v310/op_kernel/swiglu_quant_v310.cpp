#include "swiglu_quant_v310_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void swiglu_quant_v310(
    GM_ADDR x,
    GM_ADDR scale,
    GM_ADDR offset,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        NsSwigluQuantV310::SwigluQuantV310Kernel kernel;
        kernel.Init(x, scale, offset, y, &tilingData, &pipe);
        kernel.Process();
    }
}
