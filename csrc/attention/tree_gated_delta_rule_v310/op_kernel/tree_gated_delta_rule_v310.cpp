// Copyright (c) 2026. Licensed under the repository LICENSE.
#include "tree_gated_delta_rule_v310.h"
using namespace AscendC;
using namespace TreeGDN310;

extern "C" __global__ __aicore__ void tree_gated_delta_rule_v310(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR g, GM_ADDR out, GM_ADDR snapshots, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TreeGatedDeltaRuleV310TilingData);
    GET_TILING_DATA(td, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(10000)) {
        TPipe pipe;
        TreeGDNKernel kernel;
        kernel.Init(query, key, value, beta, initialState, g, out, snapshots, &td, &pipe);
        kernel.Process();
    }
}
