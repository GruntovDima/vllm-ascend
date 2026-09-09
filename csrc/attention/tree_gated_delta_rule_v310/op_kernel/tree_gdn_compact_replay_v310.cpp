// Copyright (c) 2026. Licensed under the repository LICENSE.
#include "compact_replay.h"
using namespace AscendC;
using namespace TreeGDN310;
extern "C" __global__ __aicore__ void tree_gdn_compact_replay_v310(
    GM_ADDR key, GM_ADDR initial_state, GM_ADDR records, GM_ADDR accepted_states,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReplayPlan);
    GET_TILING_DATA(td, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(10000)) {
        TPipe pipe;
        CompactReplayKernel kernel;
        kernel.Init(key, initial_state, records, accepted_states, &td, &pipe);
        kernel.Process();
    }
}
