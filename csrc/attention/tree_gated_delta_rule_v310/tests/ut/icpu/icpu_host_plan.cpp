// Keep GE format enums separate from device FORMAT_* macros.
#include "../../../op_host/tree_gdn_plan.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/broadcast/broadcast_tiling.h"
#include <stdexcept>
TreeGDN310::TreeGatedDeltaRuleV310TilingData PrepareIcpuTiling(
    const std::vector<int64_t> &parents,uint32_t hk,uint32_t hv,uint32_t r,float scale) {
    using namespace TreeGDN310;
    TreeGatedDeltaRuleV310TilingData td{};std::string error;
    if(!PlanTopology(parents,td,error))throw std::runtime_error(error);
    auto *platform=platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310P3");
    if(!platform)throw std::runtime_error("platform");
    td.keyHeads=hk;td.valueHeads=hv;td.rows=r;td.scale=scale;
    uint32_t maxTmp=0,minTmp=0;
    AscendC::GetBroadCastMaxMinTmpSize(*platform,ge::Shape({r,1}),
        ge::Shape({r,KEY_DIM}),sizeof(float),false,maxTmp,minTmp);
    PlanUb(td,maxTmp);
    uint64_t capacity=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,capacity);
    if(td.ubBytes>capacity)throw std::runtime_error("UB");
    td.coreCount=std::min(platform->GetCoreNumAiv(),hv*(VALUE_DIM/r));
    return td;
}
