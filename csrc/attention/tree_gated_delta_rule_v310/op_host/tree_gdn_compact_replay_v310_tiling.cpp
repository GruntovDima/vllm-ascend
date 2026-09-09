// Copyright (c) 2026. Licensed under the repository LICENSE.
// Reuses v4 PlatformAscendC/PlanUb tiling; no hardcoded hardware capacities.
#include "compact_plan.h"
#include "tree_gated_delta_rule_v310_tiling.h"
#include <cstring>
#include <initializer_list>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/broadcast/broadcast_tiling.h"
namespace optiling {
using namespace TreeGDN310;
static bool ReplayShape(const gert::StorageShape *s,std::initializer_list<int64_t> dims) {
    if(!s || s->GetStorageShape().GetDimNum()!=dims.size()) return false;
    size_t i=0; for(auto d:dims) if(s->GetStorageShape().GetDim(i++)!=d) return false;
    return true;
}
static ge::graphStatus TilingCompactReplay(gert::TilingContext *context) {
    if(!context || !context->GetPlatformInfo() || !context->GetAttrs() ||
       !context->GetInputShape(0) || !context->GetInputShape(1)) return ge::GRAPH_FAILED;
    const auto &key=context->GetInputShape(0)->GetStorageShape();
    const auto &initial=context->GetInputShape(1)->GetStorageShape();
    if(key.GetDimNum()!=3 || initial.GetDimNum()!=3) return ge::GRAPH_FAILED;
    const int64_t n=key.GetDim(0),hk=key.GetDim(1),hv=initial.GetDim(0);
    const auto *parents=context->GetAttrs()->GetListInt(0);
    const auto *path=context->GetAttrs()->GetListInt(1);
    const auto *requested=context->GetAttrs()->GetAttrPointer<int64_t>(2);
    if(n<1 || n>MAX_NODES || hk<1 || hv<hk || hv>32 || hv%hk ||
       !parents || parents->GetSize()!=static_cast<size_t>(n) ||
       !path || path->GetSize()<1 || path->GetSize()>MAX_DEPTH+1 ||
       !requested || (*requested!=0 && *requested!=16 && *requested!=32 && *requested!=64))
        return ge::GRAPH_FAILED;
    if(!ReplayShape(context->GetInputShape(0),{n,hk,KEY_DIM}) ||
       !ReplayShape(context->GetInputShape(1),{hv,VALUE_DIM,KEY_DIM}) ||
       !ReplayShape(context->GetInputShape(2),{n,hv,COMPACT_RECORD_DIM}) ||
       !ReplayShape(context->GetOutputShape(0),{static_cast<int64_t>(path->GetSize()),hv,VALUE_DIM,KEY_DIM}))
        return ge::GRAPH_FAILED;
    for(uint32_t i=0;i<3;++i) {
        const auto *desc=context->GetInputDesc(i);
        if(!desc || desc->GetStorageFormat()!=ge::FORMAT_ND ||
           desc->GetDataType()!=(i==2?ge::DT_FLOAT:ge::DT_FLOAT16)) return ge::GRAPH_FAILED;
    }
    const auto *output=context->GetOutputDesc(0);
    if(!output || output->GetStorageFormat()!=ge::FORMAT_ND || output->GetDataType()!=ge::DT_FLOAT16)
        return ge::GRAPH_FAILED;
    TreeGatedDeltaRuleV310TilingData td{};
    std::string error;
    const std::vector<int64_t> pv(parents->GetData(),parents->GetData()+parents->GetSize());
    if(!PlanTopology(pv,td,error)) return ge::GRAPH_FAILED;
    td.keyHeads=hk;td.valueHeads=hv;td.scale=1.0f;
    const auto platform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    const auto cores=platform.GetCoreNumAiv();
    if(!ub || !cores) return ge::GRAPH_FAILED;
    bool fits=false;
    for(uint32_t candidate:{64U,32U,16U}) {
        td.rows=*requested==0?candidate:static_cast<uint32_t>(*requested);
        uint32_t maxTmp=0,minTmp=0;
        AscendC::GetBroadCastMaxMinTmpSize(platform,ge::Shape({td.rows,1}),
            ge::Shape({td.rows,KEY_DIM}),sizeof(float),false,maxTmp,minTmp);
        if(PlanUb(td,maxTmp)<=ub) {fits=true;break;}
        if(*requested!=0) break;
    }
    if(!fits) return ge::GRAPH_FAILED;
    td.coreCount=std::min(static_cast<uint32_t>(cores),td.valueHeads*(VALUE_DIM/td.rows));
    ReplayPlan plan{};
    try {
        plan=MakeReplayPlan(td,pv,std::vector<int64_t>(path->GetData(),path->GetData()+path->GetSize()));
    } catch(const std::invalid_argument &) {return ge::GRAPH_FAILED;}
    auto *raw=context->GetRawTilingData();
    auto *ws=context->GetWorkspaceSizes(1);
    if(!raw || raw->GetCapacity()<sizeof(plan) || !ws) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(),&plan,sizeof(plan));raw->SetDataSize(sizeof(plan));
    context->SetTilingKey(10000);context->SetBlockDim(td.coreCount);
    ws[0]=platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus PrepareCompactReplay(gert::TilingParseContext *context) {
    return context?ge::GRAPH_SUCCESS:ge::GRAPH_FAILED;
}
IMPL_OP_OPTILING(TreeGdnCompactReplayV310).Tiling(TilingCompactReplay)
    .TilingParse<TreeGatedDeltaRuleV310CompileInfo>(PrepareCompactReplay);
}
