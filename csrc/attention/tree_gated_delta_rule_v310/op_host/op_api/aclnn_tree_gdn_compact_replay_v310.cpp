// Copyright (c) 2026. Licensed under the repository LICENSE.
// Template provenance: existing validated tree-GDN direct-output ACLNN adapter.
#include "aclnn_tree_gdn_compact_replay_v310.h"
#include "../compact_plan.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <initializer_list>
#include <limits>
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(TreeGdnCompactReplayV310);
static aclnnStatus Launch(const aclTensor *key, const aclTensor *initialState,
    const aclTensor *records, const aclIntArray *parents, const aclIntArray *path,
    int64_t vTile, aclTensor *acceptedStates, aclOpExecutor *executor)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(TreeGdnCompactReplayV310,
        OP_INPUT(key, initialState, records), OP_OUTPUT(acceptedStates), OP_ATTR(parents, path, vTile));
}
}
namespace {
bool ShapeIs(const aclTensor *tensor, std::initializer_list<int64_t> dims) {
    const auto &shape=tensor->GetViewShape();
    if(shape.GetDimNum()!=dims.size()) return false;
    size_t i=0;
    for(auto dim:dims) if(shape.GetDim(i++)!=dim) return false;
    return true;
}
}
extern "C" aclnnStatus aclnnTreeGdnCompactReplayV310GetWorkspaceSize(
    const aclTensor *key, const aclTensor *initialState, const aclTensor *records,
    const aclIntArray *parents, const aclIntArray *path, int64_t vTile,
    aclTensor *acceptedStates, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // Reject aliases BEFORE the executor-cache fast return, as in full verify.
    if(!workspaceSize || !executor || !parents || !path ||
       (vTile!=0 && vTile!=16 && vTile!=32 && vTile!=64)) return ACLNN_ERR_PARAM_INVALID;
    const aclTensor *tensors[]={key,initialState,records,acceptedStates};
    for(size_t i=0;i<4;++i) {
        if(!tensors[i] || !IsContiguous(tensors[i]) ||
           tensors[i]->GetStorageFormat()!=Format::FORMAT_ND ||
           tensors[i]->GetDataType()!=(i==2 ? DataType::DT_FLOAT : DataType::DT_FLOAT16) ||
           (tensors[i]->Numel()!=0 && !tensors[i]->GetData())) return ACLNN_ERR_PARAM_INVALID;
    }
    if(key->GetViewShape().GetDimNum()!=3 || initialState->GetViewShape().GetDimNum()!=3)
        return ACLNN_ERR_PARAM_INVALID;
    const int64_t n=key->GetViewShape().GetDim(0), hk=key->GetViewShape().GetDim(1);
    const int64_t hv=initialState->GetViewShape().GetDim(0), count=path->Size();
    if(n<1 || n>65 || hk<1 || hv<hk || hv>32 || hv%hk || count>5 ||
       parents->Size()!=static_cast<uint64_t>(n) ||
       !ShapeIs(key,{n,hk,128}) || !ShapeIs(initialState,{hv,128,128}) ||
       !ShapeIs(records,{n,hv,144}) || !ShapeIs(acceptedStates,{count,hv,128,128}))
        return ACLNN_ERR_PARAM_INVALID;
    TreeGDN310::TreeGatedDeltaRuleV310TilingData base{};
    base.nodes=n;
    try {
        const std::vector<int64_t> pv(parents->GetData(),parents->GetData()+parents->Size());
        std::vector<int64_t> selected;
        if(count) selected.assign(path->GetData(),path->GetData()+count);
        (void)TreeGDN310::MakeReplayPlan(base,pv,selected);
    } catch(const std::invalid_argument &) { return ACLNN_ERR_PARAM_INVALID; }
    uintptr_t starts[4],ends[4];
    for(size_t i=0;i<4;++i) {
        starts[i]=reinterpret_cast<uintptr_t>(tensors[i]->GetData());
        const uint64_t bytes=static_cast<uint64_t>(tensors[i]->Numel())*(i==2?4U:2U);
        if(bytes>std::numeric_limits<uintptr_t>::max()-starts[i]) return ACLNN_ERR_PARAM_INVALID;
        ends[i]=starts[i]+bytes;
    }
    if(count) for(size_t i=0;i<3;++i) {
        if(tensors[3]->GetStorageAddr()==tensors[i]->GetStorageAddr() ||
           (starts[3]<ends[i] && starts[i]<ends[3])) return ACLNN_ERR_PARAM_INVALID;
    }
    L2_DFX_PHASE_1(aclnnTreeGdnCompactReplayV310,
        DFX_IN(key,initialState,records,parents,path,vTile),DFX_OUT(acceptedStates));
    for(const auto *tensor:tensors) tensor->SetOriginalShape(tensor->GetViewShape());
    auto uniqueExecutor=CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get()!=nullptr,ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if(count) {
        const auto status=l0op::Launch(key,initialState,records,parents,path,vTile,
                                      acceptedStates,uniqueExecutor.get());
        CHECK_RET(status==ACLNN_SUCCESS,status);
    }
    *workspaceSize=uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnTreeGdnCompactReplayV310(
    void *workspace,uint64_t workspaceSize,aclOpExecutor *executor,aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTreeGdnCompactReplayV310);
    return CommonOpExecutorRun(workspace,workspaceSize,executor,stream);
}
