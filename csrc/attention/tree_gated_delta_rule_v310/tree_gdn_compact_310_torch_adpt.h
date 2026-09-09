// Copyright (c) 2026. Licensed under the repository LICENSE.
#pragma once
#include "tree_gated_delta_rule_310_torch_adpt.h"
namespace vllm_ascend {
namespace tree_gdn_detail {
template<class... Args>
inline void LaunchCompact(void *getAddress, void *runAddress, const char *name,
                          const at::Tensor &anchor, std::vector<at::Tensor> keepalive,
                          Args &...args)
{
    TORCH_CHECK(getAddress && runAddress, name, ": compact ACLNN package is not loaded");
    uint64_t workspaceSize=0;
    aclOpExecutor *executor=nullptr;
    auto workspaceSizeAddress=&workspaceSize;
    auto executorAddress=&executor;
    auto converted=ConvertTypes(args...,workspaceSizeAddress,executorAddress);
    auto params=std::shared_ptr<decltype(converted)>(
        new decltype(converted)(std::move(converted)),[](auto *p) {ReleaseConvertTypes(*p);delete p;});
    const auto status=call(ConvertToOpApiFunc(*params,getAddress),*params);
    TORCH_CHECK(status==0,name,": GetWorkspaceSize failed (",status,"): ",aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize<=static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "compact GDN workspace exceeds tensor capacity");
    auto workspace=at::empty({static_cast<int64_t>(workspaceSize)},anchor.options().dtype(at::kByte));
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    auto launch=[params,workspace,workspaceSize,executor,stream,keepalive,runAddress]() -> int {
        using Run=int (*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
        const int result=reinterpret_cast<Run>(runAddress)(
            workspaceSize?workspace.data_ptr():nullptr,workspaceSize,executor,stream);
        TORCH_CHECK(result==0,"compact GDN launch failed: ",aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand command;
    command.Name(name);command.SetCustomHandler(launch);command.Run();
}

inline void VerifyCompact(const at::Tensor &q,const at::Tensor &k,const at::Tensor &v,
    const at::Tensor &beta,const at::Tensor &initial,const at::Tensor &g,
    at::IntArrayRef parents,float scale,int64_t tile,at::Tensor &out,at::Tensor &records)
{
    static const auto get=GetOpApiFuncAddr("aclnnTreeGdnCompactVerifyV310GetWorkspaceSize");
    static const auto run=GetOpApiFuncAddr("aclnnTreeGdnCompactVerifyV310");
    NDView qn{q},kn{k},vn{v},bn{beta},in{initial},gn{g},on{out},rn{records};
    LaunchCompact(get,run,"aclnnTreeGdnCompactVerifyV310",q,{q,k,v,beta,initial,g,out,records},
                  qn,kn,vn,bn,in,gn,parents,scale,tile,on,rn);
}

inline void CheckReplay(const at::Tensor &key,const at::Tensor &initial,const at::Tensor &records,
                        at::IntArrayRef parents,at::IntArrayRef path)
{
    TORCH_CHECK(key.device().type()==c10::DeviceType::PrivateUse1,
                "compact replay executes on NPU only");
    TORCH_CHECK(key.dim()==3 && initial.dim()==3,"compact replay expects ND key and initial state");
    const int64_t n=key.size(0),hk=key.size(1),hv=initial.size(0);
    TORCH_CHECK(n>=1 && n<=65 && hk>=1 && hv>=hk && hv<=32 && hv%hk==0,
                "compact replay unsupported node/head shape");
    CheckTensor(key,key,at::kHalf,{n,hk,128});
    CheckTensor(initial,key,at::kHalf,{hv,128,128});
    CheckTensor(records,key,at::kFloat,{n,hv,144});
    TORCH_CHECK(parents.size()==static_cast<size_t>(n) && parents[0]==-1,
                "compact replay invalid parents");
    std::vector<int64_t> depths(n,0);
    for(int64_t i=1;i<n;++i) {
        TORCH_CHECK(parents[i]>=0 && parents[i]<i,"compact replay parent order");
        depths[i]=depths[parents[i]]+1;
        TORCH_CHECK(depths[i]<=4,"compact replay tree depth exceeds four");
    }
    TORCH_CHECK(path.size()<=5,"compact replay path too long");
    int64_t previous=-1;
    for(auto node:path) {
        TORCH_CHECK(node>=0 && node<n && parents[node]==previous,
                    "compact replay must follow a root-started ancestor path");
        previous=node;
    }
}

inline void ReplayCompact(const at::Tensor &key,const at::Tensor &initial,const at::Tensor &records,
    at::IntArrayRef parents,at::IntArrayRef path,int64_t tile,at::Tensor &accepted)
{
    static const auto get=GetOpApiFuncAddr("aclnnTreeGdnCompactReplayV310GetWorkspaceSize");
    static const auto run=GetOpApiFuncAddr("aclnnTreeGdnCompactReplayV310");
    NDView kn{key},in{initial},rn{records},an{accepted};
    LaunchCompact(get,run,"aclnnTreeGdnCompactReplayV310",key,{key,initial,records,accepted},
                  kn,in,rn,parents,path,tile,an);
}
} // namespace tree_gdn_detail

inline std::tuple<at::Tensor,at::Tensor> npu_tree_gdn_compact_verify_310_out(
    const at::Tensor &query,const at::Tensor &key,const at::Tensor &value,
    const at::Tensor &beta,const at::Tensor &initial,const at::Tensor &g,
    at::IntArrayRef parents,double scale,int64_t tile,at::Tensor &out,at::Tensor &records)
{
    tree_gdn_detail::CheckInputs(query,key,value,beta,initial,g,parents);
    c10_npu::OptionalNPUGuard guard(query.device());
    tree_gdn_detail::CheckTensor(out,query,at::kHalf,value.sizes());
    tree_gdn_detail::CheckTensor(records,query,at::kFloat,{value.size(0),value.size(1),144});
    tree_gdn_detail::VerifyCompact(query,key,value,beta,initial,g,parents,static_cast<float>(scale),tile,out,records);
    return {out,records};
}
inline std::tuple<at::Tensor,at::Tensor> npu_tree_gdn_compact_verify_310(
    const at::Tensor &query,const at::Tensor &key,const at::Tensor &value,
    const at::Tensor &beta,const at::Tensor &initial,const at::Tensor &g,
    at::IntArrayRef parents,double scale,int64_t tile)
{
    tree_gdn_detail::CheckInputs(query,key,value,beta,initial,g,parents);
    c10_npu::OptionalNPUGuard guard(query.device());
    auto out=at::empty(value.sizes(),value.options());
    auto records=at::empty({value.size(0),value.size(1),144},value.options().dtype(at::kFloat));
    tree_gdn_detail::VerifyCompact(query,key,value,beta,initial,g,parents,static_cast<float>(scale),tile,out,records);
    return {out,records};
}
inline at::Tensor npu_tree_gdn_compact_replay_310_out(
    const at::Tensor &key,const at::Tensor &initial,const at::Tensor &records,
    at::IntArrayRef parents,at::IntArrayRef path,int64_t tile,at::Tensor &accepted)
{
    tree_gdn_detail::CheckReplay(key,initial,records,parents,path);
    c10_npu::OptionalNPUGuard guard(key.device());
    TORCH_CHECK(tile==0 || tile==16 || tile==32 || tile==64,"compact replay invalid v_tile");
    tree_gdn_detail::CheckTensor(accepted,key,at::kHalf,
                                {static_cast<int64_t>(path.size()),initial.size(0),128,128});
    if(!path.empty()) tree_gdn_detail::ReplayCompact(key,initial,records,parents,path,tile,accepted);
    return accepted;
}
inline at::Tensor npu_tree_gdn_compact_replay_310(
    const at::Tensor &key,const at::Tensor &initial,const at::Tensor &records,
    at::IntArrayRef parents,at::IntArrayRef path,int64_t tile)
{
    tree_gdn_detail::CheckReplay(key,initial,records,parents,path);
    c10_npu::OptionalNPUGuard guard(key.device());
    auto accepted=at::empty({static_cast<int64_t>(path.size()),initial.size(0),128,128},initial.options());
    return npu_tree_gdn_compact_replay_310_out(key,initial,records,parents,path,tile,accepted);
}
} // namespace vllm_ascend
