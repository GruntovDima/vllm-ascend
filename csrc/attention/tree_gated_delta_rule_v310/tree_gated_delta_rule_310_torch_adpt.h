// Copyright (c) 2026. Licensed under the repository LICENSE.
#pragma once
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "../../aclnn_torch_adapter/op_api_common.h"
#include <limits>
#include <memory>
#include <tuple>

namespace vllm_ascend {
namespace tree_gdn_detail {
struct NDView { const at::Tensor& tensor; };

// The generic repository bridge labels rank-3/4 base tensors NCL/NCHW.
// This operator has an explicitly ND contract. Retag only contiguous BASE
// storage, with its actual view pointer/shape; never reinterpret NZ storage.
inline aclTensor* ConvertType(NDView& view)
{
    static const auto create = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(create != nullptr, "aclCreateTensor is unavailable");
    const auto& tensor = view.tensor;
    TORCH_CHECK(IsOpInputBaseFormat(tensor) && tensor.is_contiguous(),
                "tree-GDN requires contiguous base-format storage");
    auto result = create(tensor.sizes().data(), tensor.dim(),
        tensor.scalar_type() == at::kFloat ? ACL_FLOAT : ACL_FLOAT16,
        tensor.strides().data(), 0, ACL_FORMAT_ND,
        tensor.sizes().data(), tensor.dim(), tensor.data_ptr());
    TORCH_CHECK(result != nullptr, "tree-GDN ND descriptor creation failed");
    return result;
}

inline void CheckTensor(const at::Tensor& tensor, const at::Tensor& query,
                        at::ScalarType dtype, at::IntArrayRef shape)
{
    TORCH_CHECK(tensor.device() == query.device(), "tree-GDN tensors must share one NPU");
    TORCH_CHECK(tensor.scalar_type() == dtype && tensor.is_contiguous(),
                "tree-GDN requires contiguous tensors of the declared dtype");
    TORCH_CHECK(IsOpInputBaseFormat(tensor), "tree-GDN does not reinterpret internal NZ storage");
    TORCH_CHECK(tensor.sizes() == shape, "tree-GDN tensor shape mismatch");
}

// Keep the workspace and all tensor storages alive through OpCommand's host
// enqueue. Do not let a temporary workspace Tensor die before cmd.Run().
inline void Launch(const at::Tensor& query, const at::Tensor& key,
                   const at::Tensor& value, const at::Tensor& beta,
                   const at::Tensor& initial, const at::Tensor& g,
                   at::IntArrayRef parents, float scale, int64_t vTile,
                   at::Tensor& out, at::Tensor& snapshots)
{
    static const auto getAddress = GetOpApiFuncAddr("aclnnTreeGatedDeltaRuleV310GetWorkspaceSize");
    static const auto runAddress = GetOpApiFuncAddr("aclnnTreeGatedDeltaRuleV310");
    TORCH_CHECK(getAddress && runAddress, "tree-GDN ACLNN package is not loaded");
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto workspaceSizeAddress = &workspaceSize;
    auto executorAddress = &executor;
    NDView qNd{query}, kNd{key}, vNd{value}, bNd{beta}, initialNd{initial}, gNd{g};
    NDView outNd{out}, snapshotsNd{snapshots};
    auto converted = ConvertTypes(qNd, kNd, vNd, bNd, initialNd, gNd, parents,
                                  scale, vTile, outNd, snapshotsNd, workspaceSizeAddress, executorAddress);
    auto params = std::shared_ptr<decltype(converted)>(
        new decltype(converted)(std::move(converted)), [](auto* pointers) {
            ReleaseConvertTypes(*pointers);
            delete pointers;
        });
    auto getWorkspace = ConvertToOpApiFunc(*params, getAddress);
    const auto status = call(getWorkspace, *params);
    TORCH_CHECK(status == 0, "tree-GDN GetWorkspaceSize failed (", status, "): ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "tree-GDN workspace exceeds tensor capacity");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    std::vector<at::Tensor> keepalive{query, key, value, beta, initial, g, out, snapshots};
    auto launch = [params, workspace, workspaceSize, executor, stream, keepalive]() -> int {
        using Run = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
        const int result = reinterpret_cast<Run>(runAddress)(
            workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, stream);
        TORCH_CHECK(result == 0, "tree-GDN launch failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand command;
    command.Name("aclnnTreeGatedDeltaRuleV310");
    command.SetCustomHandler(launch);
    command.Run();
}

inline void CheckInputs(const at::Tensor& query, const at::Tensor& key,
                        const at::Tensor& value, const at::Tensor& beta,
                        const at::Tensor& initial, const at::Tensor& g,
                        at::IntArrayRef parents)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "tree-GDN executes on NPU only");
    TORCH_CHECK(query.dim() == 3 && value.dim() == 3, "tree-GDN expects [N,H,128]");
    const int64_t n = query.size(0), hk = query.size(1), hv = value.size(1);
    TORCH_CHECK(n >= 1 && n <= 65 && hk >= 1 && hv >= hk && hv <= 32 && hv % hk == 0,
                "tree-GDN unsupported node/head shape");
    TORCH_CHECK(parents.size() == static_cast<size_t>(n), "tree-GDN parent count mismatch");
    CheckTensor(query, query, at::kHalf, {n, hk, 128});
    CheckTensor(key, query, at::kHalf, {n, hk, 128});
    CheckTensor(value, query, at::kHalf, {n, hv, 128});
    CheckTensor(beta, query, at::kHalf, {n, hv});
    CheckTensor(initial, query, at::kHalf, {hv, 128, 128});
    CheckTensor(g, query, at::kFloat, {n, hv});
}
} // namespace tree_gdn_detail

inline std::tuple<at::Tensor, at::Tensor> npu_tree_gated_delta_rule_310_out(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& beta, const at::Tensor& initial, const at::Tensor& g,
    at::IntArrayRef parents, double scale, int64_t vTile,
    at::Tensor& out, at::Tensor& snapshots)
{
    tree_gdn_detail::CheckInputs(query, key, value, beta, initial, g, parents);
    c10_npu::OptionalNPUGuard guard(query.device());
    tree_gdn_detail::CheckTensor(out, query, at::kHalf, value.sizes());
    tree_gdn_detail::CheckTensor(snapshots, query, at::kHalf,
                                {value.size(0), value.size(1), 128, 128});
    tree_gdn_detail::Launch(query, key, value, beta, initial, g, parents,
                            static_cast<float>(scale), vTile, out, snapshots);
    return {out, snapshots};
}

inline std::tuple<at::Tensor, at::Tensor> npu_tree_gated_delta_rule_310(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& beta, const at::Tensor& initial, const at::Tensor& g,
    at::IntArrayRef parents, double scale, int64_t vTile)
{
    tree_gdn_detail::CheckInputs(query, key, value, beta, initial, g, parents);
    c10_npu::OptionalNPUGuard guard(query.device());
    auto out = at::empty(value.sizes(), value.options());
    auto snapshots = at::empty({value.size(0), value.size(1), 128, 128}, value.options());
    tree_gdn_detail::Launch(query, key, value, beta, initial, g, parents,
                            static_cast<float>(scale), vTile, out, snapshots);
    return {out, snapshots};
}
} // namespace vllm_ascend
