#pragma once
#include "../op_kernel/compact_layout.h"
#include "tree_gdn_plan.h"
#include <stdexcept>

namespace TreeGDN310 {
// Validate the complete path before launching or mutating native cache.
inline ReplayPlan MakeReplayPlan(const TreeGatedDeltaRuleV310TilingData &base,
                                const std::vector<int64_t> &parents,
                                const std::vector<int64_t> &path)
{
    TreeGatedDeltaRuleV310TilingData checked{};
    std::string error;
    if (!PlanTopology(parents, checked, error) || base.nodes != parents.size() ||
        path.size() > MAX_DEPTH + 1) {
        throw std::invalid_argument("Invalid replay tree or path length");
    }
    ReplayPlan plan{};
    plan.tree = base;
    plan.count = static_cast<uint32_t>(path.size());
    int64_t previous = -1;
    for (uint32_t i = 0; i < plan.count; ++i) {
        const int64_t node = path[i];
        if (node < 0 || node >= static_cast<int64_t>(parents.size()) ||
            parents[node] != previous) {
            throw std::invalid_argument("Replay must follow a root-started ancestor path");
        }
        plan.path[i] = static_cast<uint32_t>(node);
        previous = node;
    }
    return plan;
}
} // namespace TreeGDN310
