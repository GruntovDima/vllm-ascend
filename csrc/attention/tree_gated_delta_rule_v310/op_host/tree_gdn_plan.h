// Copyright (c) 2026. Licensed under the repository LICENSE.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "../op_kernel/tree_gated_delta_rule_v310_tiling_data.h"

namespace TreeGDN310 {
// Pure host helpers shared with CPU tests; no device state is copied to the host.
inline bool PlanTopology(const std::vector<int64_t> &parents,
                         TreeGatedDeltaRuleV310TilingData &td, std::string &error)
{
    const auto n = parents.size();
    if (n == 0 || n > MAX_NODES || parents[0] != -1) {
        error = "Tree must have 1..65 nodes with root parent -1";
        return false;
    }
    uint32_t nodeDepth[MAX_NODES] = {};
    uint32_t maxDepth = 0;
    for (uint32_t i = 1; i < n; ++i) {
        if (parents[i] < 0 || parents[i] >= i) {
            error = "Each non-root parent must precede its child";
            return false;
        }
        nodeDepth[i] = nodeDepth[parents[i]] + 1;
        if (nodeDepth[i] > MAX_DEPTH) {
            error = "Tree depth exceeds four draft levels";
            return false;
        }
        maxDepth = std::max(maxDepth, nodeDepth[i]);
    }
    uint32_t next = 0;
    std::function<void(uint32_t)> visit = [&](uint32_t node) {
        td.order[next] = node;
        td.depth[next++] = nodeDepth[node];
        for (uint32_t child = node + 1; child < n; ++child) {
            if (parents[child] == node) {
                visit(child);
            }
        }
    };
    visit(0);
    td.nodes = static_cast<uint32_t>(n);
    td.frames = maxDepth + 2; // Frame 0 initial state; root result is frame 1.
    return next == n;
}

inline uint32_t Align32(uint32_t bytes)
{
    return (bytes + 31U) / 32U * 32U;
}

inline uint32_t PlanUb(TreeGatedDeltaRuleV310TilingData &td, uint32_t broadcastBytes)
{
    const uint32_t r = td.rows;
    const uint32_t m = r * KEY_DIM;
    const uint32_t gates = (td.nodes * td.valueHeads + 15U) / 16U * 16U;
    td.gateCount = gates;
    td.broadcastBytes = std::max(32U, Align32(broadcastBytes));
    const uint32_t sizes[SLOT_COUNT] = {
        td.frames * m * 2, m * 4, m * 4, m * 4, r * 64 * 4,
        KEY_DIM * 2, KEY_DIM * 2, r * 2, KEY_DIM * 4, KEY_DIM * 4,
        KEY_DIM * 4, r * 4, r * 4, r * 4, r * 4, r * 4, r * 2,
        gates * 4, gates * 4, gates * 2, gates * 4, td.broadcastBytes
    };
    uint32_t next = 0;
    for (uint32_t i = 0; i < SLOT_COUNT; ++i) {
        td.offsets[i] = next;
        next += Align32(sizes[i]);
    }
    td.ubBytes = next;
    return next;
}
} // namespace TreeGDN310
