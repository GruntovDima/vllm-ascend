// Experimental record layout: arithmetic dimensions, not chip constants.
#pragma once
#include "tree_gated_delta_rule_v310_tiling_data.h"

namespace TreeGDN310 {
constexpr uint32_t COMPACT_GAMMA_LANES = 16;
constexpr uint32_t COMPACT_RECORD_DIM = VALUE_DIM + COMPACT_GAMMA_LANES;
struct ReplayPlan {
    TreeGatedDeltaRuleV310TilingData tree;
    uint32_t count;
    uint32_t path[MAX_DEPTH + 1];
};
} // namespace TreeGDN310
