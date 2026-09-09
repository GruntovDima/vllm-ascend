// Copyright (c) 2026. Licensed under the repository LICENSE.
#pragma once
#include <cstdint>

namespace TreeGDN310 {
constexpr uint32_t MAX_NODES = 65;
constexpr uint32_t MAX_DEPTH = 4;
constexpr uint32_t KEY_DIM = 128;
constexpr uint32_t VALUE_DIM = 128;

enum Slot : uint32_t {
    STACK, STATE, DECAYED, TMP, FOLD, Q_HALF, K_HALF, V_HALF,
    Q_FLOAT, K_FLOAT, Q_SCALED, V_FLOAT, REDUCED, DELTA, DELTA_SCALED,
    OUT_FLOAT, OUT_HALF, G_RAW, G_EXP, BETA_HALF, BETA_FLOAT, BROADCAST_TMP, SLOT_COUNT
};

#pragma pack(push, 8)
struct alignas(8) TreeGatedDeltaRuleV310TilingData {
    uint32_t nodes;
    uint32_t keyHeads;
    uint32_t valueHeads;
    uint32_t rows;
    uint32_t frames;
    uint32_t coreCount;
    uint32_t gateCount;
    uint32_t ubBytes;
    uint32_t broadcastBytes;
    float scale;
    uint32_t order[MAX_NODES];
    uint32_t depth[MAX_NODES];
    uint32_t offsets[SLOT_COUNT];
};
#pragma pack(pop)
} // namespace TreeGDN310
