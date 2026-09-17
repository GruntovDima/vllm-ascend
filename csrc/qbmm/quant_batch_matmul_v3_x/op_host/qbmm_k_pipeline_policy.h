/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef QBM_K_PIPELINE_POLICY_H
#define QBM_K_PIPELINE_POLICY_H

#include <cstdint>

namespace optiling {

// Keep the experimental path deliberately narrower than the general kernel.
// This helper has no CANN dependencies so its fallback matrix can be compiled
// and tested on a CPU host.
constexpr uint32_t SelectQbmKPipeline(
    bool requested, uint32_t quantGroupNum, uint32_t scaleType,
    uint32_t isPerTensor, uint32_t hasPertoken, uint32_t kTail,
    uint32_t n, uint32_t wChunkKPasses, uint32_t kPasses)
{
    return requested && quantGroupNum == 1U && scaleType == 2U &&
                   isPerTensor == 0U && hasPertoken == 0U && kTail == 0U &&
                   (n % 16U) == 0U && wChunkKPasses == 1U && kPasses > 1U
               ? 1U
               : 0U;
}

}  // namespace optiling

#endif  // QBM_K_PIPELINE_POLICY_H
