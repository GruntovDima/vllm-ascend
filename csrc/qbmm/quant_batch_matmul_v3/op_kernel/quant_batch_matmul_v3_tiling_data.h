/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_tiling_data.h
 * \brief Kernel-side tiling data structs. Field order and names must match
 *        BEGIN_TILING_DATA_DEF(QBMParams) in op_host/quant_batch_matmul_v3_tiling.h.
 * \author Feodor Pisnitchenko
 */

#ifndef QUANT_BATCH_MATMUL_V3_TILING_DATA_H
#define QUANT_BATCH_MATMUL_V3_TILING_DATA_H

#include <cstdint>

struct QBMParams {
    uint32_t batch;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t usedCoreNum;
    uint32_t hasPertoken;
    uint32_t quantGroupNum;
    uint32_t maxOneTurnToken;
    uint32_t maxWeightColOneTurn;
    uint32_t baseK;
    uint32_t scaleType;
    uint32_t isPerTensor;
    uint32_t phaseXChunk;
    uint32_t phaseXFullPairs;
    uint32_t phaseXTail;
    uint32_t MCoreNum;      // M-axis cores in the 2-D grid
    uint32_t NCoreNum;      // N-axis cores in the 2-D grid
    uint32_t wChunkKPasses; // kPasses fetched per MTE2 into wL1
    uint32_t scaleCoalesce; // 0 = per-ch scale MTE2, 1 = one-per-cb coalesced
    uint32_t l2HintMode;    // CacheMode int (0=DISABLE,1=NORMAL,4=PERSISTENT)
    uint32_t mTileCntL2;    // # M-axis super-tiles (1 = no split)
    uint32_t nTileCntL2;    // # N-axis super-tiles (1 = no split)
    uint32_t mTileBlock;    // baseM blocks per M super-tile
    uint32_t nTileBlock;    // baseN blocks per N super-tile
    uint32_t dbL0c;         // 1 = L0C ping-pong; 0 = single
    uint32_t pertokenCoalesce; // 1 = single B*M MTE2 at Process entry; 0 = per-mTile
    uint32_t hasBias;          // 1 = bias[N] int32 broadcast into L0C before Mmad
    uint32_t kTail;            // K % 32
    uint32_t ubCalcM;          // inner-tile Phase D control (Q == 1 only)
};

struct QBMTilingData {
    QBMParams params;
};

#endif // QUANT_BATCH_MATMUL_V3_TILING_DATA_H
