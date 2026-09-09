// Copyright (c) 2026. Licensed under the repository LICENSE.
#pragma once
#include "kernel_operator.h"
#include "lib/pad/broadcast.h"
#include "compact_layout.h"

namespace TreeGDN310 {
using namespace AscendC;

// EXPERIMENTAL: isolated from the installed full-snapshot kernel.
// Template provenance: elementwise-row-load (TBuf COMPUTE_BODY), flat
// multi-core-dispatch and the aligned-only mte3-write-out branch.
// Each work item owns one head / V-row tile over ALL tree nodes. No core reads
// another core's state, and no output GM is consumed as an input.
class CompactTreeGDNKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR initialState,
        GM_ADDR g, GM_ADDR out, GM_ADDR records,
        const TreeGatedDeltaRuleV310TilingData *td, TPipe *pipe)
    {
        td_ = td;
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(key));
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(value));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(beta));
        initialGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initialState));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out));
        recordsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(records));
        pipe->InitBuffer(arena_, td_->ubBytes);
        base_ = arena_.Get<uint8_t>();
        m2v_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_V));
        vm2_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
        vm3_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE3));
        m3v_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_V));
        vs_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
        sv_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
    }

    __aicore__ inline void Process()
    {
        LoadGates();
        const uint32_t tiles = VALUE_DIM / td_->rows;
        const uint32_t tasks = td_->valueHeads * tiles;
        // Stride by LAUNCHED cores, not a hardware core-count constant.
        for (uint32_t task = GetBlockIdx(); task < tasks; task += td_->coreCount) {
            ProcessTile(task / tiles, (task % tiles) * td_->rows);
        }
    }

private:
    template <typename T>
    __aicore__ inline LocalTensor<T> Buf(Slot slot)
    {
        return base_[td_->offsets[slot]].template ReinterpretCast<T>();
    }

    template <HardEvent event>
    __aicore__ inline void Fence(event_t id)
    {
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline void LoadGates()
    {
        auto raw = Buf<float>(G_RAW);
        auto gamma = Buf<float>(G_EXP);
        auto betaHalf = Buf<half>(BETA_HALF);
        auto beta = Buf<float>(BETA_FLOAT);
        const uint32_t count = td_->nodes * td_->valueHeads;
        const uint32_t gAligned = count / 8 * 8;
        const uint32_t bAligned = count / 16 * 16;
        const bool hasDma = gAligned != 0 || bAligned != 0;
        const bool hasTail = gAligned != td_->gateCount || bAligned != td_->gateCount;
        // Scalar tails/padding and DMA prefixes own disjoint, block-aligned
        // UB intervals. Never zero a prefix that DMA will overwrite.
        // These bounded S-pipe loops load only valid GM elements (no overread).
        for (uint32_t i = gAligned; i < td_->gateCount; ++i) {
            raw.SetValue(i, i < count ? gGm_.GetValue(i) : 0.0f);
        }
        for (uint32_t i = bAligned; i < td_->gateCount; ++i) {
            betaHalf.SetValue(i, i < count ? betaGm_.GetValue(i) : static_cast<half>(0.0f));
        }
        if (gAligned != 0) {
            DataCopy(raw, gGm_, gAligned);
        }
        if (bAligned != 0) {
            DataCopy(betaHalf, betaGm_, bAligned);
        }
        if (hasTail) {
            Fence<HardEvent::S_V>(sv_);
        }
        if (hasDma) {
            Fence<HardEvent::MTE2_V>(m2v_);
        }
        Exp(gamma, raw, td_->gateCount);
        Cast(beta, betaHalf, RoundMode::CAST_NONE, td_->gateCount);
        Fence<HardEvent::V_S>(vs_);
    }

    __aicore__ inline void MatVec(LocalTensor<float> dst, LocalTensor<float> matrix,
                                  LocalTensor<float> vector, bool accumulate)
    {
        for (uint32_t k = 0; k < KEY_DIM; k += 64) {
            if (accumulate) {
                MulAddDst(dst[k], matrix[k], vector[k], 64, td_->rows,
                          {1, 1, 1, 16, 16, 0});
            } else {
                Mul(dst[k], matrix[k], vector[k], 64, td_->rows,
                    {1, 1, 1, 16, 16, 0});
            }
        }
    }

    __aicore__ inline void ReduceRows(LocalTensor<float> dst, LocalTensor<float> src)
    {
        auto fold = Buf<float>(FOLD);
        // Match the existing 310P one-token recurrence's K128 reduction order.
        Add(fold, src, src[64], 64, td_->rows, {1, 1, 1, 8, 16, 16});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dst, fold, 64, td_->rows, 1, 1, 8);
    }

    __aicore__ inline void ProcessTile(uint32_t head, uint32_t firstRow)
    {
        const uint32_t m = td_->rows * KEY_DIM;
        auto stack = Buf<half>(STACK);
        auto qHalf = Buf<half>(Q_HALF);
        auto kHalf = Buf<half>(K_HALF);
        auto vHalf = Buf<half>(V_HALF);
        auto q = Buf<float>(Q_FLOAT);
        auto k = Buf<float>(K_FLOAT);
        auto qScaled = Buf<float>(Q_SCALED);
        auto v = Buf<float>(V_FLOAT);
        auto state = Buf<float>(STATE);
        auto decayed = Buf<float>(DECAYED);
        auto tmp = Buf<float>(TMP);
        auto reduced = Buf<float>(REDUCED);
        auto delta = Buf<float>(DELTA);
        auto scaled = Buf<float>(DELTA_SCALED);
        auto outFloat = Buf<float>(OUT_FLOAT);
        auto outHalf = Buf<half>(OUT_HALF);
        auto gamma = Buf<float>(G_EXP);
        auto beta = Buf<float>(BETA_FLOAT);
        auto broadcastTmp = Buf<uint8_t>(BROADCAST_TMP);
        const uint32_t keyHead = head / (td_->valueHeads / td_->keyHeads);
        Fence<HardEvent::V_MTE2>(vm2_);
        DataCopy(stack, initialGm_[(head * VALUE_DIM + firstRow) * KEY_DIM], m);
        Fence<HardEvent::MTE2_V>(m2v_);

        for (uint32_t step = 0; step < td_->nodes; ++step) {
            const uint32_t node = td_->order[step];
            const uint32_t depth = td_->depth[step];
            const uint32_t qOffset = (node * td_->keyHeads + keyHead) * KEY_DIM;
            const uint32_t vOffset = (node * td_->valueHeads + head) * VALUE_DIM + firstRow;
            Fence<HardEvent::V_MTE2>(vm2_);
            DataCopy(qHalf, queryGm_[qOffset], KEY_DIM);
            DataCopy(kHalf, keyGm_[qOffset], KEY_DIM);
            DataCopy(vHalf, valueGm_[vOffset], td_->rows);
            Fence<HardEvent::MTE2_V>(m2v_);
            // Re-read FP16 parent frame at EVERY edge, including the primary chain.
            Cast(state, stack[depth * m], RoundMode::CAST_NONE, m);
            Cast(q, qHalf, RoundMode::CAST_NONE, KEY_DIM);
            Cast(k, kHalf, RoundMode::CAST_NONE, KEY_DIM);
            Cast(v, vHalf, RoundMode::CAST_NONE, td_->rows);
            PipeBarrier<PIPE_V>();
            Muls(qScaled, q, td_->scale, KEY_DIM);
            const uint32_t gate = node * td_->valueHeads + head;
            const float decay = gamma.GetValue(gate);
            Muls(decayed, state, decay, m);
            PipeBarrier<PIPE_V>();
            MatVec(tmp, decayed, k, false);
            PipeBarrier<PIPE_V>();
            ReduceRows(reduced, tmp);
            PipeBarrier<PIPE_V>();
            Sub(delta, v, reduced, td_->rows);
            PipeBarrier<PIPE_V>();
            Muls(scaled, delta, beta.GetValue(gate), td_->rows);
            PipeBarrier<PIPE_V>();
            const uint32_t dstShape[2] = {td_->rows, KEY_DIM};
            const uint32_t srcShape[2] = {td_->rows, 1};
            Broadcast<float, 2, 1>(tmp, scaled, dstShape, srcShape, broadcastTmp);
            PipeBarrier<PIPE_V>();
            // Intentional fused multiply-add to decayed state, matching the golden.
            MatVec(decayed, tmp, k, true);
            PipeBarrier<PIPE_V>();
            MatVec(tmp, decayed, qScaled, false);
            PipeBarrier<PIPE_V>();
            ReduceRows(outFloat, tmp);
            PipeBarrier<PIPE_V>();
            Cast(stack[(depth + 1) * m], decayed, RoundMode::CAST_NONE, m);
            Cast(outHalf, outFloat, RoundMode::CAST_NONE, td_->rows);
            // DELTA has no live consumers after scaled was computed and fenced.
            // Gamma tail has one writer per head, never one writer per V tile.
            if (firstRow == 0) {
                Duplicate(delta, decay, COMPACT_GAMMA_LANES);
            }
            Fence<HardEvent::V_MTE3>(vm3_);
            const uint32_t record = (node * td_->valueHeads + head) * COMPACT_RECORD_DIM;
            DataCopy(outGm_[vOffset], outHalf, td_->rows);
            DataCopy(recordsGm_[record + firstRow], scaled, td_->rows);
            if (firstRow == 0) {
                DataCopy(recordsGm_[record + VALUE_DIM], delta, COMPACT_GAMMA_LANES);
            }
            // Conservative first version: complete DMA before any stack-frame reuse.
            Fence<HardEvent::MTE3_V>(m3v_);
        }
    }

    const TreeGatedDeltaRuleV310TilingData *td_;
    GlobalTensor<half> queryGm_, keyGm_, valueGm_, betaGm_, initialGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<half> outGm_;
    GlobalTensor<float> recordsGm_;
    TBuf<TPosition::VECCALC> arena_;
    LocalTensor<uint8_t> base_;
    event_t m2v_, vm2_, vm3_, m3v_, vs_, sv_;
};
} // namespace TreeGDN310
