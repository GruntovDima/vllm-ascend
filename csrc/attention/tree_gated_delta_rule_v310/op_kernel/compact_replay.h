// Experimental accepted-path replay. No reductions or inverse-state arithmetic.
// Composes elementwise-row-load TBuf, flat multi-core-dispatch, aligned MTE3.
#pragma once
#include "kernel_operator.h"
#include "lib/pad/broadcast.h"
#include "compact_layout.h"

namespace TreeGDN310 {
using namespace AscendC;

class CompactReplayKernel {
public:
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR initial, GM_ADDR records,
                               GM_ADDR acceptedStates, const ReplayPlan *plan,
                               TPipe *pipe)
    {
        plan_ = plan;
        td_ = &plan_->tree;
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(key));
        initialGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initial));
        recordsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(records));
        statesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(acceptedStates));
        pipe->InitBuffer(arena_, td_->ubBytes);
        base_ = arena_.Get<uint8_t>();
        m2v_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_V));
        vm2_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
        vm3_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE3));
        m3v_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        if (plan_->count == 0) {
            return;
        }
        const uint32_t tiles = VALUE_DIM / td_->rows;
        const uint32_t tasks = td_->valueHeads * tiles;
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

    __aicore__ inline void ProcessTile(uint32_t head, uint32_t firstRow)
    {
        const uint32_t m = td_->rows * KEY_DIM;
        const uint32_t keyHead = head / (td_->valueHeads / td_->keyHeads);
        auto checkpoint = Buf<half>(STACK);
        auto kHalf = Buf<half>(K_HALF);
        auto k = Buf<float>(K_FLOAT);
        auto state = Buf<float>(STATE);
        auto decayed = Buf<float>(DECAYED);
        auto broadcast = Buf<float>(TMP);
        auto delta = Buf<float>(DELTA_SCALED);
        auto broadcastTmp = Buf<uint8_t>(BROADCAST_TMP);
        Fence<HardEvent::V_MTE2>(vm2_);
        DataCopy(checkpoint, initialGm_[(head * VALUE_DIM + firstRow) * KEY_DIM], m);
        Fence<HardEvent::MTE2_V>(m2v_);

        for (uint32_t j = 0; j < plan_->count; ++j) {
            const uint32_t node = plan_->path[j];
            const uint32_t record = (node * td_->valueHeads + head) * COMPACT_RECORD_DIM;
            // Only V consumes these UB slots; gamma is an immutable GM scalar.
            Fence<HardEvent::V_MTE2>(vm2_);
            DataCopy(kHalf, keyGm_[(node * td_->keyHeads + keyHead) * KEY_DIM], KEY_DIM);
            DataCopy(delta, recordsGm_[record + firstRow], td_->rows);
            Fence<HardEvent::MTE2_V>(m2v_);
            const float decay = recordsGm_.GetValue(record + VALUE_DIM);
            Cast(state, checkpoint, RoundMode::CAST_NONE, m);
            Cast(k, kHalf, RoundMode::CAST_NONE, KEY_DIM);
            PipeBarrier<PIPE_V>();
            Muls(decayed, state, decay, m);
            const uint32_t dstShape[2] = {td_->rows, KEY_DIM};
            const uint32_t srcShape[2] = {td_->rows, 1};
            Broadcast<float, 2, 1>(broadcast, delta, dstShape, srcShape, broadcastTmp);
            PipeBarrier<PIPE_V>();
            for (uint32_t column = 0; column < KEY_DIM; column += 64) {
                MulAddDst(decayed[column], broadcast[column], k[column], 64, td_->rows,
                          {1, 1, 1, 16, 16, 0});
            }
            PipeBarrier<PIPE_V>();
            // Do not carry unrounded FP32 state into the next accepted edge.
            Cast(checkpoint, decayed, RoundMode::CAST_NONE, m);
            Fence<HardEvent::V_MTE3>(vm3_);
            const uint32_t output = ((j * td_->valueHeads + head) * VALUE_DIM + firstRow) * KEY_DIM;
            DataCopy(statesGm_[output], checkpoint, m);
            Fence<HardEvent::MTE3_V>(m3v_);
        }
    }

    const ReplayPlan *plan_;
    const TreeGatedDeltaRuleV310TilingData *td_;
    GlobalTensor<half> keyGm_, initialGm_, statesGm_;
    GlobalTensor<float> recordsGm_;
    TBuf<TPosition::VECCALC> arena_;
    LocalTensor<uint8_t> base_;
    event_t m2v_, vm2_, vm3_, m3v_;
};
} // namespace TreeGDN310
