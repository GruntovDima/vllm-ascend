/*!
 * \file swiglu_quant_v310_kernel.h
 * \brief Fused SwiGlu + per-channel AscendQuant for Ascend 310P.
 *
 * Reproduces bit-exactly the CANN built-in chains:
 *  - swi_glu (ops_nn/ascendc/swi_glu/swi_glu_bf16.hpp, SwigluVectorBF16<
 *    half,float,half> — the fp16 path used on 310P): fp32 domain
 *    Cast -> Muls(-1) -> Exp -> Adds(1) -> Div(gate, 1+e) -> Mul(up),
 *    final Cast CAST_NONE (only __CCE_AICORE__==220 uses RINT)
 *  - ascend_quant_v2 (ops_nn/ascendc/ascend_quant_v2/ascend_quant_v2_fp16.h
 *    per-channel path + CastOut MODE_ROUND, M200 branch):
 *    Cast f16->f32 -> Mul(scale f32) -> Add(offset f32) ->
 *    Cast RINT int32 -> SetDeqScale(1.0) -> Cast half -> Cast int8 CAST_ROUND
 */
#ifndef SWIGLU_QUANT_V310_KERNEL_H
#define SWIGLU_QUANT_V310_KERNEL_H

#include "kernel_operator.h"

namespace NsSwigluQuantV310 {

using namespace AscendC;

template <HardEvent event>
__aicore__ inline void SyncEvent()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

class SwigluQuantV310Kernel {
public:
    static constexpr uint32_t ROW_TILE = 4;

    __aicore__ inline SwigluQuantV310Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
                                const SwigluQuantV310TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        numRows_ = tilingData->numRows;
        hidden_ = tilingData->hidden;
        colTile_ = tilingData->colTile;
        const uint32_t blockIdx = GetBlockIdx();
        const uint64_t baseRows = tilingData->baseRowsPerCore;
        const uint32_t extraCores = tilingData->extraRowCoreCount;
        localRows_ = baseRows + (blockIdx < extraCores ? 1 : 0);
        startRow_ = static_cast<uint64_t>(blockIdx) * baseRows + (blockIdx < extraCores ? blockIdx : extraCores);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), numRows_ * hidden_ * 2);
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scale), hidden_);
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(offset), hidden_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y), numRows_ * hidden_);

        const uint32_t tileElems = ROW_TILE * colTile_;
        pipe_->InitBuffer(gateQueue_, 2, tileElems * sizeof(half));
        pipe_->InitBuffer(upQueue_, 2, tileElems * sizeof(half));
        pipe_->InitBuffer(outQueue_, 2, tileElems * sizeof(int8_t));
        pipe_->InitBuffer(scaleStage_, colTile_ * sizeof(half));
        pipe_->InitBuffer(offsetStage_, colTile_ * sizeof(half));
        pipe_->InitBuffer(scaleF_, colTile_ * sizeof(float));
        pipe_->InitBuffer(offsetF_, colTile_ * sizeof(float));
        pipe_->InitBuffer(cHalf_, tileElems * sizeof(half));
        pipe_->InitBuffer(afF_, tileElems * sizeof(float));
        pipe_->InitBuffer(cF_, tileElems * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (localRows_ == 0) {
            return;
        }
        for (uint32_t c0 = 0; c0 < hidden_; c0 += colTile_) {
            const uint32_t ct = ((hidden_ - c0) < colTile_) ? (hidden_ - c0) : colTile_;
            LoadScaleOffset(c0, ct);
            const uint64_t tileCount = (localRows_ + ROW_TILE - 1) / ROW_TILE;
            CopyInTile(0, ct, c0);
            for (uint64_t tile = 0; tile < tileCount; ++tile) {
                if (tile + 1 < tileCount) {
                    CopyInTile(tile + 1, ct, c0);
                }
                ComputeTile(tile, ct);
                CopyOutTile(tile, ct, c0);
            }
        }
    }

private:
    __aicore__ inline void LoadScaleOffset(uint32_t c0, uint32_t ct)
    {
        // WAR guard: next tile's MTE2 write to scaleStage_/offsetStage_ must
        // wait for the previous tile's V Cast reads to drain.
        if (scaleLoaded_) {
            SyncEvent<HardEvent::V_MTE2>();
        } else {
            scaleLoaded_ = true;
        }
        LocalTensor<half> sStage = scaleStage_.Get<half>();
        LocalTensor<half> oStage = offsetStage_.Get<half>();
        LocalTensor<float> sF = scaleF_.Get<float>();
        LocalTensor<float> oF = offsetF_.Get<float>();
        DataCopy(sStage, scaleGm_[c0], ct);
        DataCopy(oStage, offsetGm_[c0], ct);
        SyncEvent<HardEvent::MTE2_V>();
        Cast(sF, sStage, RoundMode::CAST_NONE, ct);
        Cast(oF, oStage, RoundMode::CAST_NONE, ct);
    }

    __aicore__ inline void CopyInTile(uint64_t tile, uint32_t ct, uint32_t c0)
    {
        LocalTensor<half> gate = gateQueue_.AllocTensor<half>();
        LocalTensor<half> up = upQueue_.AllocTensor<half>();
        const uint64_t row0 = startRow_ + tile * ROW_TILE;
        const uint32_t rows = static_cast<uint32_t>(
            (localRows_ - tile * ROW_TILE) < ROW_TILE ? (localRows_ - tile * ROW_TILE) : ROW_TILE);
        for (uint32_t i = 0; i < rows; ++i) {
            const uint64_t base = (row0 + i) * hidden_ * 2 + c0;
            DataCopy(gate[i * ct], xGm_[base], ct);
            DataCopy(up[i * ct], xGm_[base + hidden_], ct);
        }
        gateQueue_.EnQue(gate);
        upQueue_.EnQue(up);
    }

    __aicore__ inline void ComputeTile(uint64_t tile, uint32_t ct)
    {
        LocalTensor<half> gate = gateQueue_.DeQue<half>();
        LocalTensor<half> up = upQueue_.DeQue<half>();
        LocalTensor<float> afF = afF_.Get<float>();   // gate in f32, later reused for up
        LocalTensor<float> cF = cF_.Get<float>();
        LocalTensor<half> cHalf = cHalf_.Get<half>();
        LocalTensor<float> xf = afF;                  // aliases: afF is dead after Mul
        const uint64_t row0 = tile * ROW_TILE;
        const uint32_t rows = static_cast<uint32_t>(
            (localRows_ - row0) < ROW_TILE ? (localRows_ - row0) : ROW_TILE);
        const uint32_t n = rows * ct;

        // SwiGluVectorBF16<half, float, half> (fp16 path on 310P), verbatim,
        // applied to the flattened (rows x ct) tile: elementwise, bit-identical.
        Cast(afF, gate, RoundMode::CAST_NONE, n);
        Muls(cF, afF, static_cast<float>(-1.0), n);
        Exp(cF, cF, n);
        Adds(cF, cF, static_cast<float>(1.0), n);
        Div(cF, afF, cF, n);
        Cast(afF, up, RoundMode::CAST_NONE, n);
        Mul(cF, cF, afF, n);
        // __CCE_AICORE__ == 200 (310P) takes the #else branch: CAST_NONE.
        Cast(cHalf, cF, RoundMode::CAST_NONE, n);

        gateQueue_.FreeTensor(gate);
        upQueue_.FreeTensor(up);

        // AscendQuantV2 fp16 per-channel path, CastOut MODE_ROUND (M200).
        // scale/offset are per-channel (len ct), applied per packed row.
        Cast(xf, cHalf, RoundMode::CAST_NONE, n);
        for (uint32_t i = 0; i < rows; ++i) {
            Mul(xf[i * ct], xf[i * ct], scaleF_.Get<float>(), ct);
            Add(xf[i * ct], xf[i * ct], offsetF_.Get<float>(), ct);
        }
        Cast(xf.ReinterpretCast<int32_t>(), xf, RoundMode::CAST_RINT, n);
        SetDeqScale(static_cast<half>(1.000000e+00f));
        Cast(xf.ReinterpretCast<half>(), xf.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, n);

        LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
        // reinterpret FIRST, then index in half units: after the in-place
        // int32->half cast the buffer is half-packed (2B/elem), so a float-unit
        // offset would be doubled and read the wrong rows.
        LocalTensor<half> xh = xf.ReinterpretCast<half>();
        for (uint32_t i = 0; i < rows; ++i) {
            Cast(outLocal[i * ct], xh[i * ct], RoundMode::CAST_ROUND, ct);
        }
        outQueue_.EnQue(outLocal);
    }

    __aicore__ inline void CopyOutTile(uint64_t tile, uint32_t ct, uint32_t c0)
    {
        LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
        const uint64_t row0 = startRow_ + tile * ROW_TILE;
        const uint32_t rows = static_cast<uint32_t>(
            (localRows_ - tile * ROW_TILE) < ROW_TILE ? (localRows_ - tile * ROW_TILE) : ROW_TILE);
        for (uint32_t i = 0; i < rows; ++i) {
            DataCopy(yGm_[(row0 + i) * hidden_ + c0], outLocal[i * ct], ct);
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    TPipe* pipe_{nullptr};
    uint64_t numRows_{0};
    uint32_t hidden_{0};
    uint32_t colTile_{0};
    uint64_t localRows_{0};
    uint64_t startRow_{0};
    bool scaleLoaded_{false};

    GlobalTensor<half> xGm_;
    GlobalTensor<half> scaleGm_;
    GlobalTensor<half> offsetGm_;
    GlobalTensor<int8_t> yGm_;

    TQue<QuePosition::VECIN, 2> gateQueue_;
    TQue<QuePosition::VECIN, 2> upQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TBuf<TPosition::VECCALC> scaleStage_;
    TBuf<TPosition::VECCALC> offsetStage_;
    TBuf<TPosition::VECCALC> scaleF_;
    TBuf<TPosition::VECCALC> offsetF_;
    TBuf<TPosition::VECCALC> cHalf_;
    TBuf<TPosition::VECCALC> afF_;
    TBuf<TPosition::VECCALC> cF_;
};

}  // namespace NsSwigluQuantV310

#endif  // SWIGLU_QUANT_V310_KERNEL_H
