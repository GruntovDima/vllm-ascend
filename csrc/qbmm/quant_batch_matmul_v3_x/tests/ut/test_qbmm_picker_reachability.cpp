#define QBM_HOST_PICKER_TEST
#include "../../op_host/quant_batch_matmul_v3_x_tiling.cpp"

#include <cstdio>

#include "../../op_host/qbmm_k_pipeline_policy.h"

int main()
{
    constexpr uint32_t rows[] = {1, 15, 16, 128, 782, 800, 832, 1266, 1280, 2048};
    constexpr uint32_t shapes[][2] = {
        {4096, 64}, {4096, 4096}, {4096, 10240}, {4096, 12288},
        {4096, 24576}, {12288, 4096},
    };
    uint32_t reachable = 0;
    for (uint32_t hasBias : {0U, 1U}) {
    for (uint32_t m : rows) {
        for (const auto& shape : shapes) {
            const uint64_t biasReserve =
                ((static_cast<uint64_t>(shape[1]) * sizeof(int32_t) + 31U) / 32U) * 32U +
                qbmv3::UB_BANK_PAD;
            const uint64_t pickerUb = hasBias != 0U
                ? qbmv3::UB_TOTAL_BYTES - biasReserve
                : qbmv3::UB_TOTAL_BYTES;
            auto tile = optiling::ComputeInt8Tiling(
                m, shape[1], shape[0], 1, 1, 8,
                pickerUb, qbmv3::L0A_TOTAL_BYTES,
                qbmv3::L0B_TOTAL_BYTES, qbmv3::L0C_TOTAL_BYTES,
                qbmv3::L1_TOTAL_BYTES, 0, 2, 0);
            if (tile.baseM == 0U || tile.baseK == 0U) return 10;
            const uint32_t kPasses = (shape[0] + tile.baseK - 1U) / tile.baseK;
            const bool selected = optiling::SelectQbmKPipeline(
                true, 1, 2, 0, 0, 0, shape[1], tile.wChunkKPasses,
                kPasses) != 0U;
            const bool expected = shape[0] == 12288U && shape[1] == 4096U &&
                (m == 128U || m == 1266U || m == 1280U);
            if (selected != expected) {
                std::fprintf(stderr,
                    "reachability mismatch bias=%u M=%u K=%u N=%u baseM=%u "
                    "baseN=%u baseK=%u wChunk=%u\n",
                    hasBias, m, shape[0], shape[1], tile.baseM, tile.baseN,
                    tile.baseK, tile.wChunkKPasses);
                return 20;
            }
            if (selected) {
                // The production picker must keep each L0 half and both L1
                // weight buffers inside the checked-in 310P fallback
                // capacities. Runtime PlatformAscendC values can override
                // these constants and still require a device-side check.
                const uint64_t l0aBytes =
                    static_cast<uint64_t>(tile.baseM) * tile.baseK;
                const uint64_t l0bBytes =
                    static_cast<uint64_t>(tile.baseK) * tile.baseN;
                const uint64_t l1Bytes =
                    static_cast<uint64_t>(tile.baseM) * shape[0] +
                    2ULL * tile.wChunkKPasses * tile.baseK * tile.baseN;
                if (l0aBytes > qbmv3::L0A_HALF_BYTES ||
                    l0bBytes > qbmv3::L0B_HALF_BYTES ||
                    l1Bytes > qbmv3::L1_TOTAL_BYTES) {
                    return 30;
                }
                std::printf(
                    "reachable bias=%u M=%u K=%u N=%u baseM=%u baseN=%u "
                    "baseK=%u wChunk=%u L1=%llu L0A=%llu L0B=%llu\n",
                    hasBias, m, shape[0], shape[1], tile.baseM, tile.baseN,
                    tile.baseK, tile.wChunkKPasses,
                    static_cast<unsigned long long>(l1Bytes),
                    static_cast<unsigned long long>(l0aBytes),
                    static_cast<unsigned long long>(l0bBytes));
                ++reachable;
            }
        }
    }
    }
    return reachable == 6U ? 0 : 40;
}
