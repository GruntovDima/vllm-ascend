// Test-only host tiling dump: queries the same CANN platform and Broadcast API.
#include "../../op_host/tree_gdn_plan.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/broadcast/broadcast_tiling.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace TreeGDN310;
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: dump_tree_tiling params.txt");
        std::ifstream input(argv[1]);
        uint32_t n, hk, hv, rows; float scale;
        if (!(input >> n >> hk >> hv >> rows >> scale)) throw std::runtime_error("params");
        std::vector<int64_t> parents(n);
        for (auto &p : parents) if (!(input >> p)) throw std::runtime_error("parents");
        TreeGatedDeltaRuleV310TilingData td{}; std::string error;
        if (!PlanTopology(parents, td, error)) throw std::runtime_error(error);
        auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310P3");
        if (!platform) throw std::runtime_error("platform");
        uint64_t ub = 0; platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        td.keyHeads=hk; td.valueHeads=hv; td.rows=rows; td.scale=scale;
        uint32_t maxTmp=0, minTmp=0;
        AscendC::GetBroadCastMaxMinTmpSize(*platform, ge::Shape({rows,1}),
            ge::Shape({rows,KEY_DIM}), sizeof(float), false, maxTmp, minTmp);
        PlanUb(td, maxTmp);
        td.coreCount=std::min(platform->GetCoreNumAiv(), hv*(VALUE_DIM/rows));
        if (td.ubBytes>ub || !td.coreCount) throw std::runtime_error("platform capacity");
        std::cout << "{\"nodes\":" << n << ",\"keyHeads\":" << hk
            << ",\"valueHeads\":" << hv << ",\"rows\":" << rows
            << ",\"frames\":" << td.frames << ",\"coreCount\":" << td.coreCount
            << ",\"gateCount\":" << td.gateCount << ",\"ubBytes\":" << td.ubBytes
            << ",\"ubCapacity\":" << ub << ",\"broadcastBytes\":" << td.broadcastBytes
            << ",\"order\":[";
        for(uint32_t i=0;i<n;++i) std::cout << (i?",":"") << td.order[i];
        std::cout << "],\"depth\":[";
        for(uint32_t i=0;i<n;++i) std::cout << (i?",":"") << td.depth[i];
        std::cout << "],\"offsets\":[";
        for(uint32_t i=0;i<SLOT_COUNT;++i) std::cout << (i?",":"") << td.offsets[i];
        std::cout << "]}\n";
        return 0;
    } catch(const std::exception &e) { std::cerr << e.what() << "\n"; return 1; }
}
