// Host-only topology/UB tests, not an NPU correctness verdict.
#include "../../op_host/tree_gdn_plan.h"
#include <cassert>
#include <iostream>
#include <set>

using namespace TreeGDN310;
void CheckTree(const std::vector<int64_t> &parents)
{
    TreeGatedDeltaRuleV310TilingData td{};
    std::string error;
    assert(PlanTopology(parents, td, error));
    std::vector<int64_t> frames(td.frames, -99);
    frames[0] = -1;
    std::set<uint32_t> visited;
    for (uint32_t step = 0; step < td.nodes; ++step) {
        const uint32_t node = td.order[step], depth = td.depth[step];
        assert(frames[depth] == parents[node]);
        frames[depth + 1] = node;
        assert(visited.insert(node).second);
    }
    assert(visited.size() == parents.size());
    for (uint32_t rows : {16U, 32U, 64U}) {
        for (uint32_t heads : {1U, 3U, 8U, 32U}) {
            td.rows = rows;
            td.valueHeads = heads;
            PlanUb(td, rows * 128 * 4); // Arbitrary scratch extent, not a HW constant.
            assert(td.offsets[0] == 0);
            for (uint32_t slot = 1; slot < SLOT_COUNT; ++slot) {
                assert(td.offsets[slot] % 32 == 0);
                assert(td.offsets[slot] > td.offsets[slot - 1]);
            }
            assert(td.ubBytes == td.offsets[BROADCAST_TMP] + td.broadcastBytes);
            assert(td.gateCount >= td.nodes * heads);
            assert(td.gateCount % 16 == 0);
        }
    }
}
int main()
{
    for (auto parents : std::vector<std::vector<int64_t>>{
             {-1}, {-1, 0}, {-1, 0, 1, 2, 3}, {-1, 0, 0, 0},
             {-1, 0, 0, 1, 2, 3, 4}, {-1, 0, 0, 1, 1, 2, 2}}) {
        CheckTree(parents);
    }
    for (uint32_t width : {1U, 2U, 4U, 8U, 16U}) {
        for (uint32_t depth = 1; depth <= 4; ++depth) {
            std::vector<int64_t> parents{-1};
            uint32_t primary = 0;
            for (uint32_t d = 0; d < depth; ++d) {
                uint32_t next = parents.size();
                for (uint32_t branch = 0; branch < width; ++branch) {
                    parents.push_back(primary);
                }
                primary = next;
            }
            CheckTree(parents);
        }
    }
    for (auto parents : std::vector<std::vector<int64_t>>{
             {}, {0}, {-1, -1}, {-1, 1}, {-1, 2, 0}, {-1, 0, 1, 2, 3, 4},
             std::vector<int64_t>(66, 0)}) {
        TreeGatedDeltaRuleV310TilingData td{};
        std::string error;
        assert(!PlanTopology(parents, td, error));
    }
    std::cout << "OK: 26 valid topologies, 7 invalid, DFS parent frames and UB layouts\n";
}
