// Isolated integration probe; no installed vLLM registrations are changed.
#include <torch/library.h>
#include "../../tree_gated_delta_rule_310_torch_adpt.h"
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
TORCH_LIBRARY(tree_gdn_probe, m) {
    m.def("forward(Tensor query, Tensor key, Tensor value, Tensor beta, Tensor initial, Tensor g, int[] parents, float scale, int v_tile=0) -> (Tensor, Tensor)");
    m.def("forward.out(Tensor query, Tensor key, Tensor value, Tensor beta, Tensor initial, Tensor g, int[] parents, float scale, int v_tile=0, *, Tensor(a!) out, Tensor(b!) snapshots) -> (Tensor(a!), Tensor(b!))");
}
TORCH_LIBRARY_IMPL(tree_gdn_probe, PrivateUse1, m) {
    m.impl("forward", &vllm_ascend::npu_tree_gated_delta_rule_310);
    m.impl("forward.out", &vllm_ascend::npu_tree_gated_delta_rule_310_out);
}
