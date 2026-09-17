#include "../../op_host/qbmm_k_pipeline_policy.h"

using optiling::SelectQbmKPipeline;

static_assert(SelectQbmKPipeline(true, 1, 2, 0, 0, 0, 64, 1, 2) == 1);
static_assert(SelectQbmKPipeline(false, 1, 2, 0, 0, 0, 64, 1, 2) == 0);
static_assert(SelectQbmKPipeline(true, 2, 2, 0, 0, 0, 64, 1, 2) == 0);
static_assert(SelectQbmKPipeline(true, 1, 0, 0, 0, 0, 64, 1, 2) == 0);
static_assert(SelectQbmKPipeline(true, 1, 2, 1, 0, 0, 64, 1, 2) == 0);
static_assert(SelectQbmKPipeline(true, 1, 2, 0, 1, 0, 64, 1, 2) == 0);
static_assert(SelectQbmKPipeline(true, 1, 2, 0, 0, 2, 64, 1, 2) == 0);
static_assert(SelectQbmKPipeline(true, 1, 2, 0, 0, 0, 63, 1, 2) == 0);
static_assert(SelectQbmKPipeline(true, 1, 2, 0, 0, 0, 64, 2, 2) == 0);
static_assert(SelectQbmKPipeline(true, 1, 2, 0, 0, 0, 64, 1, 1) == 0);

int main()
{
    return 0;
}
