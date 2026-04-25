#pragma once

#include <cstdint>
#include <string>

namespace ooverlap {

bool tma_ipc_two_gpu_allreduce_rank_smoke_test(
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    int iters = 1);

} // namespace ooverlap
