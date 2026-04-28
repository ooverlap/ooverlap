#pragma once

#include <cstdint>
#include <string>

namespace ooverlap {

std::string benchmark_public_allreduce_2gpu_sm90(
    int64_t min_bytes,
    int64_t max_bytes,
    int points,
    int iters,
    int warmup,
    int tuning_mode,
    int dev0 = 0,
    int dev1 = 1);

} // namespace ooverlap
