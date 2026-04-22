#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace ooverlap {

std::map<std::string, double> benchmark_persistent_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0 = 0,
    int dev1 = 1);

} // namespace ooverlap
