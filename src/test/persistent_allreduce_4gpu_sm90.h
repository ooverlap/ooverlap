#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace ooverlap {

bool tma_persistent_four_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0 = 0,
    int dev1 = 1,
    int dev2 = 2,
    int dev3 = 3);

std::map<std::string, double> benchmark_persistent_four_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0 = 0,
    int dev1 = 1,
    int dev2 = 2,
    int dev3 = 3);

} // namespace ooverlap
