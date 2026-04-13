#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

std::map<std::string, double> benchmark_2gpu_copy_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0 = 0,
    int dev1 = 1);

std::map<std::string, double> benchmark_basic_ngpu_collective_sm90(
    const std::string& op,
    int64_t numel,
    const std::vector<int64_t>& devices,
    int iters,
    int warmup);

} // namespace ooverlap
