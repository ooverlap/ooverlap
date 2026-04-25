#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

std::map<std::string, double> benchmark_ipc_two_gpu_allreduce_rank_sm90(
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify);

} // namespace ooverlap
