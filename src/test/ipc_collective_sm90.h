#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

// OOVERLAP_IPC_COLLECTIVE_MULTI_GPU_V1
bool smoke_ipc_collective_rank_sm90(
    const std::string& collective,
    int64_t numel,
    int local_rank,
    const std::vector<int>& devices,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    bool verify = true);

std::vector<std::map<std::string, double>> benchmark_ipc_collective_rank_sm90(
    const std::string& collective,
    const std::vector<int64_t>& sizes,
    int local_rank,
    const std::vector<int>& devices,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify);

// Backward-compatible two-GPU overloads.
bool smoke_ipc_collective_rank_sm90(
    const std::string& collective,
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    bool verify = true);

std::vector<std::map<std::string, double>> benchmark_ipc_collective_rank_sm90(
    const std::string& collective,
    const std::vector<int64_t>& sizes,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify);

} // namespace ooverlap
