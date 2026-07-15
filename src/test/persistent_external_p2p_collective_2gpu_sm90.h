#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

bool external_p2p_collective_smoke_test(
    const std::string& collective,
    int64_t numel,
    const std::vector<int>& devices);

bool external_p2p_allreduce_smoke_test(
    int64_t numel,
    const std::vector<int>& devices);

std::map<std::string, double> benchmark_external_p2p_collective_sm90(
    const std::string& collective,
    int64_t numel,
    int iters,
    int warmup,
    const std::vector<int>& devices);

std::map<std::string, double> benchmark_external_p2p_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    const std::vector<int>& devices);

// Backward-compatible two-GPU wrappers.
bool external_p2p_two_gpu_collective_smoke_test(
    const std::string& collective,
    int64_t numel,
    int dev0 = 0,
    int dev1 = 1);

bool external_p2p_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0 = 0,
    int dev1 = 1);

std::map<std::string, double> benchmark_external_p2p_two_gpu_collective_sm90(
    const std::string& collective,
    int64_t numel,
    int iters,
    int warmup,
    int dev0 = 0,
    int dev1 = 1);

std::map<std::string, double> benchmark_external_p2p_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0 = 0,
    int dev1 = 1);

} // namespace ooverlap
