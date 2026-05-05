#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

/*
 * New clean API for the experiment.
 *
 * sizes_bytes:
 *   Exact buffer sizes to test.
 *
 * num_blocks_list:
 *   CTA counts to test. Each CTA count becomes a separate line/shape later
 *   on the Python plotting side.
 *
 * Returned rows use numeric IDs:
 *
 *   experiment:
 *     0 = copy
 *     1 = reduce_add_f16
 *
 *   scenario:
 *     0 = local_to_peer
 *     1 = peer_to_local
 *
 *   method:
 *     0 = tma_copy
 *     1 = fast_copy_u128
 *     2 = nccl_sendrecv
 *     3 = tma_reduce_add_f16
 *     4 = fast_add_f16_u128
 */
std::vector<std::map<std::string, double>>
benchmark_tma_bandwidth_experiment_sweep_sm90(
    const std::vector<int64_t>& sizes_bytes,
    const std::vector<int>& num_blocks_list,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    bool include_nccl);

/*
 * Compatibility wrapper for the current pybind/CMake side.
 *
 * This still runs the new cleaned-up experiment, but it generates powers of two
 * between min_bytes and max_bytes and uses one num_blocks value.
 *
 * include_mem_async is intentionally ignored now.
 */
std::vector<std::map<std::string, double>>
benchmark_tma_bandwidth_experiment_sm90(
    int64_t min_bytes,
    int64_t max_bytes,
    int iters,
    int warmup,
    int num_blocks,
    int dev0,
    int dev1,
    bool include_mem_async,
    bool include_nccl);

} // namespace ooverlap
