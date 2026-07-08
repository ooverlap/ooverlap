#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

/*
 * Batched TMA experiment:
 *
 * Experiment 0: reduce_two_to_one
 *   - sequential baseline:
 *       all CTAs reduce peer_src0 -> local_dst,
 *       then all CTAs reduce peer_src1 -> local_dst.
 *   - split-CTA method:
 *       one kernel launch, total CTA budget fixed,
 *       half CTAs reduce peer_src0 -> local_dst,
 *       half CTAs reduce peer_src1 -> local_dst,
 *       using TmaReduceScope::Gpu.
 *
 * Experiment 1: copy_one_to_two
 *   - sequential baseline:
 *       all CTAs copy local_src -> peer_dst0,
 *       then all CTAs copy local_src -> peer_dst1.
 *   - fanout method:
 *       load each local_src chunk once,
 *       issue two TMA stores,
 *       commit once.
 *
 * Returned rows use numeric IDs:
 *
 *   experiment:
 *     0 = reduce_two_to_one
 *     1 = copy_one_to_two
 *
 *   method:
 *     0 = reduce_sequential_all_ctas
 *     1 = reduce_split_ctas_gpu_scope
 *     2 = copy_sequential_all_ctas
 *     3 = copy_fanout_one_load_two_stores
 *
 * Common fields:
 *   bytes:
 *     one logical input buffer size
 *
 *   payload_bytes:
 *     2 * bytes for both experiments
 *
 *   num_blocks:
 *     total CTA budget
 *
 *   split0_ctas, split1_ctas:
 *     CTA split for concurrent variants; baseline rows use num_blocks/num_blocks.
 */
std::vector<std::map<std::string, double>>
benchmark_tma_batch_experiment_sweep_sm90(
    const std::vector<int64_t>& sizes_bytes,
    const std::vector<int>& num_blocks_list,
    int iters,
    int warmup,
    int dev0,
    int dev1);

std::vector<std::map<std::string, double>>
benchmark_tma_batch_experiment_sm90(
    int64_t min_bytes,
    int64_t max_bytes,
    int iters,
    int warmup,
    int num_blocks,
    int dev0,
    int dev1);

/*
 * Compatibility wrappers for the previous tma_bandwidth_experiment API.
 * During this experiment these call the new batch experiment.
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
