#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

/*
 * In-process external-buffer P2P collective sweep.
 *
 * All GPU ranks live in one process and are created through
 * oo_group_create_p2p(). Unlike the persistent benchmark, this interface uses
 * one work buffer per rank/backend and resets it outside the timed region for
 * every iteration. It intentionally does not use a buffer ring.
 */
std::vector<std::map<std::string, double>>
benchmark_external_p2p_collective_sweep_sm90(
    const std::string& collective,
    const std::vector<int64_t>& sizes,
    int iters,
    int warmup,
    const std::vector<int>& devices,
    bool verify);

} // namespace ooverlap
