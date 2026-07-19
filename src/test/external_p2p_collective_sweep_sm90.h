#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

/*
 * In-process external-buffer P2P collective sweep.
 *
 * use_ring=false measures one synchronized collective at a time.
 * use_ring=true measures steady-state throughput with a buffer ring.
 */
std::vector<std::map<std::string, double>>
benchmark_external_p2p_collective_sweep_sm90(
    const std::string& collective,
    const std::vector<int64_t>& sizes,
    int iters,
    int warmup,
    const std::vector<int>& devices,
    bool verify,
    bool use_ring);

} // namespace ooverlap
