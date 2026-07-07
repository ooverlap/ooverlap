#pragma once

#include <map>
#include <string>

namespace ooverlap {

std::map<std::string, long long> host_mapped_ready_signal_roundtrip(
    int dev_publish,
    int dev_wait,
    int value,
    unsigned long long max_iters);

} // namespace ooverlap
