#pragma once

#include <cstdint>
#include <vector>

namespace ooverlap {

bool endpoint_persistent_smoke_test(
    int64_t numel,
    const std::vector<int64_t>& devices = {},
    int timeout_ms = 5000);

} // namespace ooverlap
