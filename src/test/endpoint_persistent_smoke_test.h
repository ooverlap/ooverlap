#pragma once

#include <cstdint>

namespace ooverlap {

bool endpoint_persistent_smoke_test(
    int64_t numel,
    int dev0 = 0,
    int dev1 = 1);

} // namespace ooverlap
