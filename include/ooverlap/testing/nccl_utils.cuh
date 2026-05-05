#pragma once

#include <nccl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ooverlap {
namespace testing {

inline ncclUniqueId make_nccl_unique_id(
    const std::vector<int64_t>& encoded) {
    ncclUniqueId id;
    std::memset(&id, 0, sizeof(id));

    const size_t expected_bytes = sizeof(id.internal);

    if (encoded.size() * sizeof(int64_t) == expected_bytes) {
        std::memcpy(id.internal, encoded.data(), expected_bytes);
        return id;
    }

    if (encoded.size() == expected_bytes) {
        for (size_t i = 0; i < encoded.size(); ++i) {
            if (encoded[i] < 0 || encoded[i] > 255) {
                throw std::invalid_argument("NCCL unique ID byte out of range");
            }

            id.internal[i] = static_cast<char>(encoded[i]);
        }

        return id;
    }

    throw std::invalid_argument(
        "NCCL unique ID has wrong encoded size: got " +
        std::to_string(encoded.size()) +
        " int64 values; expected either " +
        std::to_string(expected_bytes / sizeof(int64_t)) +
        " packed int64 values or " +
        std::to_string(expected_bytes) +
        " byte values");
}

} // namespace testing
} // namespace ooverlap
