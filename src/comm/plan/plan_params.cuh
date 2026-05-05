#pragma once

namespace ooverlap {
namespace comm {
namespace plan {

constexpr int kTmaMultiGpuAllReduceMaxPeers = 15;
constexpr int kTmaMultiGpuReduceScatterMaxPeers = 15;
constexpr int kTmaMultiGpuAllGatherMaxPeers = 15;

constexpr int kTmaMultiGpuAllReduceMaxWindowTasks = 256;
constexpr int kTmaMultiGpuReduceScatterMaxWindowTasks = 256;
constexpr int kTmaMultiGpuAllGatherMaxWindowTasks = 256;

} // namespace plan
} // namespace comm
} // namespace ooverlap
