#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/group.h"

namespace ooverlap {
namespace comm {

// New model:
//   Endpoint  -> per GPU
//   Channel   -> per directed peer pair
//   Group     -> participating set of endpoints
//
// Compatibility:
//   current code still uses Communicator/TmaCommunicator and the old helper APIs.
//   We keep those names as aliases/wrappers over Group.
using Communicator = Group;
using TmaCommunicator = Communicator;

bool communicator_init(
    Communicator* comm,
    const std::vector<int>& devices,
    size_t max_full_numel,
    int num_channel_slots = 1);

void communicator_destroy(
    Communicator* comm);

CommChannel* communicator_get_channel(
    Communicator* comm,
    int src_rank,
    int dst_rank);

const CommChannel* communicator_get_channel(
    const Communicator* comm,
    int src_rank,
    int dst_rank);

transport::CommBuffer* channel_get_slot_buffer(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

const transport::CommBuffer* channel_get_slot_buffer(
    const Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

transport::CommBuffer* channel_get_slot_signal_buffer(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

const transport::CommBuffer* channel_get_slot_signal_buffer(
    const Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx);

transport::CommBuffer* communicator_get_local_shard_buffer(
    Communicator* comm,
    int rank);

const transport::CommBuffer* communicator_get_local_shard_buffer(
    const Communicator* comm,
    int rank);

transport::CommBuffer* communicator_get_local_full_buffer(
    Communicator* comm,
    int rank);

const transport::CommBuffer* communicator_get_local_full_buffer(
    const Communicator* comm,
    int rank);

cudaError_t channel_send_bulk_tma(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx,
    const half* src,
    size_t numel,
    cudaStream_t stream);

} // namespace comm
} // namespace ooverlap
