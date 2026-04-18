#pragma once

#include "comm/chunk.h"

namespace ooverlap {
namespace comm {

struct ChunkScheduler {
    const ChunkRegistration* registrations = nullptr;
    int num_registrations = 0;

    int ticket = 0;
    int ticket_stride = 1;

    Chunk current{};
};

__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler* sched) {
    sched->registrations = nullptr;
    sched->num_registrations = 0;
    sched->ticket = 0;
    sched->ticket_stride = 1;
    chunk_clear(&sched->current);
}

__host__ __device__ __forceinline__ int chunk_scheduler_total_chunks(
    const ChunkScheduler* sched) {
    if (sched == nullptr || sched->registrations == nullptr || sched->num_registrations <= 0) {
        return 0;
    }

    int total = 0;
    for (int i = 0; i < sched->num_registrations; ++i) {
        total += chunk_registration_num_chunks(&sched->registrations[i]);
    }
    return total;
}

__host__ __device__ __forceinline__ bool chunk_scheduler_resolve_ticket(
    const ChunkScheduler* sched,
    int ticket,
    Chunk* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_clear(out);

    if (sched == nullptr || sched->registrations == nullptr || sched->num_registrations <= 0) {
        return false;
    }
    if (ticket < 0) {
        return false;
    }

    int remaining = ticket;
    for (int reg_idx = 0; reg_idx < sched->num_registrations; ++reg_idx) {
        const ChunkRegistration* reg = &sched->registrations[reg_idx];
        const int num_chunks = chunk_registration_num_chunks(reg);
        if (remaining < num_chunks) {
            return chunk_registration_resolve_chunk(
                reg,
                reg_idx,
                remaining,
                out);
        }
        remaining -= num_chunks;
    }

    return false;
}

__host__ __device__ __forceinline__ void chunk_scheduler_init(
    ChunkScheduler* sched,
    const ChunkRegistration* registrations,
    int num_registrations,
    int start_ticket,
    int ticket_stride) {
    sched->registrations = registrations;
    sched->num_registrations = num_registrations;
    sched->ticket = start_ticket;
    sched->ticket_stride = ticket_stride;
    chunk_clear(&sched->current);
}

__host__ __device__ __forceinline__ void chunk_scheduler_prime(
    ChunkScheduler* sched) {
    Chunk resolved{};
    if (!chunk_scheduler_resolve_ticket(sched, sched->ticket, &resolved)) {
        chunk_clear(&sched->current);
        return;
    }
    sched->current = resolved;
}

__host__ __device__ __forceinline__ bool chunk_scheduler_active(
    const ChunkScheduler* sched) {
    return chunk_is_valid(&sched->current);
}

__host__ __device__ __forceinline__ const Chunk* chunk_scheduler_current(
    const ChunkScheduler* sched) {
    return &sched->current;
}

__host__ __device__ __forceinline__ bool chunk_scheduler_peek_next(
    const ChunkScheduler* sched,
    Chunk* out) {
    if (sched == nullptr) {
        return false;
    }
    return chunk_scheduler_resolve_ticket(
        sched,
        sched->ticket + sched->ticket_stride,
        out);
}

__host__ __device__ __forceinline__ void chunk_scheduler_advance(
    ChunkScheduler* sched) {
    sched->ticket += sched->ticket_stride;

    Chunk resolved{};
    if (!chunk_scheduler_resolve_ticket(sched, sched->ticket, &resolved)) {
        chunk_clear(&sched->current);
        return;
    }
    sched->current = resolved;
}

} // namespace comm
} // namespace ooverlap
