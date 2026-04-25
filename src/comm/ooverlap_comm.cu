#include "comm/ooverlap_comm.h"

#include "comm/tma_two_gpu_peer_allreduce_sm90.h"
#include "ooverlap/system/peer_buffer.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kOoMaxLocalDevices = 16;

struct oo_group {
    int num_devices = 0;
    int devices[kOoMaxLocalDevices] = {};
};

struct oo_node {
    oo_group_t* group = nullptr;
    int rank = -1;
    int device = -1;
};

struct oo_buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t mapped_bytes = 0;
    oo_buffer_kind_t kind = OO_BUFFER_KIND_WRAPPED;

    // Internal validation/debug metadata. Public semantics should still treat
    // Buffer as pointer + size + kind.
    oo_group_t* group = nullptr;
    int owner_device = -1;

    // Valid only for OO_BUFFER_KIND_VMM.
    ooverlap::system::mapped_peer_buffer mapped{};
};

oo_status_t cuda_status_to_oo(cudaError_t err) {
    return (err == cudaSuccess) ? OO_SUCCESS : OO_ERROR_CUDA;
}

bool valid_cuda_device(int device) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return false;
    }
    return device >= 0 && device < count;
}

bool checked_mul_size(size_t a, size_t b, size_t* out) {
    if (out == nullptr) {
        return false;
    }
    if (a != 0 && b > static_cast<size_t>(-1) / a) {
        return false;
    }
    *out = a * b;
    return true;
}

size_t dtype_size(oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
            return sizeof(half);
        default:
            return 0;
    }
}

bool same_group(const oo_group_t* a, const oo_group_t* b) {
    return a != nullptr && b != nullptr && a == b;
}

} // namespace

extern "C" {

oo_status_t oo_group_create(
    const int* devices,
    int num_devices,
    oo_group_t** out_group) {
    if (out_group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_group = nullptr;

    if (devices == nullptr || num_devices <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (num_devices != 2) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (num_devices > kOoMaxLocalDevices) {
        return OO_ERROR_UNSUPPORTED;
    }

    for (int i = 0; i < num_devices; ++i) {
        if (!valid_cuda_device(devices[i])) {
            return OO_ERROR_INVALID_DEVICE;
        }
        for (int j = 0; j < i; ++j) {
            if (devices[i] == devices[j]) {
                return OO_ERROR_INVALID_ARGUMENT;
            }
        }
    }

    oo_group_t* group = new (std::nothrow) oo_group_t;
    if (group == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    group->num_devices = num_devices;
    for (int i = 0; i < num_devices; ++i) {
        group->devices[i] = devices[i];
    }

    *out_group = group;
    return OO_SUCCESS;
}

void oo_group_destroy(
    oo_group_t* group) {
    delete group;
}

int oo_group_size(
    const oo_group_t* group) {
    return (group != nullptr) ? group->num_devices : 0;
}

int oo_group_device(
    const oo_group_t* group,
    int rank) {
    if (group == nullptr || rank < 0 || rank >= group->num_devices) {
        return -1;
    }
    return group->devices[rank];
}

oo_status_t oo_node_create(
    oo_group_t* group,
    int rank,
    oo_node_t** out_node) {
    if (out_node == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_node = nullptr;

    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (rank < 0 || rank >= group->num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_node_t* node = new (std::nothrow) oo_node_t;
    if (node == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    node->group = group;
    node->rank = rank;
    node->device = group->devices[rank];

    *out_node = node;
    return OO_SUCCESS;
}

void oo_node_destroy(
    oo_node_t* node) {
    delete node;
}

oo_group_t* oo_node_group(
    const oo_node_t* node) {
    return (node != nullptr) ? node->group : nullptr;
}

int oo_node_rank(
    const oo_node_t* node) {
    return (node != nullptr) ? node->rank : -1;
}

int oo_node_device(
    const oo_node_t* node) {
    return (node != nullptr) ? node->device : -1;
}

oo_status_t oo_buffer_alloc(
    oo_node_t* node,
    size_t bytes,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr || bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    try {
        std::vector<int> access_devices;
        access_devices.reserve(static_cast<size_t>(node->group->num_devices));
        for (int i = 0; i < node->group->num_devices; ++i) {
            access_devices.push_back(node->group->devices[i]);
        }

        buffer->mapped = ooverlap::system::alloc_peer_visible_buffer(
            bytes,
            node->device,
            access_devices);

        buffer->ptr = buffer->mapped.ptr;
        buffer->bytes = bytes;
        buffer->mapped_bytes = buffer->mapped.mapped_size;
        buffer->kind = OO_BUFFER_KIND_VMM;
        buffer->group = node->group;
        buffer->owner_device = node->device;
    } catch (const std::bad_alloc&) {
        delete buffer;
        return OO_ERROR_INTERNAL;
    } catch (const std::exception&) {
        delete buffer;
        return OO_ERROR_CUDA;
    } catch (...) {
        delete buffer;
        return OO_ERROR_INTERNAL;
    }

    *out_buffer = buffer;
    return OO_SUCCESS;
}

oo_status_t oo_buffer_wrap(
    oo_node_t* node,
    void* ptr,
    size_t bytes,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr || ptr == nullptr || bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    buffer->ptr = ptr;
    buffer->bytes = bytes;
    buffer->mapped_bytes = bytes;
    buffer->kind = OO_BUFFER_KIND_WRAPPED;
    buffer->group = node->group;
    buffer->owner_device = node->device;

    *out_buffer = buffer;
    return OO_SUCCESS;
}

void oo_buffer_destroy(
    oo_buffer_t* buffer) {
    if (buffer == nullptr) {
        return;
    }

    if (buffer->kind == OO_BUFFER_KIND_VMM) {
        ooverlap::system::free_peer_visible_buffer(buffer->mapped);
    }

    delete buffer;
}

void* oo_buffer_ptr(
    const oo_buffer_t* buffer) {
    return (buffer != nullptr) ? buffer->ptr : nullptr;
}

size_t oo_buffer_bytes(
    const oo_buffer_t* buffer) {
    return (buffer != nullptr) ? buffer->bytes : 0;
}

size_t oo_buffer_mapped_bytes(
    const oo_buffer_t* buffer) {
    return (buffer != nullptr) ? buffer->mapped_bytes : 0;
}

oo_buffer_kind_t oo_buffer_kind(
    const oo_buffer_t* buffer) {
    return (buffer != nullptr) ? buffer->kind : OO_BUFFER_KIND_WRAPPED;
}

oo_status_t oo_allreduce(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream) {
    if (node == nullptr || node->group == nullptr ||
        local == nullptr || peer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (node->group->num_devices != 2) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (dtype != OO_DTYPE_FLOAT16 || op != OO_REDUCE_SUM) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (count == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (local->ptr == nullptr || peer->ptr == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (!same_group(local->group, node->group) ||
        !same_group(peer->group, node->group)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int peer_rank = node->rank ^ 1;
    const int expected_local_device = node->device;
    const int expected_peer_device = node->group->devices[peer_rank];

    if (local->owner_device != expected_local_device ||
        peer->owner_device != expected_peer_device) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    size_t required_bytes = 0;
    if (!checked_mul_size(count, dtype_size(dtype), &required_bytes)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (local->bytes < required_bytes || peer->bytes < required_bytes) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        cudaError_t err = ooverlap::enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
            reinterpret_cast<const half*>(local->ptr),
            reinterpret_cast<half*>(local->ptr),
            reinterpret_cast<half*>(peer->ptr),
            count,
            node->rank,
            node->group->devices[0],
            node->group->devices[1],
            stream);

        return cuda_status_to_oo(err);
    } catch (const std::exception&) {
        return OO_ERROR_CUDA;
    } catch (...) {
        return OO_ERROR_INTERNAL;
    }
}

} // extern "C"
