#include "comm/ooverlap_comm_internal.h"

#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr size_t kOoReadySignalBytes = sizeof(int);

struct oo_ready_signal_ipc_desc {
    cudaIpcMemHandle_t handle{};
    std::uint64_t bytes = 0;
    int owner_rank = -1;
    int owner_device = -1;
};

oo_status_t report_cuda_error(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(
            stderr,
            "[ooverlap][ipc] %s failed: %s\n",
            what,
            cudaGetErrorString(err));
        std::fflush(stderr);
        return OO_ERROR_CUDA;
    }
    return OO_SUCCESS;
}

oo_status_t report_exception(const char* where, const std::exception& e) {
    std::fprintf(
        stderr,
        "[ooverlap][ipc] %s threw: %s\n",
        where,
        e.what());
    std::fflush(stderr);
    return OO_ERROR_INTERNAL;
}

oo_status_t report_unknown_exception(const char* where) {
    std::fprintf(
        stderr,
        "[ooverlap][ipc] %s threw unknown exception\n",
        where);
    std::fflush(stderr);
    return OO_ERROR_INTERNAL;
}

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

bool reduce_op_supported_for_dtype(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    if (op == OO_REDUCE_ADD) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16 ||
               dtype == OO_DTYPE_FLOAT32;
    }

    if (op == OO_REDUCE_MIN || op == OO_REDUCE_MAX) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16;
    }

    return false;
}

bool same_group(const oo_group_t* a, const oo_group_t* b) {
    return a != nullptr && b != nullptr && a == b;
}

int find_rank_for_device(const oo_group_t* group, int device) {
    if (group == nullptr) {
        return -1;
    }

    for (int r = 0; r < group->num_devices; ++r) {
        if (group->devices[r] == device) {
            return r;
        }
    }

    return -1;
}

std::vector<int> group_devices_vector(const oo_group_t* group) {
    std::vector<int> out;
    if (group == nullptr) {
        return out;
    }

    out.reserve(static_cast<size_t>(group->num_devices));
    for (int i = 0; i < group->num_devices; ++i) {
        out.push_back(group->devices[i]);
    }

    return out;
}

void clear_ready_signal_slot(oo_ready_signal& slot) {
    slot.ptr = nullptr;
    slot.bytes = 0;
    slot.mapped_bytes = 0;
    slot.owner_rank = -1;
    slot.owner_device = -1;
    slot.kind = oo_ready_signal_kind::empty;
    slot.owned_legacy_ptr = nullptr;
}

void free_ready_signal_slot(oo_ready_signal& slot) {
    switch (slot.kind) {
        case oo_ready_signal_kind::owned_vmm:
            ooverlap::system::free_peer_visible_buffer(slot.owned_vmm);
            break;

        case oo_ready_signal_kind::owned_legacy:
            if (slot.owned_legacy_ptr != nullptr) {
                if (slot.owner_device >= 0) {
                    cudaSetDevice(slot.owner_device);
                }
                cudaFree(slot.owned_legacy_ptr);
                slot.owned_legacy_ptr = nullptr;
            }
            break;
            
        case oo_ready_signal_kind::imported_legacy:
            if (slot.ptr != nullptr) {
                cudaIpcCloseMemHandle(slot.ptr);
            }
            break;

        case oo_ready_signal_kind::imported_vmm:
            slot.imported.reset();
            break;

        case oo_ready_signal_kind::empty:
        default:
            break;
    }

    clear_ready_signal_slot(slot);
}

void mirror_ready_signal_for_compat(oo_group_t* group, int rank) {
    if (group == nullptr || rank < 0 || rank >= group->num_devices) {
        return;
    }

    oo_ready_signal& slot = group->ready_signal_slots[rank];
    group->ready_signals[rank].ptr = slot.ptr;
    group->ready_signals[rank].mapped_size = slot.mapped_bytes;
    group->ready_signals[rank].requested_size = slot.bytes;
    group->ready_signals[rank].owner_device = slot.owner_device;
}

void clear_ready_signal_compat_mirror(oo_group_t* group, int rank) {
    if (group == nullptr || rank < 0 || rank >= kOoMaxLocalDevices) {
        return;
    }

    group->ready_signals[rank].ptr = nullptr;
    group->ready_signals[rank].mapped_size = 0;
    group->ready_signals[rank].requested_size = 0;
    group->ready_signals[rank].owner_device = -1;
}

void free_group_ready_signals_same_process(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    for (int r = 0; r < group->num_devices; ++r) {
        free_ready_signal_slot(group->ready_signal_slots[r]);
        clear_ready_signal_compat_mirror(group, r);
    }
}

void free_group_ready_signals_ipc(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    /*
     * Two-phase cleanup:
     *
     * 1. All ranks arrive.
     * 2. Everyone closes imported mappings.
     * 3. All ranks arrive again.
     * 4. Each rank frees only its own exported allocation.
     *
     * This avoids freeing a local cudaMalloc allocation while a peer process
     * still has an open cudaIpc mapping to it.
     */
    try {
        if (group->broker) {
            group->broker->sync();
        }
    } catch (...) {
        // Destructors/cleanup should be best-effort.
    }

    for (int r = 0; r < group->num_devices; ++r) {
        oo_ready_signal& slot = group->ready_signal_slots[r];
        if (slot.kind == oo_ready_signal_kind::imported_legacy ||
            slot.kind == oo_ready_signal_kind::imported_vmm) {
            free_ready_signal_slot(slot);
            clear_ready_signal_compat_mirror(group, r);
        }
    }

    try {
        if (group->broker) {
            group->broker->sync();
        }
    } catch (...) {
    }

    const int local_rank = group->local_rank;
    if (local_rank >= 0 && local_rank < group->num_devices) {
        oo_ready_signal& local_slot = group->ready_signal_slots[local_rank];
        if (local_slot.kind == oo_ready_signal_kind::owned_legacy ||
            local_slot.kind == oo_ready_signal_kind::owned_vmm) {
            free_ready_signal_slot(local_slot);
            clear_ready_signal_compat_mirror(group, local_rank);
        }
    }

    try {
        if (group->broker) {
            group->broker->sync();
        }
    } catch (...) {
    }
}

void free_group_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc) {
        free_group_ready_signals_ipc(group);
    } else {
        free_group_ready_signals_same_process(group);
    }
}

oo_status_t init_group_ready_signals_same_process(oo_group_t* group) {
    if (group == nullptr || group->num_devices <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<int> access_devices = group_devices_vector(group);

        for (int rank = 0; rank < group->num_devices; ++rank) {
            oo_ready_signal& slot = group->ready_signal_slots[rank];

            slot.owned_vmm =
                ooverlap::system::alloc_peer_visible_buffer(
                    kOoReadySignalBytes,
                    group->devices[rank],
                    access_devices);

            slot.ptr = slot.owned_vmm.ptr;
            slot.bytes = kOoReadySignalBytes;
            slot.mapped_bytes = slot.owned_vmm.mapped_size;
            slot.owner_rank = rank;
            slot.owner_device = group->devices[rank];
            slot.kind = oo_ready_signal_kind::owned_vmm;

            mirror_ready_signal_for_compat(group, rank);

            cudaError_t set_err = cudaSetDevice(group->devices[rank]);
            if (set_err != cudaSuccess) {
                free_group_ready_signals_same_process(group);
                return OO_ERROR_CUDA;
            }

            cudaError_t memset_err = cudaMemset(
                slot.ptr,
                0,
                slot.mapped_bytes);
            if (memset_err != cudaSuccess) {
                free_group_ready_signals_same_process(group);
                return OO_ERROR_CUDA;
            }
        }
    } catch (const std::bad_alloc&) {
        free_group_ready_signals_same_process(group);
        return OO_ERROR_INTERNAL;
    } catch (const std::exception&) {
        free_group_ready_signals_same_process(group);
        return OO_ERROR_CUDA;
    } catch (...) {
        free_group_ready_signals_same_process(group);
        return OO_ERROR_INTERNAL;
    }

    return OO_SUCCESS;
}

oo_status_t init_group_ready_signals_ipc(oo_group_t* group) {
    if (group == nullptr ||
        group->num_devices <= 0 ||
        group->local_rank < 0 ||
        group->local_rank >= group->num_devices ||
        !group->broker) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int local_rank = group->local_rank;
    const int signal_owner_rank = 0;
    const int signal_owner_device = group->devices[signal_owner_rank];
    const size_t signal_block_bytes =
        static_cast<size_t>(group->num_devices) * kOoReadySignalBytes;

    auto cleanup_ready_signals_local_only = [&]() {
        for (int r = 0; r < group->num_devices; ++r) {
            oo_ready_signal& slot = group->ready_signal_slots[r];

            if (slot.kind == oo_ready_signal_kind::owned_vmm ||
                slot.kind == oo_ready_signal_kind::owned_legacy ||
                slot.kind == oo_ready_signal_kind::imported_legacy ||
                slot.kind == oo_ready_signal_kind::imported_vmm) {
                free_ready_signal_slot(slot);
                clear_ready_signal_compat_mirror(group, r);
            } else {
                clear_ready_signal_slot(slot);
                clear_ready_signal_compat_mirror(group, r);
            }
        }
    };

    try {
        std::vector<int> access_devices = group_devices_vector(group);

        ooverlap::system::vmm_peer_buffer_descriptor local_desc{};
        ooverlap::system::ipc::vmm_handle local_fd{};

        void* signal_base = nullptr;
        ooverlap::system::imported_peer_buffer imported_signal_block{};

        if (local_rank == signal_owner_rank) {
            cudaError_t err = cudaSetDevice(signal_owner_device);
            if (err != cudaSuccess) {
                return report_cuda_error(err, "cudaSetDevice(signal_owner_device)");
            }

            ooverlap::system::mapped_peer_buffer block =
                ooverlap::system::alloc_peer_visible_buffer(
                    signal_block_bytes,
                    signal_owner_device,
                    access_devices);

            err = cudaMemset(block.ptr, 0, block.mapped_size);
            if (err != cudaSuccess) {
                ooverlap::system::free_peer_visible_buffer(block);
                return report_cuda_error(err, "cudaMemset(VMM ready-signal block)");
            }

            local_desc =
                ooverlap::system::make_vmm_peer_buffer_descriptor(
                    block,
                    signal_block_bytes);

            local_fd =
                ooverlap::system::export_vmm_peer_buffer_fd(block);

            /*
             * Store ownership on slot 0. Other slots are borrowed views into
             * the same VMM block and must not free it.
             */
            group->ready_signal_slots[signal_owner_rank].owned_vmm = block;
            signal_base = block.ptr;
        }

        /*
         * Broadcast descriptor through the shared-memory broker.
         * Only rank 0's descriptor is meaningful.
         */
        std::vector<ooverlap::system::vmm_peer_buffer_descriptor> all_desc(
            static_cast<size_t>(group->num_devices));

        group->broker->exchange_data(
            all_desc.data(),
            &local_desc,
            sizeof(local_desc));

        const ooverlap::system::vmm_peer_buffer_descriptor signal_desc =
            all_desc[signal_owner_rank];

        if (signal_desc.bytes != signal_block_bytes ||
            signal_desc.mapped_size == 0 ||
            signal_desc.owner_device != signal_owner_device) {
            std::fprintf(
                stderr,
                "[ooverlap][ipc] bad VMM ready-signal descriptor: "
                "bytes=%llu mapped=%llu owner_device=%d expected_bytes=%llu expected_device=%d\n",
                static_cast<unsigned long long>(signal_desc.bytes),
                static_cast<unsigned long long>(signal_desc.mapped_size),
                signal_desc.owner_device,
                static_cast<unsigned long long>(signal_block_bytes),
                signal_owner_device);
            std::fflush(stderr);
            cleanup_ready_signals_local_only();
            return OO_ERROR_INTERNAL;
        }

        /*
         * Transfer rank 0's VMM FD using SCM_RIGHTS.
         * Non-root passes src_fd=-1; broadcast_fd only checks src_fd on root.
         */
        int imported_fd = -1;
        if (local_rank == signal_owner_rank) {
            int ignored_fd = -1;
            group->broker->broadcast_fd(
                &ignored_fd,
                local_fd.value,
                signal_owner_rank);

            /*
             * broadcast_fd consumes/closes the exported fd on the sender.
             */
            local_fd.value = -1;
        } else {
            group->broker->broadcast_fd(
                &imported_fd,
                -1,
                signal_owner_rank);

            imported_signal_block =
                ooverlap::system::import_vmm_peer_buffer(
                    imported_fd,
                    signal_desc,
                    access_devices);

            signal_base = imported_signal_block.ptr;
        }

        if (signal_base == nullptr) {
            cleanup_ready_signals_local_only();
            return OO_ERROR_INTERNAL;
        }

        /*
         * Build two int slots inside one shared signal block:
         *
         *   slot 0: rank 0 epoch
         *   slot 1: rank 1 epoch
         *
         * Rank 0 owns the VMM block. Rank 1 owns one imported VMM mapping.
         * The non-owning slots are intentionally kind=empty so cleanup frees
         * the block exactly once per process.
         */
        for (int rank = 0; rank < group->num_devices; ++rank) {
            oo_ready_signal& slot = group->ready_signal_slots[rank];

            slot.ptr = reinterpret_cast<void*>(
                reinterpret_cast<std::uint8_t*>(signal_base) +
                static_cast<size_t>(rank) * kOoReadySignalBytes);

            slot.bytes = kOoReadySignalBytes;
            slot.mapped_bytes = kOoReadySignalBytes;
            slot.owner_rank = rank;
            slot.owner_device = group->devices[rank];
            slot.kind = oo_ready_signal_kind::empty;

            mirror_ready_signal_for_compat(group, rank);
        }

        if (local_rank == signal_owner_rank) {
            oo_ready_signal& owner_slot =
                group->ready_signal_slots[signal_owner_rank];
            owner_slot.kind = oo_ready_signal_kind::owned_vmm;
        } else {
            oo_ready_signal& imported_owner_slot =
                group->ready_signal_slots[signal_owner_rank];
            imported_owner_slot.kind = oo_ready_signal_kind::imported_vmm;
            imported_owner_slot.imported = std::move(imported_signal_block);
        }

        /*
         * After all slots are constructed, make sure both processes see a
         * complete ready-signal setup before returning from group creation.
         */
        group->broker->sync();

        return OO_SUCCESS;

    } catch (const std::bad_alloc&) {
        cleanup_ready_signals_local_only();
        return OO_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        cleanup_ready_signals_local_only();
        return report_exception("init_group_ready_signals_ipc", e);
    } catch (...) {
        cleanup_ready_signals_local_only();
        return report_unknown_exception("init_group_ready_signals_ipc");
    }
}

oo_status_t init_group_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc) {
        return init_group_ready_signals_ipc(group);
    }

    return init_group_ready_signals_same_process(group);
}

bool buffer_is_valid_for_allreduce(
    const oo_buffer_t* buffer,
    size_t required_bytes) {
    if (buffer == nullptr) {
        return false;
    }
    if (buffer->ptr == nullptr) {
        return false;
    }
    if (buffer->bytes < required_bytes) {
        return false;
    }
    if (buffer->mapped_bytes < required_bytes) {
        return false;
    }
    if (buffer->owner_device < 0) {
        return false;
    }
    if (buffer->system_kind == ooverlap::system::peer_buffer_kind::empty) {
        return false;
    }

    return true;
}

bool buffer_is_imported(const oo_buffer_t* buffer) {
    return buffer != nullptr &&
           (buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_legacy ||
            buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_vmm);
}

void initialize_buffer_from_view(
    oo_buffer_t* buffer,
    oo_node_t* node,
    const ooverlap::system::peer_buffer_view& view,
    oo_buffer_kind_t public_kind) {
    buffer->ptr = view.ptr;
    buffer->bytes = view.bytes;
    buffer->mapped_bytes = view.mapped_size;
    buffer->kind = public_kind;
    buffer->group = node->group;
    buffer->owner_device = view.owner_device;
    buffer->owner_rank = find_rank_for_device(node->group, view.owner_device);
    buffer->system_kind = view.kind;
}

} // namespace

oo_status_t oo_buffer_export_legacy_descriptor(
    oo_buffer_t* buffer,
    ooverlap::system::legacy_peer_buffer_descriptor* out_desc) {
    if (buffer == nullptr || out_desc == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (buffer->ptr == nullptr || buffer->bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (buffer->owner_device < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    /*
     * Legacy cudaIpcGetMemHandle is meant for cudaMalloc/PyTorch-style device
     * allocations. VMM allocations should use the VMM FD path instead.
     */
    if (buffer->system_kind == ooverlap::system::peer_buffer_kind::owned_vmm ||
        buffer->kind == OO_BUFFER_KIND_VMM) {
        return OO_ERROR_UNSUPPORTED;
    }

    try {
        *out_desc = ooverlap::system::export_legacy_peer_buffer(
            buffer->ptr,
            buffer->bytes,
            buffer->owner_device,
            buffer->mapped_bytes != 0 ? buffer->mapped_bytes : buffer->bytes);
    } catch (const std::bad_alloc&) {
        return OO_ERROR_INTERNAL;
    } catch (const std::exception&) {
        return OO_ERROR_CUDA;
    } catch (...) {
        return OO_ERROR_INTERNAL;
    }

    return OO_SUCCESS;
}

oo_status_t oo_buffer_import_legacy_descriptor(
    oo_node_t* node,
    const ooverlap::system::legacy_peer_buffer_descriptor& desc,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (desc.bytes == 0 || desc.mapped_size == 0 || desc.owner_device < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int owner_rank = find_rank_for_device(node->group, desc.owner_device);
    if (owner_rank < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    try {
        std::vector<int> access_devices = group_devices_vector(node->group);

        cudaError_t set_err = cudaSetDevice(node->device);
        if (set_err != cudaSuccess) {
            delete buffer;
            return OO_ERROR_CUDA;
        }

        ooverlap::system::imported_peer_buffer imported =
            ooverlap::system::import_legacy_peer_buffer(
                desc,
                access_devices);

        ooverlap::system::peer_buffer_view view = imported.view();
        initialize_buffer_from_view(buffer, node, view, OO_BUFFER_KIND_WRAPPED);

        buffer->owner_rank = owner_rank;
        buffer->imported = std::move(imported);

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

oo_status_t oo_buffer_adopt_imported_peer_buffer(
    oo_node_t* node,
    ooverlap::system::imported_peer_buffer&& imported,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (!imported.valid()) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int owner_rank = find_rank_for_device(node->group, imported.owner_device);
    if (owner_rank < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    ooverlap::system::peer_buffer_view view = imported.view();
    initialize_buffer_from_view(buffer, node, view, OO_BUFFER_KIND_WRAPPED);
    buffer->owner_rank = owner_rank;
    buffer->imported = std::move(imported);

    *out_buffer = buffer;
    return OO_SUCCESS;
}

extern "C" {

size_t oo_dtype_size(
    oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
            return sizeof(half);
        case OO_DTYPE_BFLOAT16:
            return sizeof(__nv_bfloat16);
        case OO_DTYPE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

int oo_allreduce_supported(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    return reduce_op_supported_for_dtype(dtype, op) ? 1 : 0;
}

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
    group->bootstrap_kind = oo_group_bootstrap_kind::same_process;
    group->local_rank = -1;
    group->local_world_size = num_devices;

    for (int i = 0; i < num_devices; ++i) {
        group->devices[i] = devices[i];
    }

    oo_status_t sig_status = init_group_ready_signals(group);
    if (sig_status != OO_SUCCESS) {
        delete group;
        return sig_status;
    }

    *out_group = group;
    return OO_SUCCESS;
}

oo_status_t oo_group_create_ipc(
    const int* devices,
    int num_devices,
    int local_rank,
    const char* broker_key,
    oo_group_t** out_group) {
    if (out_group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_group = nullptr;

    if (devices == nullptr || broker_key == nullptr || num_devices <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (num_devices != 2) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (num_devices > kOoMaxLocalDevices) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (local_rank < 0 || local_rank >= num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
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
    group->bootstrap_kind = oo_group_bootstrap_kind::multiprocess_ipc;
    group->local_rank = local_rank;
    group->local_world_size = num_devices;

    for (int i = 0; i < num_devices; ++i) {
        group->devices[i] = devices[i];
    }

    try {
        group->broker = std::make_unique<ooverlap::system::Broker>(
            local_rank,
            num_devices,
            broker_key);
    } catch (const std::bad_alloc&) {
        delete group;
        return OO_ERROR_INTERNAL;
    } catch (const std::exception&) {
        delete group;
        return OO_ERROR_INTERNAL;
    } catch (...) {
        delete group;
        return OO_ERROR_INTERNAL;
    }

    oo_status_t sig_status = init_group_ready_signals(group);
    if (sig_status != OO_SUCCESS) {
        delete group;
        return sig_status;
    }

    *out_group = group;
    return OO_SUCCESS;
}

void oo_group_destroy(
    oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    free_group_ready_signals(group);
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

    /*
     * In multiprocess IPC mode, this process should only create/use its local
     * rank. The peer rank lives in the peer process.
     */
    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc &&
        rank != group->local_rank) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_node_t* node = new (std::nothrow) oo_node_t;
    if (node == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    node->group = group;
    node->rank = rank;
    node->device = group->devices[rank];
    node->collective_epoch = 0;

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
        std::vector<int> access_devices = group_devices_vector(node->group);

        buffer->mapped = ooverlap::system::alloc_peer_visible_buffer(
            bytes,
            node->device,
            access_devices);

        ooverlap::system::peer_buffer_view view =
            ooverlap::system::make_owned_peer_buffer_view(
                buffer->mapped,
                bytes);

        initialize_buffer_from_view(buffer, node, view, OO_BUFFER_KIND_VMM);
        buffer->owner_rank = node->rank;
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

    if (node == nullptr || node->group == nullptr ||
        ptr == nullptr || bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    ooverlap::system::peer_buffer_view view =
        ooverlap::system::make_wrapped_peer_buffer_view(
            ptr,
            bytes,
            node->device);

    initialize_buffer_from_view(buffer, node, view, OO_BUFFER_KIND_WRAPPED);
    buffer->owner_rank = node->rank;

    *out_buffer = buffer;
    return OO_SUCCESS;
}

void oo_buffer_destroy(
    oo_buffer_t* buffer) {
    if (buffer == nullptr) {
        return;
    }

    if (buffer->system_kind == ooverlap::system::peer_buffer_kind::owned_vmm ||
        buffer->kind == OO_BUFFER_KIND_VMM) {
        ooverlap::system::free_peer_visible_buffer(buffer->mapped);
    } else if (
        buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_legacy ||
        buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_vmm) {
        buffer->imported.reset();
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

    const size_t elem_bytes = oo_dtype_size(dtype);
    if (elem_bytes == 0) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (!reduce_op_supported_for_dtype(dtype, op)) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (count == 0) {
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

    if (local->owner_rank != node->rank ||
        peer->owner_rank != peer_rank) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    /*
     * In multiprocess mode, a peer buffer must be a real imported IPC mapping.
     * This is the guard that prevents accidentally passing a fake wrapped peer
     * pointer and then hanging/faulting inside the TMA kernel.
     */
    if (node->group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc &&
        !buffer_is_imported(peer)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    size_t required_bytes = 0;
    if (!checked_mul_size(count, elem_bytes, &required_bytes)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (!buffer_is_valid_for_allreduce(local, required_bytes) ||
        !buffer_is_valid_for_allreduce(peer, required_bytes)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_ready_signal& local_signal_slot =
        node->group->ready_signal_slots[node->rank];
    oo_ready_signal& peer_signal_slot =
        node->group->ready_signal_slots[peer_rank];

    if (local_signal_slot.ptr == nullptr ||
        peer_signal_slot.ptr == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    if (local_signal_slot.owner_rank != node->rank ||
        peer_signal_slot.owner_rank != peer_rank) {
        return OO_ERROR_INTERNAL;
    }

    const int collective_epoch = ++node->collective_epoch;

    int* local_ready_signal =
        reinterpret_cast<int*>(local_signal_slot.ptr);
    const int* peer_ready_signal =
        reinterpret_cast<const int*>(peer_signal_slot.ptr);

    try {
        cudaError_t set_err = cudaSetDevice(node->device);
        if (set_err != cudaSuccess) {
            return OO_ERROR_CUDA;
        }

        cudaError_t err = ooverlap::enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
            local->ptr,
            local->ptr,
            peer->ptr,
            count,
            dtype,
            op,
            node->rank,
            node->group->devices[0],
            node->group->devices[1],
            stream,
            local_ready_signal,
            peer_ready_signal,
            collective_epoch);

        return cuda_status_to_oo(err);
    } catch (const std::exception&) {
        return OO_ERROR_CUDA;
    } catch (...) {
        return OO_ERROR_INTERNAL;
    }
}

} // extern "C"
