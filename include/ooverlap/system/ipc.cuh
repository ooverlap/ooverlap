#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__linux__)
#include <unistd.h>   // close()
#else
#error "ooverlap::system::ipc currently assumes Linux/POSIX file descriptor handles."
#endif

#include "cuda_check.cuh"
#include "vmm.cuh"

namespace ooverlap {
namespace system {
namespace ipc {

enum class flavor {
    legacy = 0,
    vmm    = 1
};

template <flavor F>
struct handle;

template <>
struct handle<flavor::legacy> {
    static constexpr flavor flavor_ = flavor::legacy;
    cudaIpcMemHandle_t value{};
};

template <>
struct handle<flavor::vmm> {
    static constexpr flavor flavor_ = flavor::vmm;
    int value = -1;  // POSIX file descriptor
};

using legacy_handle = handle<flavor::legacy>;
using vmm_handle    = handle<flavor::vmm>;

template <typename T>
inline constexpr bool is_ipc_handle_v =
    std::is_same_v<T, legacy_handle> || std::is_same_v<T, vmm_handle>;

inline void check_support(int device_id) {
    vmm::init_driver_once();

    CUdevice device;
    OOVERLAP_CUCHECK(cuDeviceGet(&device, device_id));

    int ipc_event_supported = 0;
    OOVERLAP_CUDACHECK(
        cudaDeviceGetAttribute(
            &ipc_event_supported,
            cudaDevAttrIpcEventSupport,
            device_id));

    int posix_fd_handle_supported = 0;
    OOVERLAP_CUCHECK(
        cuDeviceGetAttribute(
            &posix_fd_handle_supported,
            CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
            device));

    if (!ipc_event_supported || !posix_fd_handle_supported) {
        throw std::runtime_error("CUDA IPC is not supported on this device");
    }
}

template <typename IPC_HANDLE>
inline void export_handle(
    IPC_HANDLE* out_ipc_handle,
    void* ptr
) {
    static_assert(is_ipc_handle_v<IPC_HANDLE>, "Invalid IPC handle type");

    if (out_ipc_handle == nullptr) {
        throw std::invalid_argument("export_handle(ptr): out_ipc_handle is null");
    }
    if (ptr == nullptr) {
        throw std::invalid_argument("export_handle(ptr): ptr is null");
    }

    if constexpr (IPC_HANDLE::flavor_ == flavor::legacy) {
        OOVERLAP_CUDACHECK(cudaIpcGetMemHandle(&out_ipc_handle->value, ptr));
    } else if constexpr (IPC_HANDLE::flavor_ == flavor::vmm) {
        vmm::handle memory_handle{};
        vmm::vm_retrieve_handle(&memory_handle, ptr);

        // Important:
        // cuMemExportToShareableHandle creates a shareable OS handle (FD on Linux).
        // The receiver must close() its imported FD after cuMemImportFromShareableHandle.
        // The sender also owns its exported FD and should close it when the FD is no longer needed.
        OOVERLAP_CUCHECK(
            cuMemExportToShareableHandle(
                &out_ipc_handle->value,
                memory_handle,
                vmm::kShareableHandleType,
                0));

        vmm::vm_free(memory_handle);
    }
}

template <typename IPC_HANDLE>
inline void export_handle(
    IPC_HANDLE* out_ipc_handle,
    vmm::handle& memory_handle
) {
    static_assert(is_ipc_handle_v<IPC_HANDLE>, "Invalid IPC handle type");

    if (out_ipc_handle == nullptr) {
        throw std::invalid_argument("export_handle(memory_handle): out_ipc_handle is null");
    }

    if constexpr (IPC_HANDLE::flavor_ == flavor::vmm) {
        OOVERLAP_CUCHECK(
            cuMemExportToShareableHandle(
                &out_ipc_handle->value,
                memory_handle,
                vmm::kShareableHandleType,
                0));
    } else {
        throw std::runtime_error(
            "export_handle(memory_handle): legacy CUDA IPC cannot export a CUmemGenericAllocationHandle");
    }
}

template <typename IPC_HANDLE>
inline void import_handle(
    void** out_ptr,
    IPC_HANDLE& ipc_handle,
    size_t size,
    const std::vector<int>& accessible_device_ids
) {
    static_assert(is_ipc_handle_v<IPC_HANDLE>, "Invalid IPC handle type");

    if (out_ptr == nullptr) {
        throw std::invalid_argument("import_handle(ptr): out_ptr is null");
    }
    if (size == 0) {
        throw std::invalid_argument("import_handle(ptr): size must be > 0");
    }

    if constexpr (IPC_HANDLE::flavor_ == flavor::legacy) {
        OOVERLAP_CUDACHECK(
            cudaIpcOpenMemHandle(
                out_ptr,
                ipc_handle.value,
                cudaIpcMemLazyEnablePeerAccess));
    } else if constexpr (IPC_HANDLE::flavor_ == flavor::vmm) {
        vmm::handle memory_handle{};
        OOVERLAP_CUCHECK(
            cuMemImportFromShareableHandle(
                &memory_handle,
                reinterpret_cast<void*>(static_cast<uintptr_t>(ipc_handle.value)),
                vmm::kShareableHandleType));

        vmm::vm_map(out_ptr, memory_handle, size);
        vmm::vm_set_access(*out_ptr, size, accessible_device_ids);
        vmm::vm_free(memory_handle);

        // After import, the FD is no longer needed.
        if (ipc_handle.value >= 0) {
            ::close(ipc_handle.value);
            ipc_handle.value = -1;
        }
    }
}

template <typename IPC_HANDLE>
inline void import_handle(
    void** out_ptr,
    IPC_HANDLE& ipc_handle,
    size_t size,
    int num_devices
) {
    static_assert(is_ipc_handle_v<IPC_HANDLE>, "Invalid IPC handle type");

    if constexpr (IPC_HANDLE::flavor_ == flavor::legacy) {
        OOVERLAP_CUDACHECK(
            cudaIpcOpenMemHandle(
                out_ptr,
                ipc_handle.value,
                cudaIpcMemLazyEnablePeerAccess));
    } else if constexpr (IPC_HANDLE::flavor_ == flavor::vmm) {
        vmm::handle memory_handle{};
        OOVERLAP_CUCHECK(
            cuMemImportFromShareableHandle(
                &memory_handle,
                reinterpret_cast<void*>(static_cast<uintptr_t>(ipc_handle.value)),
                vmm::kShareableHandleType));

        vmm::vm_map(out_ptr, memory_handle, size);
        vmm::vm_set_access(*out_ptr, size, num_devices);
        vmm::vm_free(memory_handle);

        if (ipc_handle.value >= 0) {
            ::close(ipc_handle.value);
            ipc_handle.value = -1;
        }
    }
}

template <typename IPC_HANDLE>
inline void import_handle(
    vmm::handle* out_memory_handle,
    IPC_HANDLE& ipc_handle
) {
    static_assert(is_ipc_handle_v<IPC_HANDLE>, "Invalid IPC handle type");

    if (out_memory_handle == nullptr) {
        throw std::invalid_argument("import_handle(memory_handle): out_memory_handle is null");
    }

    if constexpr (IPC_HANDLE::flavor_ == flavor::vmm) {
        OOVERLAP_CUCHECK(
            cuMemImportFromShareableHandle(
                out_memory_handle,
                reinterpret_cast<void*>(static_cast<uintptr_t>(ipc_handle.value)),
                vmm::kShareableHandleType));

        if (ipc_handle.value >= 0) {
            ::close(ipc_handle.value);
            ipc_handle.value = -1;
        }
    } else {
        throw std::runtime_error(
            "import_handle(memory_handle): legacy CUDA IPC does not produce a CUmemGenericAllocationHandle");
    }
}

template <flavor F>
inline void free_handle(
    void* ptr,
    size_t size
) {
    if (ptr == nullptr) return;

    if constexpr (F == flavor::legacy) {
        OOVERLAP_CUDACHECK(cudaIpcCloseMemHandle(ptr));
    } else if constexpr (F == flavor::vmm) {
        vmm::vm_unmap(ptr, size);
    }
}

// For the exporting side: close the exported POSIX FD when you no longer need to send it.
inline void close_exported_handle(vmm_handle& h) {
    if (h.value >= 0) {
        ::close(h.value);
        h.value = -1;
    }
}

} // namespace ipc
} // namespace system
} // namespace ooverlap
