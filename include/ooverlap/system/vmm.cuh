#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <vector>

#include "cuda_check.cuh"

namespace ooverlap {
namespace system {
namespace vmm {

#if defined(__linux__)
inline constexpr CUmemAllocationHandleType kShareableHandleType =
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
#error "ooverlap::system::vmm currently assumes Linux/POSIX shareable handles."
#endif

using handle = CUmemGenericAllocationHandle;

inline void init_driver_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        OOVERLAP_CUCHECK(cuInit(0));
    });
}

inline size_t round_up_to(size_t x, size_t granularity) {
    return ((x + granularity - 1) / granularity) * granularity;
}

inline void vm_alloc(
    handle* out_handle,
    size_t* out_allocated_size,
    size_t requested_size,
    int owner_device_id
) {
    init_driver_once();

    if (out_handle == nullptr || out_allocated_size == nullptr) {
        throw std::invalid_argument("vm_alloc: output pointer is null");
    }
    if (requested_size == 0) {
        throw std::invalid_argument("vm_alloc: requested_size must be > 0");
    }

    CUmemAllocationProp prop = {};
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = owner_device_id;
    prop.requestedHandleTypes = kShareableHandleType;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    size_t granularity = 0;
    OOVERLAP_CUCHECK(
        cuMemGetAllocationGranularity(
            &granularity,
            &prop,
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    *out_allocated_size = round_up_to(requested_size, granularity);

    OOVERLAP_CUCHECK(cuMemCreate(out_handle, *out_allocated_size, &prop, 0));
}

inline void vm_map(
    void** out_ptr,
    const handle& h,
    size_t mapped_size
) {
    init_driver_once();

    if (out_ptr == nullptr) {
        throw std::invalid_argument("vm_map: out_ptr is null");
    }
    if (mapped_size == 0) {
        throw std::invalid_argument("vm_map: mapped_size must be > 0");
    }

    CUdeviceptr addr = 0;
    OOVERLAP_CUCHECK(cuMemAddressReserve(&addr, mapped_size, 0, 0, 0));
    OOVERLAP_CUCHECK(cuMemMap(addr, mapped_size, 0, h, 0));

    *out_ptr = reinterpret_cast<void*>(addr);
}

inline void vm_set_access(
    void* ptr,
    size_t size,
    const std::vector<int>& device_ids
) {
    init_driver_once();

    if (ptr == nullptr) {
        throw std::invalid_argument("vm_set_access: ptr is null");
    }
    if (size == 0) {
        throw std::invalid_argument("vm_set_access: size must be > 0");
    }
    if (device_ids.empty()) {
        throw std::invalid_argument("vm_set_access: device_ids must not be empty");
    }

    std::vector<CUmemAccessDesc> descs(device_ids.size());
    for (size_t i = 0; i < device_ids.size(); ++i) {
        descs[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        descs[i].location.id   = device_ids[i];
        descs[i].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    OOVERLAP_CUCHECK(
        cuMemSetAccess(
            reinterpret_cast<CUdeviceptr>(ptr),
            size,
            descs.data(),
            static_cast<size_t>(descs.size())));
}

inline void vm_set_access(
    void* ptr,
    size_t size,
    int num_devices
) {
    if (num_devices <= 0) {
        throw std::invalid_argument("vm_set_access: num_devices must be > 0");
    }

    std::vector<int> device_ids(num_devices);
    for (int i = 0; i < num_devices; ++i) {
        device_ids[i] = i;
    }
    vm_set_access(ptr, size, device_ids);
}

inline void vm_retrieve_handle(
    handle* out_handle,
    void* ptr
) {
    init_driver_once();

    if (out_handle == nullptr) {
        throw std::invalid_argument("vm_retrieve_handle: out_handle is null");
    }
    if (ptr == nullptr) {
        throw std::invalid_argument("vm_retrieve_handle: ptr is null");
    }

    // Every retain must eventually be matched by cuMemRelease.
    OOVERLAP_CUCHECK(cuMemRetainAllocationHandle(out_handle, ptr));
}

inline void vm_unmap(
    void* ptr,
    size_t size
) {
    init_driver_once();

    if (ptr == nullptr || size == 0) return;

    OOVERLAP_CUCHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size));
    OOVERLAP_CUCHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size));
}

inline void vm_free(handle& h) {
    init_driver_once();
    OOVERLAP_CUCHECK(cuMemRelease(h));
}

inline handle vm_alloc_map_set_access_keep_handle(
    void** out_ptr,
    size_t* out_allocated_size,
    size_t requested_size,
    int owner_device_id,
    const std::vector<int>& device_ids
) {
    handle h{};
    vm_alloc(&h, out_allocated_size, requested_size, owner_device_id);
    vm_map(out_ptr, h, *out_allocated_size);
    vm_set_access(*out_ptr, *out_allocated_size, device_ids);
    return h;
}

inline void vm_alloc_map_set_access(
    void** out_ptr,
    size_t* out_allocated_size,
    size_t requested_size,
    int owner_device_id,
    const std::vector<int>& device_ids
) {
    handle h = vm_alloc_map_set_access_keep_handle(
        out_ptr,
        out_allocated_size,
        requested_size,
        owner_device_id,
        device_ids);

    // Safe only when you do NOT need to export/share the handle later.
    vm_free(h);
}

inline bool multicast_supported(int device_id) {
    init_driver_once();

    CUdevice device;
    OOVERLAP_CUCHECK(cuDeviceGet(&device, device_id));

    int supported = 0;
    OOVERLAP_CUCHECK(
        cuDeviceGetAttribute(
            &supported,
            CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
            device));

    return supported != 0;
}

inline void multicast_check(int device_id) {
    if (!multicast_supported(device_id)) {
        throw std::runtime_error("Device does not support multicast");
    }
}

inline void multicast_create_handle(
    handle* out_handle,
    size_t* out_allocated_size,
    size_t requested_size,
    int num_devices
) {
    init_driver_once();

    if (out_handle == nullptr || out_allocated_size == nullptr) {
        throw std::invalid_argument("multicast_create_handle: output pointer is null");
    }
    if (num_devices <= 1) {
        throw std::invalid_argument("multicast_create_handle: num_devices must be >= 2");
    }

    CUmulticastObjectProp prop = {};
    prop.numDevices  = num_devices;
    prop.handleTypes = kShareableHandleType;

    size_t granularity = 0;
    OOVERLAP_CUCHECK(
        cuMulticastGetGranularity(
            &granularity,
            &prop,
            CU_MULTICAST_GRANULARITY_RECOMMENDED));

    *out_allocated_size = round_up_to(requested_size, granularity);
    prop.size = *out_allocated_size;

    OOVERLAP_CUCHECK(cuMulticastCreate(out_handle, &prop));
}

inline void multicast_bind_device(
    const handle& multicast_handle,
    int device_id
) {
    init_driver_once();

    CUdevice device;
    OOVERLAP_CUCHECK(cuDeviceGet(&device, device_id));
    OOVERLAP_CUCHECK(cuMulticastAddDevice(multicast_handle, device));
}

inline void multicast_bind_memory(
    const handle& multicast_handle,
    const handle& memory_handle,
    size_t size
) {
    init_driver_once();
    OOVERLAP_CUCHECK(cuMulticastBindMem(multicast_handle, 0, memory_handle, 0, size, 0));
}

inline void multicast_bind_address(
    const handle& multicast_handle,
    void* ptr,
    size_t size
) {
    handle memory_handle{};
    vm_retrieve_handle(&memory_handle, ptr);
    multicast_bind_memory(multicast_handle, memory_handle, size);
    vm_free(memory_handle);
}

inline void multicast_unbind_device(
    const handle& multicast_handle,
    size_t size,
    int device_id
) {
    init_driver_once();

    CUdevice device;
    OOVERLAP_CUCHECK(cuDeviceGet(&device, device_id));
    OOVERLAP_CUCHECK(cuMulticastUnbind(multicast_handle, device, 0, size));
}

// Optional small RAII helper.
struct mapped_allocation {
    void*  ptr            = nullptr;
    size_t requested_size = 0;
    size_t mapped_size    = 0;
    int    owner_device   = -1;

    mapped_allocation() = default;
    mapped_allocation(const mapped_allocation&) = delete;
    mapped_allocation& operator=(const mapped_allocation&) = delete;

    ~mapped_allocation() {
        if (ptr != nullptr && mapped_size != 0) {
            vm_unmap(ptr, mapped_size);
        }
    }
};

} // namespace vmm
} // namespace system
} // namespace ooverlap
