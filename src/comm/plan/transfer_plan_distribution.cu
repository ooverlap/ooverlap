#include "comm/plan/transfer_plan_distribution.h"

#include "comm/plan/transfer_planner.h"
#include "ooverlap/comm.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ooverlap {
namespace comm {
namespace plan {
namespace {

constexpr int kPlanCacheEntries = 16;

bool same_launch_config(
    const LaunchConfig& a,
    const LaunchConfig& b) {
    if (a.threads != b.threads ||
        a.max_ctas != b.max_ctas ||
        a.window_chunks != b.window_chunks ||
        a.chunk_bytes != b.chunk_bytes ||
        a.stage_depth != b.stage_depth ||
        a.plan_for != b.plan_for) {
        return false;
    }

    switch (a.plan_for) {
        case CollectivePlanFor::AllReduce:
            return a.plan.allreduce == b.plan.allreduce;

        case CollectivePlanFor::ReduceScatter:
            return a.plan.reduce_scatter == b.plan.reduce_scatter;

        case CollectivePlanFor::AllGather:
            return a.plan.all_gather == b.plan.all_gather;

        default:
            return false;
    }
}

struct TransferPlanRequestKey {
    CollectivePlanFor collective = CollectivePlanFor::AllReduce;
    int world_size = 0;

    size_t count = 0;
    size_t dtype_size = 0;

    oo_dtype_t dtype = OO_DTYPE_FLOAT16;
    oo_reduce_op_t op = OO_REDUCE_ADD;

    LaunchConfig config{};
};

bool same_request_key(
    const TransferPlanRequestKey& a,
    const TransferPlanRequestKey& b) {
    return a.collective == b.collective &&
           a.world_size == b.world_size &&
           a.count == b.count &&
           a.dtype_size == b.dtype_size &&
           a.dtype == b.dtype &&
           a.op == b.op &&
           same_launch_config(a.config, b.config);
}

TransferPlanRequestKey make_request_key(
    CollectivePlanFor collective,
    const ooverlap::comm::api::CollectiveLaunchState& launch,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    const LaunchConfig& config) {
    TransferPlanRequestKey key{};
    key.collective = collective;
    key.world_size = launch.world_size;
    key.count = count;
    key.dtype_size = launch.dtype_size;
    key.dtype = dtype;
    key.op = op;
    key.config = config;
    return key;
}

TransferPlanBuildInput make_build_input(
    oo_group_t* group,
    const ooverlap::comm::api::CollectiveLaunchState& launch,
    size_t count,
    const LaunchConfig& config,
    CollectivePlanFor collective) {
    TransferPlanBuildInput input{};
    input.topo.topology =
        (group != nullptr && group->topology_valid) ? &group->topology : nullptr;
    input.topo.rank_devices = group != nullptr ? group->devices : nullptr;
    input.topo.world_size = group != nullptr ? group->num_devices : 0;

    input.collective = collective;
    input.launch_config = config;
    input.world_size = launch.world_size;
    input.count = count;
    input.dtype_size = launch.dtype_size;
    input.out_of_place = false;

    input.staging_slot_count =
        launch.staging_slot_count < kPlannerMaxStagingSlots
            ? launch.staging_slot_count
            : kPlannerMaxStagingSlots;
    
    if (input.staging_slot_count < 0) {
        input.staging_slot_count = 0;
    }
    
    for (int slot = 0; slot < input.staging_slot_count; ++slot) {
        input.staging_bytes[slot] =
            launch.staging_bytes[slot];
        input.staging_numa_nodes[slot] =
            launch.staging_numa_nodes[slot];
    }

    return input;
}

template <int MaxTransferTasks>
struct CachedPlanEntry {
    bool valid = false;
    TransferPlanRequestKey key{};
    TransferPlan<MaxTransferTasks> plan{};
};

template <int MaxTransferTasks>
struct SameProcessPlanCache {
    std::mutex mutex;
    int next_victim = 0;
    CachedPlanEntry<MaxTransferTasks> entries[kPlanCacheEntries] = {};
};

template <int MaxTransferTasks, typename Builder>
oo_status_t get_or_build_cached_same_process_plan(
    SameProcessPlanCache<MaxTransferTasks>* cache,
    const TransferPlanRequestKey& key,
    TransferPlan<MaxTransferTasks>** out_plan,
    Builder&& builder) {
    if (cache == nullptr || key.world_size <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(cache->mutex);

    for (int i = 0; i < kPlanCacheEntries; ++i) {
        CachedPlanEntry<MaxTransferTasks>& entry =
            cache->entries[i];

        if (entry.valid && same_request_key(entry.key, key)) {
            *out_plan = &entry.plan;
            return OO_SUCCESS;
        }
    }

    int slot = -1;

    for (int i = 0; i < kPlanCacheEntries; ++i) {
        if (!cache->entries[i].valid) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        slot = cache->next_victim;
        cache->next_victim =
            (cache->next_victim + 1) % kPlanCacheEntries;
    }

    CachedPlanEntry<MaxTransferTasks>& entry =
        cache->entries[slot];

    entry.valid = false;
    entry.key = key;
    transfer_plan_clear(&entry.plan);

    const oo_status_t status =
        builder(&entry.plan);

    if (status != OO_SUCCESS) {
        return status;
    }

    entry.valid = true;
    *out_plan = &entry.plan;
    return OO_SUCCESS;
}

class SameProcessTransferPlanDistributionBackend final
    : public TransferPlanDistributionBackend {
public:
    oo_status_t get_allreduce_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t op,
        const ooverlap::comm::LaunchConfig& config,
        AllreduceTransferPlan** out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::AllReduce) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::AllReduce,
                launch,
                count,
                dtype,
                op,
                config);

        return get_or_build_cached_same_process_plan(
            &allreduce_,
            key,
            out_plan,
            [group, &launch, count, &config](AllreduceTransferPlan* plan) {
                if (group == nullptr ||
                    group->num_devices != launch.world_size) {
                    return OO_ERROR_INVALID_ARGUMENT;
                }

                const TransferPlanBuildInput input =
                    make_build_input(
                        group,
                        launch,
                        count,
                        config,
                        CollectivePlanFor::AllReduce);

                return build_allreduce_transfer_plan(plan, input)
                    ? OO_SUCCESS
                    : OO_ERROR_UNSUPPORTED;
            });
    }

    oo_status_t get_reduce_scatter_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t op,
        const ooverlap::comm::LaunchConfig& config,
        ReduceScatterTransferPlan** out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            out_plan == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::ReduceScatter) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::ReduceScatter,
                launch,
                count,
                dtype,
                op,
                config);

        return get_or_build_cached_same_process_plan(
            &reduce_scatter_,
            key,
            out_plan,
            [group, &launch, count, &config](ReduceScatterTransferPlan* plan) {
                if (group == nullptr ||
                    group->num_devices != launch.world_size) {
                    return OO_ERROR_INVALID_ARGUMENT;
                }

                const TransferPlanBuildInput input =
                    make_build_input(
                        group,
                        launch,
                        count,
                        config,
                        CollectivePlanFor::ReduceScatter);

                return build_reduce_scatter_transfer_plan(plan, input)
                    ? OO_SUCCESS
                    : OO_ERROR_UNSUPPORTED;
            });
    }

    oo_status_t get_all_gather_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        const ooverlap::comm::LaunchConfig& config,
        AllGatherTransferPlan** out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            out_plan == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::AllGather) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::AllGather,
                launch,
                count,
                dtype,
                OO_REDUCE_ADD,
                config);

        return get_or_build_cached_same_process_plan(
            &all_gather_,
            key,
            out_plan,
            [group, &launch, count, &config](AllGatherTransferPlan* plan) {
                if (group == nullptr ||
                    group->num_devices != launch.world_size) {
                    return OO_ERROR_INVALID_ARGUMENT;
                }

                const TransferPlanBuildInput input =
                    make_build_input(
                        group,
                        launch,
                        count,
                        config,
                        CollectivePlanFor::AllGather);

                return build_all_gather_transfer_plan(plan, input)
                    ? OO_SUCCESS
                    : OO_ERROR_UNSUPPORTED;
            });
    }

private:
    SameProcessPlanCache<kTmaMultiGpuAllReduceMaxTransferTasks> allreduce_{};
    SameProcessPlanCache<kTmaMultiGpuReduceScatterMaxTransferTasks> reduce_scatter_{};
    SameProcessPlanCache<kTmaMultiGpuAllGatherMaxTransferTasks> all_gather_{};
};


/*
 * OOVERLAP_IPC_SHARED_PLAN_BACKEND_IMPL_PATCH:
 *
 * IPC shared-plan distribution backend.
 *
 * This is intentionally different from Broker::exchange_data-based plan
 * movement.  A shared POSIX shm object is mapped into every process.  Rank 0
 * builds a TransferPlan into that mapping; every rank reads the same bytes after
 * a Broker barrier.  Broker is used only for synchronization, not for moving the
 * plan payload.
 *
 * Scope:
 *   - setup / plan distribution only
 *   - no user-buffer IPC registration here
 *   - no IPC staging here
 *   - assumes one matched collective sequence per IPC group, same as the rest of
 *     the public API
 */
constexpr std::uint32_t kIpcSharedPlanMagic = 0x4f4f5053u; // "OOPS"
constexpr std::uint32_t kIpcSharedPlanVersion = 1u;

template <typename A, typename B>
struct StaticMaxSize {
    static constexpr std::size_t value =
        sizeof(A) > sizeof(B) ? sizeof(A) : sizeof(B);
};

constexpr std::size_t kIpcSharedPlanPayloadBytesAB =
    StaticMaxSize<AllreduceTransferPlan, ReduceScatterTransferPlan>::value;

constexpr std::size_t kIpcSharedPlanPayloadBytes =
    sizeof(AllGatherTransferPlan) > kIpcSharedPlanPayloadBytesAB
        ? sizeof(AllGatherTransferPlan)
        : kIpcSharedPlanPayloadBytesAB;

struct IpcSharedPlanArena {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t world_size = 0;
    std::uint32_t sequence = 0;

    int status = static_cast<int>(OO_SUCCESS);
    int collective = 0;
    int max_transfer_tasks = 0;
    std::uint64_t payload_bytes = 0;

    alignas(16) unsigned char payload[kIpcSharedPlanPayloadBytes] = {};
};

static_assert(
    sizeof(IpcSharedPlanArena) >= kIpcSharedPlanPayloadBytes,
    "shared plan arena must contain plan payload space");

std::string sanitize_ipc_plan_key_component(const char* key) {
    if (key == nullptr || key[0] == '\0') {
        throw std::runtime_error("IPC plan arena: broker key is empty");
    }

    std::string out;
    out.reserve(std::strlen(key));

    for (const char* p = key; *p; ++p) {
        const char c = *p;
        const bool ok =
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '-' || c == '.';

        out.push_back(ok ? c : '_');
    }

    if (out.empty()) {
        throw std::runtime_error("IPC plan arena: sanitized key is empty");
    }

    return out;
}

std::string make_ipc_plan_shm_name(const char* broker_key) {
    std::string name = "/ooverlap_ipc_plan_";
    name += sanitize_ipc_plan_key_component(broker_key);

    if (name.size() >= 240) {
        throw std::runtime_error("IPC plan arena: shm name too long");
    }

    return name;
}

void wait_ipc_plan_arena_ready(volatile IpcSharedPlanArena* arena) {
    if (arena == nullptr) {
        throw std::runtime_error("IPC plan arena: null mapping");
    }

    while (arena->magic != kIpcSharedPlanMagic ||
           arena->version != kIpcSharedPlanVersion) {
        usleep(1);
    }

    __sync_synchronize();
}

class IpcSharedPlanMapping {
public:
    IpcSharedPlanMapping(
        const char* broker_key,
        int local_rank,
        int world_size)
        : shm_name_(make_ipc_plan_shm_name(broker_key)),
          local_rank_(local_rank),
          world_size_(world_size) {
        if (local_rank_ < 0 ||
            local_rank_ >= world_size_ ||
            world_size_ <= 0 ||
            world_size_ > kOoMaxLocalDevices) {
            throw std::invalid_argument("IPC plan arena: invalid rank/world_size");
        }

        if (local_rank_ == 0) {
            create_rank0();
        } else {
            open_nonzero_rank();
        }

        wait_ipc_plan_arena_ready(arena_);
    }

    IpcSharedPlanMapping(const IpcSharedPlanMapping&) = delete;
    IpcSharedPlanMapping& operator=(const IpcSharedPlanMapping&) = delete;

    ~IpcSharedPlanMapping() {
        if (arena_ != nullptr) {
            munmap(arena_, sizeof(IpcSharedPlanArena));
            arena_ = nullptr;
        }

        /*
         * Only rank 0 removes the name. Existing mappings remain valid until
         * every process unmaps them.
         */
        if (local_rank_ == 0 && !shm_name_.empty()) {
            shm_unlink(shm_name_.c_str());
        }
    }

    IpcSharedPlanArena* arena() const {
        return arena_;
    }

private:
    void create_rank0() {
        shm_unlink(shm_name_.c_str());

        const int fd =
            shm_open(
                shm_name_.c_str(),
                O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
                0600);

        if (fd < 0) {
            throw std::runtime_error(
                std::string("IPC plan arena: shm_open create failed: ") +
                std::strerror(errno));
        }

        if (ftruncate(fd, static_cast<off_t>(sizeof(IpcSharedPlanArena))) != 0) {
            const int saved_errno = errno;
            close(fd);
            shm_unlink(shm_name_.c_str());
            throw std::runtime_error(
                std::string("IPC plan arena: ftruncate failed: ") +
                std::strerror(saved_errno));
        }

        void* mapped =
            mmap(
                nullptr,
                sizeof(IpcSharedPlanArena),
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0);

        close(fd);

        if (mapped == MAP_FAILED) {
            shm_unlink(shm_name_.c_str());
            throw std::runtime_error(
                std::string("IPC plan arena: mmap create failed: ") +
                std::strerror(errno));
        }

        arena_ =
            reinterpret_cast<IpcSharedPlanArena*>(mapped);

        std::memset(arena_, 0, sizeof(IpcSharedPlanArena));

        arena_->world_size =
            static_cast<std::uint32_t>(world_size_);
        arena_->version = kIpcSharedPlanVersion;

        __sync_synchronize();

        arena_->magic = kIpcSharedPlanMagic;

        __sync_synchronize();
    }

    void open_nonzero_rank() {
        int fd = -1;

        while (true) {
            fd =
                shm_open(
                    shm_name_.c_str(),
                    O_RDWR | O_CLOEXEC,
                    0);

            if (fd >= 0) {
                break;
            }

            if (errno != ENOENT) {
                throw std::runtime_error(
                    std::string("IPC plan arena: shm_open open failed: ") +
                    std::strerror(errno));
            }

            usleep(1);
        }

        struct stat st {};

        do {
            if (fstat(fd, &st) != 0) {
                const int saved_errno = errno;
                close(fd);
                throw std::runtime_error(
                    std::string("IPC plan arena: fstat failed: ") +
                    std::strerror(saved_errno));
            }

            if (static_cast<std::size_t>(st.st_size) >=
                sizeof(IpcSharedPlanArena)) {
                break;
            }

            usleep(1);
        } while (true);

        void* mapped =
            mmap(
                nullptr,
                sizeof(IpcSharedPlanArena),
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0);

        close(fd);

        if (mapped == MAP_FAILED) {
            throw std::runtime_error(
                std::string("IPC plan arena: mmap open failed: ") +
                std::strerror(errno));
        }

        arena_ =
            reinterpret_cast<IpcSharedPlanArena*>(mapped);
    }

    std::string shm_name_;
    int local_rank_ = -1;
    int world_size_ = 0;
    IpcSharedPlanArena* arena_ = nullptr;
};

template <int MaxTransferTasks, typename Builder>
oo_status_t build_or_read_shared_ipc_plan(
    oo_node_t* node,
    IpcSharedPlanArena* arena,
    CollectivePlanFor collective,
    TransferPlan<MaxTransferTasks>* out_plan,
    Builder&& builder) {
    if (node == nullptr ||
        node->group == nullptr ||
        node->group->broker == nullptr ||
        arena == nullptr ||
        out_plan == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    static_assert(
        sizeof(TransferPlan<MaxTransferTasks>) <= kIpcSharedPlanPayloadBytes,
        "TransferPlan does not fit in IPC shared plan arena");

    oo_group_t* group = node->group;

    const int rank = group->local_rank;
    const int world_size = group->local_world_size;

    if (rank < 0 ||
        rank >= world_size ||
        world_size <= 0 ||
        world_size != group->num_devices ||
        arena->magic != kIpcSharedPlanMagic ||
        arena->version != kIpcSharedPlanVersion ||
        arena->world_size != static_cast<std::uint32_t>(world_size)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    /*
     * Make sure all ranks have reached the same cache miss before rank 0 writes
     * the shared arena.
     */
    group->broker->sync();

    if (rank == 0) {
        TransferPlan<MaxTransferTasks> root_plan{};
        transfer_plan_clear(&root_plan);

        const oo_status_t status =
            builder(&root_plan);

        arena->status = static_cast<int>(status);
        arena->collective = static_cast<int>(collective);
        arena->max_transfer_tasks = MaxTransferTasks;
        arena->payload_bytes =
            static_cast<std::uint64_t>(sizeof(root_plan));

        if (status == OO_SUCCESS) {
            std::memcpy(
                arena->payload,
                &root_plan,
                sizeof(root_plan));
        } else {
            std::memset(
                arena->payload,
                0,
                sizeof(root_plan));
        }

        __sync_synchronize();
        arena->sequence += 1;
        __sync_synchronize();
    }

    /*
     * Publish barrier: after this, all ranks may read rank 0's plan bytes.
     */
    group->broker->sync();

    oo_status_t result =
        static_cast<oo_status_t>(arena->status);

    if (result == OO_SUCCESS) {
        if (arena->collective != static_cast<int>(collective) ||
            arena->max_transfer_tasks != MaxTransferTasks ||
            arena->payload_bytes != sizeof(*out_plan)) {
            result = OO_ERROR_INTERNAL;
        } else {
            std::memcpy(
                out_plan,
                arena->payload,
                sizeof(*out_plan));
        }
    }

    /*
     * Read-complete barrier: prevents a fast rank 0 from overwriting the single
     * shared arena for a later cache miss while another rank is still reading.
     */
    group->broker->sync();

    return result;
}

class IpcSharedPlanTransferPlanDistributionBackend final
    : public TransferPlanDistributionBackend {
public:
    IpcSharedPlanTransferPlanDistributionBackend(
        const char* broker_key,
        int local_rank,
        int world_size)
        : mapping_(broker_key, local_rank, world_size) {}

    oo_status_t get_allreduce_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t op,
        const ooverlap::comm::LaunchConfig& config,
        AllreduceTransferPlan** out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            node->group->broker == nullptr ||
            out_plan == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::AllReduce) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::AllReduce,
                launch,
                count,
                dtype,
                op,
                config);

        return get_or_build_cached_same_process_plan(
            &allreduce_,
            key,
            out_plan,
            [this, node, group, &launch, count, &config](AllreduceTransferPlan* plan) {
                return build_or_read_shared_ipc_plan(
                    node,
                    mapping_.arena(),
                    CollectivePlanFor::AllReduce,
                    plan,
                    [group, &launch, count, &config](AllreduceTransferPlan* root_plan) {
                        if (group == nullptr ||
                            group->num_devices != launch.world_size) {
                            return OO_ERROR_INVALID_ARGUMENT;
                        }

                        const TransferPlanBuildInput input =
                            make_build_input(
                                group,
                                launch,
                                count,
                                config,
                                CollectivePlanFor::AllReduce);

                        return build_allreduce_transfer_plan(root_plan, input)
                            ? OO_SUCCESS
                            : OO_ERROR_UNSUPPORTED;
                    });
            });
    }

    oo_status_t get_reduce_scatter_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t op,
        const ooverlap::comm::LaunchConfig& config,
        ReduceScatterTransferPlan** out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            node->group->broker == nullptr ||
            out_plan == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::ReduceScatter) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::ReduceScatter,
                launch,
                count,
                dtype,
                op,
                config);

        return get_or_build_cached_same_process_plan(
            &reduce_scatter_,
            key,
            out_plan,
            [this, node, group, &launch, count, &config](ReduceScatterTransferPlan* plan) {
                return build_or_read_shared_ipc_plan(
                    node,
                    mapping_.arena(),
                    CollectivePlanFor::ReduceScatter,
                    plan,
                    [group, &launch, count, &config](ReduceScatterTransferPlan* root_plan) {
                        if (group == nullptr ||
                            group->num_devices != launch.world_size) {
                            return OO_ERROR_INVALID_ARGUMENT;
                        }

                        const TransferPlanBuildInput input =
                            make_build_input(
                                group,
                                launch,
                                count,
                                config,
                                CollectivePlanFor::ReduceScatter);

                        return build_reduce_scatter_transfer_plan(root_plan, input)
                            ? OO_SUCCESS
                            : OO_ERROR_UNSUPPORTED;
                    });
            });
    }

    oo_status_t get_all_gather_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        const ooverlap::comm::LaunchConfig& config,
        AllGatherTransferPlan** out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            node->group->broker == nullptr ||
            out_plan == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::AllGather) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::AllGather,
                launch,
                count,
                dtype,
                OO_REDUCE_ADD,
                config);

        return get_or_build_cached_same_process_plan(
            &all_gather_,
            key,
            out_plan,
            [this, node, group, &launch, count, &config](AllGatherTransferPlan* plan) {
                return build_or_read_shared_ipc_plan(
                    node,
                    mapping_.arena(),
                    CollectivePlanFor::AllGather,
                    plan,
                    [group, &launch, count, &config](AllGatherTransferPlan* root_plan) {
                        if (group == nullptr ||
                            group->num_devices != launch.world_size) {
                            return OO_ERROR_INVALID_ARGUMENT;
                        }

                        const TransferPlanBuildInput input =
                            make_build_input(
                                group,
                                launch,
                                count,
                                config,
                                CollectivePlanFor::AllGather);

                        return build_all_gather_transfer_plan(root_plan, input)
                            ? OO_SUCCESS
                            : OO_ERROR_UNSUPPORTED;
                    });
            });
    }

private:
    IpcSharedPlanMapping mapping_;
    SameProcessPlanCache<kTmaMultiGpuAllReduceMaxTransferTasks> allreduce_{};
    SameProcessPlanCache<kTmaMultiGpuReduceScatterMaxTransferTasks> reduce_scatter_{};
    SameProcessPlanCache<kTmaMultiGpuAllGatherMaxTransferTasks> all_gather_{};
};


} // namespace

std::unique_ptr<TransferPlanDistributionBackend>
make_same_process_transfer_plan_distribution_backend() {
    return std::unique_ptr<TransferPlanDistributionBackend>(
        new SameProcessTransferPlanDistributionBackend());
}

std::unique_ptr<TransferPlanDistributionBackend>
make_ipc_shared_plan_transfer_plan_distribution_backend(
    const char* broker_key,
    int local_rank,
    int world_size) {
    return std::unique_ptr<TransferPlanDistributionBackend>(
        new IpcSharedPlanTransferPlanDistributionBackend(
            broker_key,
            local_rank,
            world_size));
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
