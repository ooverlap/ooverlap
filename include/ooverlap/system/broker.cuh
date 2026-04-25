#pragma once

/*
 * ooverlap/system/broker.cuh
 *
 * Declaration-only node-local process broker.
 *
 * Important:
 *   Broker methods are intentionally NOT defined inline here.
 *   The implementation lives in src/system/broker.cpp to avoid ODR/linker
 *   layout mismatches across CUDA/C++ translation units.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#error "ooverlap::system::Broker is not supported on Windows"
#endif

#include <cstddef>
#include <cstdint>

#include <string>
#include <type_traits>

namespace ooverlap {
namespace system {

namespace broker_detail {

static constexpr int MAX_LOCAL_WORLD_SIZE = 128;
static constexpr int VAULT_SIZE_PER_RANK = 256;

/*
 * Opaque here. The full definition is in src/system/broker.cpp.
 */
struct Vault;

} // namespace broker_detail

class Broker {
public:
    Broker(int local_rank, int local_world_size, const char* key);

    Broker(const Broker&) = delete;
    Broker& operator=(const Broker&) = delete;

    Broker(Broker&&) = delete;
    Broker& operator=(Broker&&) = delete;

    ~Broker();

    int local_rank() const;
    int local_world_size() const;

    void sync(int num_ranks = -1);

    /*
     * All-gather a fixed-size blob from each local rank.
     *
     * dst must have local_world_size * size bytes.
     * size must be <= broker_detail::VAULT_SIZE_PER_RANK.
     */
    void exchange_data(void* dst, const void* src, size_t size);

    template <typename T>
    void exchange_pod(T* dst, const T& src) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "Broker::exchange_pod requires trivially copyable T");
        exchange_data(dst, &src, sizeof(T));
    }

    /*
     * Exchange one FD from each rank.
     *
     * src_fd is consumed/closed by this function.
     * dst_fds must have local_world_size entries.
     * dst_fds[local_rank] is set to -1.
     */
    void exchange_fds(int* dst_fds, int src_fd);

    /*
     * Broadcast one FD from src_rank to all other local ranks.
     *
     * src_fd is consumed/closed by this function on src_rank.
     */
    void broadcast_fd(int* dst_fd, int src_fd, int src_rank);

    void destroy();

private:
    void check_alive(const char* where) const;

    int local_rank_;
    int local_world_size_;

    std::string shm_key_;
    std::string socket_prefix_;

    void* shm_raw_;
    broker_detail::Vault* shm_;
    int sock_;
};

} // namespace system
} // namespace ooverlap
