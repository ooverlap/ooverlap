#pragma once

#include <nccl.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NCCL_CHECK(cmd)                                                                                           \
    do {                                                                                                          \
        ncclResult_t result = cmd;                                                                                \
        if (result != ncclSuccess) {                                                                              \
            std::printf("[ERROR] NCCL error %s:%d '%s' : %s\n", __FILE__, __LINE__, #cmd, ncclGetErrorString(result)); \
            std::exit(-1);                                                                                        \
        }                                                                                                         \
    } while (0)

std::vector<int64_t> generate_nccl_id();
