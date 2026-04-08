#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#ifndef OOVERLAP_CUCHECK
#define OOVERLAP_CUCHECK(cmd)                                                     \
    do {                                                                          \
        CUresult _err = (cmd);                                                    \
        if (_err != CUDA_SUCCESS) {                                               \
            const char* _name = nullptr;                                          \
            const char* _str  = nullptr;                                          \
            cuGetErrorName(_err, &_name);                                         \
            cuGetErrorString(_err, &_str);                                        \
            std::fprintf(stderr,                                                  \
                         "CUDA Driver API error at %s:%d: %s (%s)\n",             \
                         __FILE__, __LINE__,                                      \
                         _name ? _name : "UNKNOWN",                               \
                         _str  ? _str  : "no description");                       \
            std::abort();                                                         \
        }                                                                         \
    } while (0)
#endif

#ifndef OOVERLAP_CUDACHECK
#define OOVERLAP_CUDACHECK(cmd)                                                   \
    do {                                                                          \
        cudaError_t _err = (cmd);                                                 \
        if (_err != cudaSuccess) {                                                \
            std::fprintf(stderr,                                                  \
                         "CUDA Runtime API error at %s:%d: %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(_err));           \
            std::abort();                                                         \
        }                                                                         \
    } while (0)
#endif
