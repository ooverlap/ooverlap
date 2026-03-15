#pragma once

#include <ATen/ATen.h>

namespace ooverlap {

void rmsnorm(at::Tensor X, at::Tensor O, at::Tensor RW);

void reorder_rmsnorm(
    at::Tensor X,
    at::Tensor O,
    at::Tensor RW,
    int64_t BM,
    int64_t BN,
    int64_t rldn,
    at::Tensor RA);

} // namespace ooverlap
