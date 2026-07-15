#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/gpu_p2p_access_check.cu"
BINARY="${P2P_CHECK_BINARY:-$SCRIPT_DIR/gpu_p2p_access_check}"
NVCC_BIN="${NVCC:-nvcc}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"
ELEMENTS="${P2P_ELEMENTS:-1048576}"
ATOMIC_OPS="${P2P_ATOMIC_OPS:-262144}"

"$NVCC_BIN" \
  -O3 \
  -std=c++17 \
  -lineinfo \
  -arch="$CUDA_ARCH" \
  "$SOURCE" \
  -o "$BINARY"

exec "$BINARY" "$ELEMENTS" "$ATOMIC_OPS"
