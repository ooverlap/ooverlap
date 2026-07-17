#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal Arrhenius GH200 build environment for ooverlap.
#
# This script:
#   1. Loads the Arrhenius GCC/CUDA build environment.
#   2. Creates a small Python venv for CMake, Ninja, NumPy and Matplotlib.
#   3. Builds only the native ooverlap core library.
#
# It intentionally does NOT install or build:
#   - vLLM
#   - PyTorch integrations
#   - Python extension
#   - GEMM/CUTLASS helpers
#   - tests/benchmarks
#   - legacy overlap code
#   - RMSNorm helpers
#
# Usage:
#   CLEAN_VENV=1 ./setup_env.sh ../venv
#
# Later:
#   source ../ooverlap-env/ooverlap_core_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
VENV_PATH_ARG="${1:-}"

JOBS="${JOBS:-12}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
CLEAN_VENV="${CLEAN_VENV:-0}"
SKIP_MODULES="${SKIP_MODULES:-0}"

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-/usr/bin/python3}"

MODULES_LOADED=0
VENV_DIR=""
PYTHON_BIN=""
CUDA_HOME=""
CUDACXX=""
GCC_RUNTIME_DIR=""
OOTMP="${OOTMP:-}"
ENV_OUT="${ENV_OUT:-}"

ARRHENIUS_MODULES=(
  GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
  GPU/buildenv-gcccuda/2026.03-cu13.0
)

log() {
  printf '[info] %s\n' "$*"
}

warn() {
  printf '[warning] %s\n' "$*" >&2
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

on_error() {
  local status=$?

  printf '[error] Command failed at line %s:\n' \
    "${BASH_LINENO[0]:-unknown}" >&2
  printf '        %s\n' "${BASH_COMMAND:-unknown}" >&2

  exit "$status"
}

trap on_error ERR

prepend_path() {
  local variable_name="$1"
  local directory="$2"
  local current_value="${!variable_name:-}"

  [[ -n "$directory" && -d "$directory" ]] || return 0

  case ":$current_value:" in
    *":$directory:"*)
      return 0
      ;;
  esac

  if [[ -n "$current_value" ]]; then
    printf -v "$variable_name" '%s:%s' "$directory" "$current_value"
  else
    printf -v "$variable_name" '%s' "$directory"
  fi

  export "$variable_name"
}

parse_args() {
  if (( $# > 1 )); then
    die "Usage: ./setup_env.sh [venv-path]"
  fi
}

load_modules() {
  if [[ "$SKIP_MODULES" == "1" ]]; then
    log "Skipping environment modules"
    return
  fi

  if ! command -v module >/dev/null 2>&1; then
    die "The module command is unavailable"
  fi

  module --force purge

  local module_name
  for module_name in "${ARRHENIUS_MODULES[@]}"; do
    log "Loading module: $module_name"
    module load "$module_name"
  done

  MODULES_LOADED=1
  module list
}

validate_host() {
  local architecture
  architecture="$(uname -m)"

  log "Host architecture: $architecture"

  case "$architecture" in
    aarch64|arm64)
      ;;
    *)
      warn "Expected a GH200 ARM64 node, but detected: $architecture"
      ;;
  esac

  [[ -x "$PYTHON_BOOTSTRAP" ]] ||
    die "Python is not executable: $PYTHON_BOOTSTRAP"

  log "Bootstrap Python: $PYTHON_BOOTSTRAP"
  "$PYTHON_BOOTSTRAP" --version

  local python_file
  python_file="$(file -L "$PYTHON_BOOTSTRAP")"
  log "Python binary: $python_file"

  if [[ "$architecture" == "aarch64" || "$architecture" == "arm64" ]]; then
    if grep -qiE 'x86-64|x86_64' <<<"$python_file"; then
      die "Selected Python is an x86-64 binary"
    fi
  fi

  "$PYTHON_BOOTSTRAP" -m venv --help >/dev/null 2>&1 ||
    die "Python venv support is unavailable"
}

check_build_tools() {
  local missing=()
  local command_name

  for command_name in \
    gcc \
    g++ \
    nvcc \
    git \
    file \
    readlink
  do
    command -v "$command_name" >/dev/null 2>&1 ||
      missing+=("$command_name")
  done

  if (( ${#missing[@]} > 0 )); then
    printf '[error] Missing required commands:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    exit 1
  fi

  log "GCC: $(command -v gcc)"
  gcc --version | head -n 1

  log "G++: $(command -v g++)"
  g++ --version | head -n 1

  log "NVCC: $(command -v nvcc)"
  nvcc --version | tail -n 1
}

setup_gcc_runtime() {
  local libstdcpp

  libstdcpp="$(g++ -print-file-name=libstdc++.so.6)"

  if [[ -z "$libstdcpp" || "$libstdcpp" == "libstdc++.so.6" ]]; then
    die "g++ could not locate libstdc++.so.6"
  fi

  libstdcpp="$(readlink -f "$libstdcpp")"

  [[ -f "$libstdcpp" ]] ||
    die "libstdc++ does not exist: $libstdcpp"

  GCC_RUNTIME_DIR="$(dirname "$libstdcpp")"
  export GCC_RUNTIME_DIR

  prepend_path LD_LIBRARY_PATH "$GCC_RUNTIME_DIR"
  prepend_path LIBRARY_PATH "$GCC_RUNTIME_DIR"

  log "GCC runtime: $libstdcpp"
}

setup_cuda() {
  local nvcc_path
  nvcc_path="$(command -v nvcc)"

  [[ -x "$nvcc_path" ]] ||
    die "nvcc was not found"

  CUDACXX="$(readlink -f "$nvcc_path")"
  CUDA_HOME="$(cd "$(dirname "$CUDACXX")/.." && pwd)"

  export CUDACXX CUDA_HOME
  export CMAKE_CUDA_COMPILER="$CUDACXX"

  prepend_path PATH "$CUDA_HOME/bin"

  if [[ -d "$CUDA_HOME/lib64" ]]; then
    prepend_path LD_LIBRARY_PATH "$CUDA_HOME/lib64"
    prepend_path LIBRARY_PATH "$CUDA_HOME/lib64"
  fi

  if [[ -d "$CUDA_HOME/lib" ]]; then
    prepend_path LD_LIBRARY_PATH "$CUDA_HOME/lib"
    prepend_path LIBRARY_PATH "$CUDA_HOME/lib"
  fi

  # Make sure the GCC runtime remains preferred over /lib64.
  prepend_path LD_LIBRARY_PATH "$GCC_RUNTIME_DIR"
  prepend_path LIBRARY_PATH "$GCC_RUNTIME_DIR"

  log "CUDA home: $CUDA_HOME"
  log "CUDA compiler: $CUDACXX"
}

setup_directories() {
  if [[ -n "$VENV_PATH_ARG" ]]; then
    if [[ "$VENV_PATH_ARG" == /* ]]; then
      VENV_DIR="$VENV_PATH_ARG"
    else
      VENV_DIR="$(realpath -m "$PWD/$VENV_PATH_ARG")"
    fi
  else
    VENV_DIR="$HOME/venv-ooverlap-core"
  fi

  if [[ -z "$OOTMP" ]]; then
    OOTMP="$(dirname "$VENV_DIR")/ooverlap-env"
  fi

  mkdir -p \
    "$OOTMP/tmp" \
    "$OOTMP/pip-cache" \
    "$OOTMP/xdg-cache" \
    "$OOTMP/xdg-config"

  export VENV_DIR OOTMP
  export TMPDIR="$OOTMP/tmp"
  export PIP_CACHE_DIR="$OOTMP/pip-cache"
  export XDG_CACHE_HOME="$OOTMP/xdg-cache"
  export XDG_CONFIG_HOME="$OOTMP/xdg-config"

  log "Venv: $VENV_DIR"
  log "Build directory: $BUILD_DIR"
}

create_venv() {
  if [[ "$CLEAN_VENV" == "1" && -e "$VENV_DIR" ]]; then
    log "Removing existing venv"
    rm -rf "$VENV_DIR"
  fi

  if [[ -e "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
    die "$VENV_DIR exists but is not a valid virtual environment"
  fi

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating Python virtual environment"
    "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  PYTHON_BIN="$VENV_DIR/bin/python"
  export PYTHON_BIN

  "$PYTHON_BIN" --version
}

install_basic_python_tools() {
  log "Installing basic Python build and analysis tools"

  "$PYTHON_BIN" -m pip install --upgrade \
    pip \
    setuptools \
    wheel

  "$PYTHON_BIN" -m pip install --upgrade \
    cmake \
    ninja \
    numpy \
    matplotlib torch

  log "CMake: $VENV_DIR/bin/cmake"
  "$VENV_DIR/bin/cmake" --version | head -n 1

  log "Ninja: $VENV_DIR/bin/ninja"
  "$VENV_DIR/bin/ninja" --version

  "$PYTHON_BIN" - <<'PY'
import matplotlib
import numpy

print("NumPy:", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
PY
}

prepare_repository() {
  if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
    die "CMakeLists.txt was not found in $ROOT_DIR"
  fi

  # Core-only building does not require CUTLASS, nlohmann_json, or NCCL
  # submodules, but initializing existing submodules is harmless and avoids
  # surprises if the build configuration changes.
  if [[ -d "$ROOT_DIR/.git" ]]; then
    log "Updating repository submodules"
    git -C "$ROOT_DIR" submodule update --init --recursive
  fi
}

configure_core_build() {
  if [[ "$CLEAN_BUILD" == "1" ]]; then
    log "Removing previous build directory"
    rm -rf "$BUILD_DIR"
  fi

  mkdir -p "$BUILD_DIR"

  local cmake_bin
  cmake_bin="$VENV_DIR/bin/cmake"

  log "Configuring core-only ooverlap build"

  local torch_prefix
  torch_prefix="$("$PYTHON_BIN" -c 'import torch; print(torch.utils.cmake_prefix_path)')"

  "$cmake_bin" \
    -S "$ROOT_DIR" \
    -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_PREFIX_PATH="$torch_prefix" \
    -DOOVERLAP_BUILD_TORCH_COLLECTIVES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

  log "CMake configuration completed"
}

build_core() {
  local cmake_bin
  cmake_bin="$VENV_DIR/bin/cmake"

  log "Building ooverlap core with $JOBS parallel jobs"

  "$cmake_bin" \
    --build "$BUILD_DIR" \
    --parallel "$JOBS"

  log "Core build completed"
}

show_build_outputs() {
  log "Built libraries:"

  find "$BUILD_DIR" \
    -type f \
    \( \
      -name 'libooverlap*.a' -o \
      -name 'libooverlap*.so' \
    \) \
    -print |
    sort
}

write_runtime_environment() {
  if [[ -z "$ENV_OUT" ]]; then
    ENV_OUT="$OOTMP/ooverlap_core_env.sh"
  fi

  {
    printf '%s\n' '#!/usr/bin/env bash'

    if [[ "$MODULES_LOADED" == "1" ]]; then
      printf '%s\n' 'module --force purge'

      local module_name
      for module_name in "${ARRHENIUS_MODULES[@]}"; do
        printf 'module load %q\n' "$module_name"
      done
    fi

    printf 'source %q\n' "$VENV_DIR/bin/activate"

    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    printf 'export CUDACXX=%q\n' "$CUDACXX"
    printf 'export GCC_RUNTIME_DIR=%q\n' "$GCC_RUNTIME_DIR"
    printf 'export OOVERLAP_ROOT=%q\n' "$ROOT_DIR"
    printf 'export OOVERLAP_BUILD_DIR=%q\n' "$BUILD_DIR"
    printf 'export OOTMP=%q\n' "$OOTMP"
    printf 'export TMPDIR=%q\n' "$TMPDIR"
    printf 'export PIP_CACHE_DIR=%q\n' "$PIP_CACHE_DIR"
    printf 'export XDG_CACHE_HOME=%q\n' "$XDG_CACHE_HOME"
    printf 'export XDG_CONFIG_HOME=%q\n' "$XDG_CONFIG_HOME"

    printf 'export PATH=%q:$PATH\n' "$CUDA_HOME/bin"
    printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
    printf 'export LIBRARY_PATH=%q\n' "$LIBRARY_PATH"
  } >"$ENV_OUT"

  chmod 700 "$ENV_OUT"

  log "Runtime environment: $ENV_OUT"
}

main() {
  parse_args "$@"
  load_modules
  validate_host
  check_build_tools
  setup_gcc_runtime
  setup_cuda
  setup_directories
  create_venv
  install_basic_python_tools
  prepare_repository
  configure_core_build
  build_core
  show_build_outputs
  write_runtime_environment

  log "Done"
  log "Activate later with: source $ENV_OUT"
}

main "$@"
