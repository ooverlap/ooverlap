#!/usr/bin/env bash
set -Eeuo pipefail

# ARM64/GH200 environment for ooverlap + vLLM.
#
# Usage:
#   CLEAN_VENV=1 CLEAN_BUILD=1 ./setup_arm_env.sh ../venv
#
# Optional machine-specific modules and paths are read from the gitignored
# modules_folders.txt file. Copy modules_folders.example.txt to customize it.
#
# Later:
#   source ../ooverlap-env/ooverlap_vllm_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
VLLM_VER="${VLLM_VER:-0.25.0}"
VENV_NAME="${VENV_NAME:-torch_venv}"
JOBS="${JOBS:-12}"
CUDA_ARCH="${CUDA_ARCH:-90a}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
CLEAN_VENV="${CLEAN_VENV:-0}"
SKIP_MODULES="${SKIP_MODULES:-0}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
MODULES_LOADED=0
VENV_PATH_ARG=""
LOCAL_CONFIG="${OOVERLAP_LOCAL_CONFIG:-$ROOT_DIR/modules_folders.txt}"
LOCAL_MODULES=()
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-}"
CUDADEVRT_HINT="${OOVERLAP_CUDADEVRT_HINT:-}"

log() { printf '[info] %s\n' "$*"; }
warn() { printf '[warning] %s\n' "$*" >&2; }
die() { printf '[error] %s\n' "$*" >&2; exit 1; }

on_error() {
  local status=$?
  printf '[error] Command failed at line %s:\n' "${BASH_LINENO[0]:-unknown}" >&2
  printf '        %s\n' "${BASH_COMMAND:-unknown}" >&2
  exit "$status"
}
trap on_error ERR

load_local_config() {
  [[ -f "$LOCAL_CONFIG" ]] || {
    log "Local config not found: $LOCAL_CONFIG; using the current environment"
    return
  }

  local line kind value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue

    kind="${line%%[[:space:]]*}"
    value="${line#"$kind"}"
    value="${value#"${value%%[![:space:]]*}"}"
    [[ -n "$value" ]] || die "Missing value for '$kind' in $LOCAL_CONFIG"

    case "$kind" in
      module)
        LOCAL_MODULES+=("$value")
        ;;
      local_tmp_root)
        [[ -n "$LOCAL_TMP_ROOT" ]] || LOCAL_TMP_ROOT="$value"
        ;;
      cudadevrt_hint)
        [[ -n "$CUDADEVRT_HINT" ]] || CUDADEVRT_HINT="$value"
        ;;
      *)
        die "Unknown directive '$kind' in $LOCAL_CONFIG"
        ;;
    esac
  done < "$LOCAL_CONFIG"

  if [[ -n "$CUDADEVRT_HINT" ]]; then
    export OOVERLAP_CUDADEVRT_HINT="$CUDADEVRT_HINT"
  fi
}

initialize_modules() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi

  local init_script
  for init_script in \
    /etc/profile.d/lmod.sh \
    /etc/profile.d/modules.sh \
    /usr/share/lmod/lmod/init/bash \
    /usr/share/Modules/init/bash
  do
    if [[ -r "$init_script" ]]; then
      log "Initializing environment modules from $init_script"
      set +u
      # shellcheck disable=SC1090
      source "$init_script"
      set -u
      break
    fi
  done

  if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
    log "Initializing environment modules from /etc/profile"
    set +u
    # shellcheck disable=SC1091
    source /etc/profile
    set -u
  fi

  command -v module >/dev/null 2>&1 || \
    die "Could not initialize the environment-modules command"
}

parse_args() {
  if (( $# > 1 )); then
    die "Usage: ./setup_arm_env.sh [venv-path]"
  fi
  VENV_PATH_ARG="${1:-}"
}

load_environment_modules() {
  if [[ "$SKIP_MODULES" == "1" || ${#LOCAL_MODULES[@]} -eq 0 ]]; then
    log "No local modules requested; using the current system toolchain"
    return
  fi

  initialize_modules
  module --force purge

  local name
  for name in "${LOCAL_MODULES[@]}"; do
    log "Loading module: $name"
    module load "$name"
  done

  module list
  MODULES_LOADED=1
}

check_system_toolchain() {
  local missing=()
  local tool

  for tool in "$PYTHON_BOOTSTRAP" cmake curl c++ gcc g++ git file readlink make; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done

  if ! command -v ninja >/dev/null 2>&1; then
    warn "System Ninja not found; Python package Ninja will be installed"
  fi

  local nvcc_path
  nvcc_path="${CUDACXX:-$(command -v nvcc || true)}"
  [[ -n "$nvcc_path" && -x "$nvcc_path" ]] || missing+=("nvcc")

  if command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 && \
     ! "$PYTHON_BOOTSTRAP" -m venv --help >/dev/null 2>&1; then
    missing+=("$PYTHON_BOOTSTRAP venv module")
  fi

  if (( ${#missing[@]} > 0 )); then
    printf '[error] Missing required tools:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    exit 1
  fi

  local architecture
  architecture="$(uname -m)"
  log "Host architecture: $architecture"
  [[ "$architecture" == "aarch64" || "$architecture" == "arm64" ]] || \
    warn "Expected an ARM64 GH200 node"

  local python_path python_file
  python_path="$(command -v "$PYTHON_BOOTSTRAP")"
  python_file="$(file -L "$python_path")"
  log "Bootstrap Python: $python_path"
  "$python_path" --version
  log "Python binary: $python_file"

  if [[ "$architecture" == "aarch64" || "$architecture" == "arm64" ]]; then
    if grep -qiE 'x86-64|x86_64' <<<"$python_file"; then
      die "Selected Python is an x86-64 binary: $python_path"
    fi
  fi
}

setup_cuda_toolchain() {
  local nvcc_path
  nvcc_path="${CUDACXX:-$(command -v nvcc || true)}"

  [[ -n "$nvcc_path" && -x "$nvcc_path" ]] || \
    die 'nvcc was not found; set CUDACXX or load the CUDA build environment'

  CUDACXX="$(readlink -f "$nvcc_path")"
  CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$CUDACXX")/.." && pwd)}"

  export CUDACXX CUDA_HOME
  export CMAKE_CUDA_COMPILER="$CUDACXX"
  export PATH="$CUDA_HOME/bin:$PATH"

  if [[ -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  elif [[ -d "$CUDA_HOME/lib" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  fi

  log "CUDA: $CUDA_HOME"
  log "NVCC: $CUDACXX"
  "$CUDACXX" --version | tail -n 1
}

setup_local_environment() {
  if [[ -n "$VENV_PATH_ARG" ]]; then
    if [[ "$VENV_PATH_ARG" == /* ]]; then
      VENV_DIR="$VENV_PATH_ARG"
    else
      VENV_DIR="$(realpath -m "$PWD/$VENV_PATH_ARG")"
    fi
  elif [[ -n "$LOCAL_TMP_ROOT" ]]; then
    VENV_DIR="$LOCAL_TMP_ROOT/${USER:-user}/$VENV_NAME"
  else
    VENV_DIR="$HOME/$VENV_NAME"
  fi

  if [[ -z "${OOTMP:-}" ]]; then
    OOTMP="$(dirname "$VENV_DIR")/ooverlap-env"
  fi

  local dirs=(
    tmp uv-cache pip-cache torch-extensions hf-cache
    xdg-cache xdg-config vllm-cache vllm-config vllm-deps nccl-link
  )
  local dir
  for dir in "${dirs[@]}"; do
    mkdir -p "$OOTMP/$dir"
  done

  export VENV_DIR OOTMP
  export TMPDIR="$OOTMP/tmp"
  export UV_CACHE_DIR="$OOTMP/uv-cache"
  export PIP_CACHE_DIR="$OOTMP/pip-cache"
  export TORCH_EXTENSIONS_DIR="$OOTMP/torch-extensions"
  export XDG_CACHE_HOME="$OOTMP/xdg-cache"
  export XDG_CONFIG_HOME="$OOTMP/xdg-config"
  export HF_HOME="$OOTMP/hf-cache"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export TRANSFORMERS_CACHE="$HF_HOME/transformers"
  export VLLM_CACHE_ROOT="$OOTMP/vllm-cache"
  export VLLM_CONFIG_ROOT="$OOTMP/vllm-config"
  export FETCHCONTENT_BASE_DIR="$OOTMP/vllm-deps"
  export VLLM_NO_USAGE_STATS=1
  export VLLM_DO_NOT_TRACK=1

  log "Venv: $VENV_DIR"
  log "Build directory: $BUILD_DIR"
}

install_python_packages() {
  if [[ "$CLEAN_VENV" == "1" && -e "$VENV_DIR" ]]; then
    log "Removing existing venv"
    rm -rf "$VENV_DIR"
  fi

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 || \
      die "Python bootstrap executable not found: $PYTHON_BOOTSTRAP"
    log "Creating Python virtual environment"
    "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="$VENV_DIR/bin/python"
  export PYTHON_BIN

  "$PYTHON_BIN" -m pip install --upgrade pip uv
  "$VENV_DIR/bin/uv" pip install --python "$PYTHON_BIN" --upgrade \
    setuptools wheel cmake ninja numpy setuptools_scm setuptools_rust matplotlib \
    "vllm==$VLLM_VER"

  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import platform
import torch
import vllm

root = Path(torch.__file__).resolve().parent
header = root / "include/c10/cuda/impl/cuda_cmake_macros.h"
torch_config = root / "share/cmake/Torch/TorchConfig.cmake"

print("Architecture:", platform.machine())
print("Python-compatible vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Torch root:", root)
print("Required CUDA header exists:", header.is_file())
print("TorchConfig.cmake exists:", torch_config.is_file())

if torch.version.cuda is None:
    raise SystemExit("The installed PyTorch is CPU-only")
if not header.is_file():
    raise SystemExit("PyTorch is missing cuda_cmake_macros.h")
if not torch_config.is_file():
    raise SystemExit("PyTorch is missing TorchConfig.cmake")
PY

  local vllm_communicators_dir
  vllm_communicators_dir="$($PYTHON_BIN - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent / "distributed" / "device_communicators")
PY
)"

  curl -fsSL \
    https://raw.githubusercontent.com/ooverlap/vllm/refs/heads/oo-force-allreduce-backends/vllm/distributed/device_communicators/cuda_communicator.py \
    -o "$vllm_communicators_dir/cuda_communicator.py"

  curl -fsSL \
    https://raw.githubusercontent.com/ooverlap/vllm/refs/heads/oo-force-allreduce-backends/vllm/distributed/device_communicators/ooverlap_all_reduce.py \
    -o "$vllm_communicators_dir/ooverlap_all_reduce.py"
}

prepare_repository() {
  [[ -f "$ROOT_DIR/CMakeLists.txt" ]] || \
    die "CMakeLists.txt was not found in $ROOT_DIR"

  if [[ -d "$ROOT_DIR/.git" ]]; then
    log "Initializing all Git submodules"
    git -C "$ROOT_DIR" submodule sync --recursive
    git -C "$ROOT_DIR" submodule update --init --recursive
  fi

  local json_header="$ROOT_DIR/src/third-party/nlohmann_json/include/nlohmann/json.hpp"
  [[ -f "$json_header" ]] || \
    die "nlohmann/json.hpp is missing after submodule initialization: $json_header"

  # NCCL uses GNU Make. Do not let an outer Ninja generator invoke NCCL with Ninja.
  local nccl_cmake="$ROOT_DIR/cmake/third_party/nccl.cmake"
  if [[ -f "$nccl_cmake" ]] && grep -q '\${CMAKE_MAKE_PROGRAM}' "$nccl_cmake"; then
    log "Patching bundled NCCL build to use GNU Make"
    sed -i 's|${CMAKE_MAKE_PROGRAM}|/usr/bin/make|' "$nccl_cmake"
  fi
}

setup_nccl_paths() {
  if ! "$PYTHON_BIN" -c 'import nvidia.nccl' >/dev/null 2>&1; then
    warn "Python package nvidia.nccl is unavailable; relying on project/system NCCL"
    return 0
  fi

  NCCL_PKG_DIR="$($PYTHON_BIN - <<'PY'
from pathlib import Path
import nvidia.nccl
print(Path(next(iter(nvidia.nccl.__path__))).resolve())
PY
)"

  export NCCL_PKG_DIR
  export VLLM_NCCL_INCLUDE_PATH="$NCCL_PKG_DIR/include"
  export VLLM_NCCL_SO_PATH="$NCCL_PKG_DIR/lib/libnccl.so.2"

  if [[ -f "$VLLM_NCCL_SO_PATH" ]]; then
    ln -sf "$VLLM_NCCL_SO_PATH" "$OOTMP/nccl-link/libnccl.so"
    export LIBRARY_PATH="$OOTMP/nccl-link:$NCCL_PKG_DIR/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$OOTMP/nccl-link:$NCCL_PKG_DIR/lib:${LD_LIBRARY_PATH:-}"
  fi
}

build_ooverlap() {
  [[ -f "$ROOT_DIR/patch_gcc_typename.py" ]] && \
    "$PYTHON_BIN" "$ROOT_DIR/patch_gcc_typename.py"

  if [[ "$CLEAN_BUILD" == "1" ]]; then
    log "Removing previous build directory"
    rm -rf "$BUILD_DIR"
  fi
  mkdir -p "$BUILD_DIR"

  local cmake_bin="$VENV_DIR/bin/cmake"
  local ninja_bin="$VENV_DIR/bin/ninja"
  local torch_prefix torch_dir

  torch_prefix="$($PYTHON_BIN -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  torch_dir="$($PYTHON_BIN -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "share/cmake/Torch")')"

  log "Torch CMake prefix: $torch_prefix"
  log "Torch_DIR: $torch_dir"

  "$cmake_bin" -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$ninja_bin" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_PREFIX_PATH="$torch_prefix" \
    -DTorch_DIR="$torch_dir" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_FIND_STRATEGY=LOCATION \
    -DPython3_FIND_IMPLEMENTATIONS=CPython \
    -DOOVERLAP_BUILD_TORCH_COLLECTIVES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

  "$cmake_bin" --build "$BUILD_DIR" --parallel "$JOBS"

  local ext_path
  ext_path="$(find "$BUILD_DIR" -type f -name 'ooverlap_torch_ext*.so' -print -quit)"
  [[ -n "$ext_path" ]] || die "ooverlap_torch_ext shared library was not produced"
  export VLLM_OOVERLAP_TORCH_EXT="$ext_path"

  log "Torch extension: $VLLM_OOVERLAP_TORCH_EXT"
}

write_runtime_env() {
  ENV_OUT="${ENV_OUT:-$OOTMP/ooverlap_vllm_env.sh}"

  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set +u'

    local name
    if [[ "$MODULES_LOADED" == "1" ]]; then
      printf '%s\n' 'if ! command -v module >/dev/null 2>&1; then'
      printf '%s\n' '  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do'
      printf '%s\n' '    [[ -r "$f" ]] && source "$f" && break'
      printf '%s\n' '  done'
      printf '%s\n' 'fi'
      printf '%s\n' 'module --force purge'
      for name in "${LOCAL_MODULES[@]}"; do
        printf 'module load %q\n' "$name"
      done
    fi

    printf 'source %q\n' "$VENV_DIR/bin/activate"
    printf 'export PATH=%q:$PATH\n' "$CUDA_HOME/bin"

    local vars=(
      OOTMP TMPDIR UV_CACHE_DIR PIP_CACHE_DIR TORCH_EXTENSIONS_DIR
      XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE
      VLLM_CACHE_ROOT VLLM_CONFIG_ROOT FETCHCONTENT_BASE_DIR
      VLLM_NO_USAGE_STATS VLLM_DO_NOT_TRACK
      CUDA_HOME CUDACXX VLLM_OOVERLAP_TORCH_EXT
    )

    if [[ -n "${NCCL_PKG_DIR:-}" ]]; then
      vars+=(NCCL_PKG_DIR VLLM_NCCL_INCLUDE_PATH VLLM_NCCL_SO_PATH)
    fi
    vars+=(LIBRARY_PATH LD_LIBRARY_PATH)

    for name in "${vars[@]}"; do
      printf 'export %s=%q\n' "$name" "${!name:-}"
    done

    printf '%s\n' 'set -u'
  } > "$ENV_OUT"

  chmod 700 "$ENV_OUT"
  log "Runtime env: $ENV_OUT"
}

main() {
  parse_args "$@"
  load_local_config
  load_environment_modules
  check_system_toolchain
  setup_cuda_toolchain
  setup_local_environment
  install_python_packages
  prepare_repository
  setup_nccl_paths
  build_ooverlap
  write_runtime_env
  log "Done. Later run: source $ENV_OUT"
}

main "$@"
