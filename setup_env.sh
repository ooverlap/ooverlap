#!/usr/bin/env bash
set -Eeuo pipefail

# Run with:
#   bash ./setup_env.sh
#   bash ./setup_env.sh /path/to/venv
#
# Optional machine-specific modules and paths are read from the gitignored
# modules_folders.txt file. Copy modules_folders.example.txt to customize it.
# Then activate later with the env file printed at the end.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
VLLM_VER="${VLLM_VER:-0.25.0}"
VENV_NAME="${VENV_NAME:-torch_venv}"
JOBS="${JOBS:-12}"
CUDA_ARCH="${CUDA_ARCH:-90a}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
SKIP_MODULES="${SKIP_MODULES:-0}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
MODULES_LOADED=0
VENV_PATH_ARG=""
LOCAL_CONFIG="${OOVERLAP_LOCAL_CONFIG:-$ROOT_DIR/modules_folders.txt}"
LOCAL_MODULES=()
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-}"
CUDADEVRT_HINT="${OOVERLAP_CUDADEVRT_HINT:-}"

log() { printf '[info] %s\n' "$*"; }
die() { printf '[error] %s\n' "$*" >&2; exit 1; }

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

parse_args() {
  if (( $# > 1 )); then
    die "Usage: bash ./setup_env.sh [venv-path]"
  fi

  VENV_PATH_ARG="${1:-}"
}

load_environment_modules() {
  if [[ "$SKIP_MODULES" == "1" || ${#LOCAL_MODULES[@]} -eq 0 ]]; then
    log "No local modules requested; checking the current system toolchain"
    return
  fi

  command -v module >/dev/null 2>&1 || \
    die "Local config requests modules, but the environment-modules command is unavailable"

  module purge
  module load "${LOCAL_MODULES[@]}"
  module list
  MODULES_LOADED=1
}

check_system_toolchain() {
  [[ "$MODULES_LOADED" == "0" ]] || return 0

  local missing=()
  local tool

  for tool in "$PYTHON_BOOTSTRAP" cmake curl c++ git; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done

  if ! command -v ninja >/dev/null 2>&1 &&
     ! command -v make >/dev/null 2>&1; then
    missing+=("ninja or make")
  fi

  local nvcc_path
  nvcc_path="${CUDACXX:-$(command -v nvcc || true)}"
  [[ -n "$nvcc_path" && -x "$nvcc_path" ]] || missing+=("nvcc")

  if command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 &&
     ! "$PYTHON_BOOTSTRAP" -m venv --help >/dev/null 2>&1; then
    missing+=("$PYTHON_BOOTSTRAP venv module")
  fi

  if (( ${#missing[@]} > 0 )); then
    printf '[error] Missing required system tools:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    printf '[error] Install the missing tools and rerun setup_env.sh.\n' >&2
    exit 1
  fi

  local detected_cuda_home
  detected_cuda_home="${CUDA_HOME:-$(cd "$(dirname "$(readlink -f "$nvcc_path")")/.." && pwd)}"

  local missing_cuda_libs=()
  local lib
  for lib in libnvrtc.so libcublas.so; do
    if ! find -L "$detected_cuda_home" -type f -name "$lib" -print -quit 2>/dev/null | grep -q .; then
      missing_cuda_libs+=("$lib")
    fi
  done

  if (( ${#missing_cuda_libs[@]} > 0 )); then
    printf '[error] CUDA developer libraries are missing from %s:\n' "$detected_cuda_home" >&2
    printf '  - %s\n' "${missing_cuda_libs[@]}" >&2
    printf '[error] Install the matching CUDA developer libraries ' >&2
    printf '(NVRTC and cuBLAS), then rerun setup_env.sh.\n' >&2
    exit 1
  fi

  log "System toolchain checks passed"
}

setup_cuda_toolchain() {
  local nvcc_path
  nvcc_path="${CUDACXX:-$(command -v nvcc || true)}"

  [[ -n "$nvcc_path" && -x "$nvcc_path" ]] || \
    die 'nvcc was not found; set CUDACXX or CUDA_HOME'

  CUDACXX="$(readlink -f "$nvcc_path")"
  CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$CUDACXX")/.." && pwd)}"

  export CUDACXX CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  log "CUDA: $CUDA_HOME"
}

setup_local_environment() {
  if [[ -n "$VENV_PATH_ARG" ]]; then
    if [[ "$VENV_PATH_ARG" == /* ]]; then
      VENV_DIR="$VENV_PATH_ARG"
    else
      VENV_DIR="$PWD/$VENV_PATH_ARG"
    fi
  elif [[ -n "$LOCAL_TMP_ROOT" ]]; then
    VENV_DIR="$LOCAL_TMP_ROOT/${USER:-user}/$VENV_NAME"
  else
    VENV_DIR="$HOME/$VENV_NAME"
  fi

  if [[ -z "${OOTMP:-}" ]]; then
    if [[ -n "$VENV_PATH_ARG" ]]; then
      OOTMP="$(dirname "$VENV_DIR")/ooverlap-env"
    elif [[ -n "$LOCAL_TMP_ROOT" ]]; then
      OOTMP="$(dirname "$VENV_DIR")"
    else
      OOTMP="$HOME/ooverlap-env"
    fi
  fi

  local dirs=(
    tmp uv-cache pip-cache torch-extensions hf-cache
    xdg-cache xdg-config vllm-cache vllm-config vllm-deps nccl-link
  )
  local dir
  for dir in "${dirs[@]}"; do
    mkdir -p "$OOTMP/$dir"
  done

  export OOTMP
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
}

install_python_packages() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 || \
      die "Python bootstrap executable not found: $PYTHON_BOOTSTRAP"
    "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="$VENV_DIR/bin/python"

  "$PYTHON_BIN" -m pip install --upgrade pip uv
  "$VENV_DIR/bin/uv" pip install --python "$PYTHON_BIN" --upgrade \
    setuptools wheel cmake numpy setuptools_scm setuptools_rust matplotlib \
    "vllm==$VLLM_VER"

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

  "$PYTHON_BIN" - <<'PY'
import torch, vllm
print("vllm:", vllm.__version__)
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("NCCL:", torch.cuda.nccl.version())
PY
}

prepare_repository() {
  [[ -f "$ROOT_DIR/CMakeLists.txt" ]] || \
    die "CMakeLists.txt was not found in $ROOT_DIR"

  if [[ -d "$ROOT_DIR/.git" ]]; then
    log "Initializing all Git submodules"
    git -C "$ROOT_DIR" submodule sync --recursive
    git -C "$ROOT_DIR" submodule update --init --recursive
  fi

  local cutlass_patch_script="$ROOT_DIR/scripts/apply_cutlass_patch.sh"
  [[ -f "$cutlass_patch_script" ]] || \
    die "CUTLASS patch helper was not found: $cutlass_patch_script"
  bash "$cutlass_patch_script"
}

setup_nccl_paths() {
  NCCL_PKG_DIR="$($PYTHON_BIN - <<'PY'
from pathlib import Path
import nvidia.nccl
print(Path(next(iter(nvidia.nccl.__path__))).resolve())
PY
)"

  export NCCL_PKG_DIR
  export VLLM_NCCL_INCLUDE_PATH="$NCCL_PKG_DIR/include"
  export VLLM_NCCL_SO_PATH="$NCCL_PKG_DIR/lib/libnccl.so.2"

  ln -sf "$VLLM_NCCL_SO_PATH" "$OOTMP/nccl-link/libnccl.so"
  export LIBRARY_PATH="$OOTMP/nccl-link:$NCCL_PKG_DIR/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$OOTMP/nccl-link:$NCCL_PKG_DIR/lib:${LD_LIBRARY_PATH:-}"
}

build_ooverlap() {
  [[ -f "$ROOT_DIR/patch_gcc_typename.py" ]] && \
    "$PYTHON_BIN" "$ROOT_DIR/patch_gcc_typename.py"

  [[ "$CLEAN_BUILD" == "1" ]] && rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"

  local torch_prefix
  torch_prefix="$($PYTHON_BIN -c 'import torch; print(torch.utils.cmake_prefix_path)')"

  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$torch_prefix" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_FIND_STRATEGY=LOCATION \
    -DPython3_FIND_IMPLEMENTATIONS=CPython \
    -DOOVERLAP_BUILD_TORCH_COLLECTIVES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

  cmake --build "$BUILD_DIR" -j "$JOBS"
  export VLLM_OOVERLAP_TORCH_EXT="$BUILD_DIR/lib/ooverlap_torch_ext.so"
}

write_runtime_env() {
  ENV_OUT="${ENV_OUT:-$OOTMP/ooverlap_vllm_env.sh}"

  {
    printf '%s\n' '#!/usr/bin/env bash'
    local name
    if [[ "$MODULES_LOADED" == "1" ]]; then
      printf '%s\n' 'module purge'
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
      VLLM_NO_USAGE_STATS VLLM_DO_NOT_TRACK NCCL_PKG_DIR
      VLLM_NCCL_INCLUDE_PATH VLLM_NCCL_SO_PATH LIBRARY_PATH LD_LIBRARY_PATH
      CUDA_HOME CUDACXX VLLM_OOVERLAP_TORCH_EXT
    )
    for name in "${vars[@]}"; do
      printf 'export %s=%q\n' "$name" "${!name}"
    done
  } > "$ENV_OUT"

  chmod +x "$ENV_OUT"
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
