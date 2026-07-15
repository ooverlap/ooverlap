#!/usr/bin/env bash
set -Eeuo pipefail

# Run with:
#   bash ./setup_env.sh
# Then activate later with the env file printed at the end.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
VLLM_VER="${VLLM_VER:-0.25.0}"
VENV_NAME="${VENV_NAME:-torch_venv}"
JOBS="${JOBS:-12}"
CUDA_ARCH="${CUDA_ARCH:-90a}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"

VERA_MODULES=(
  foss/2025b
  CUDA/13.3.0
  Ninja/1.13.0-GCCcore-14.3.0
  Python/3.13.5-GCCcore-14.3.0
  protobuf/31.1-GCCcore-14.3.0
  numactl/2.0.19-GCCcore-14.3.0
  FFmpeg/7.1.2-GCCcore-14.3.0
  Rust/1.88.0-GCCcore-14.3.0
  nodejs/22.17.1-GCCcore-14.3.0
)

log() { printf '[info] %s\n' "$*"; }
die() { printf '[error] %s\n' "$*" >&2; exit 1; }

load_vera_modules() {
  module purge
  module load "${VERA_MODULES[@]}"
  module list
}

pick_vera_tmp() {
  local path

  while read -r path; do
    [[ -d "$path" && -w "$path" ]] && { printf '%s\n' "$path"; return; }
  done < <(df -P | awk '$6 ~ /^\/local\/tmp\./ {print $6}')

  while read -r path; do
    [[ -d "$path" && -w "$path" ]] && { printf '%s\n' "$path"; return; }
  done < <(find /local -maxdepth 1 -type d -name 'tmp.*' 2>/dev/null | sort)

  die 'No writable /local/tmp.* directory found'
}

setup_local_environment() {
  LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-$(pick_vera_tmp)}"
  VENV_DIR="${VENV_DIR:-$LOCAL_TMP_ROOT/$USER/$VENV_NAME}"
  OOTMP="${OOTMP:-$(dirname "$VENV_DIR")}"

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
    python3.13 -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="$VENV_DIR/bin/python"

  "$PYTHON_BIN" -m pip install --upgrade pip uv
  "$VENV_DIR/bin/uv" pip install --python "$PYTHON_BIN" --upgrade \
    setuptools wheel cmake numpy setuptools_scm setuptools_rust matplotlib \
    "vllm==$VLLM_VER"

  "$PYTHON_BIN" - <<'PY'
import torch, vllm
print("vllm:", vllm.__version__)
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("NCCL:", torch.cuda.nccl.version())
PY
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
    printf '%s\n' '#!/usr/bin/env bash' 'module purge'
    local name
    for name in "${VERA_MODULES[@]}"; do
      printf 'module load %q\n' "$name"
    done
    printf 'source %q\n' "$VENV_DIR/bin/activate"

    local vars=(
      OOTMP TMPDIR UV_CACHE_DIR PIP_CACHE_DIR TORCH_EXTENSIONS_DIR
      XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE
      VLLM_CACHE_ROOT VLLM_CONFIG_ROOT FETCHCONTENT_BASE_DIR
      VLLM_NO_USAGE_STATS VLLM_DO_NOT_TRACK NCCL_PKG_DIR
      VLLM_NCCL_INCLUDE_PATH VLLM_NCCL_SO_PATH LIBRARY_PATH LD_LIBRARY_PATH
      VLLM_OOVERLAP_TORCH_EXT
    )
    for name in "${vars[@]}"; do
      printf 'export %s=%q\n' "$name" "${!name}"
    done
  } > "$ENV_OUT"

  chmod +x "$ENV_OUT"
  log "Runtime env: $ENV_OUT"
}

main() {
  load_vera_modules
  setup_local_environment
  install_python_packages
  setup_nccl_paths
  build_ooverlap
  write_runtime_env
  log "Done. Later run: source $ENV_OUT"
}

main "$@"
