#!/usr/bin/env bash
set -Eeuo pipefail

# -------- user-tunable defaults --------
MODULE_SETUP="${MODULE_SETUP:-$HOME/modules-conf/setup_env_gcore14-2.sh}"
#INSTALL_TORCH="${INSTALL_TORCH:-$HOME/scripts/install_torch_14.sh}"
INSTALL_TORCH="${INSTALL_TORCH:-$HOME/scripts/install_torch_14.sh}"
BUILD_DIR="${BUILD_DIR:-$HOME/ooverlap/build}"
JOBS="${JOBS:-12}"
CUDA_ARCH="${CUDA_ARCH:-90a}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
VENV_NAME="${VENV_NAME:-torch_venv}"
USER_TAG="${USER_TAG:-$USER}"

log() {
  printf '[info] %s\n' "$*"
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

pick_local_tmp() {
  local candidates=()
  local mp

  # Prefer mounted /local/tmp.* entries from df
  while read -r mp; do
    [[ -n "$mp" ]] && candidates+=("$mp")
  done < <(df -P | awk '$6 ~ /^\/local\/tmp\./ {print $6}')

  # Fallback: existing directories
  if [[ ${#candidates[@]} -eq 0 ]]; then
    while read -r mp; do
      [[ -n "$mp" ]] && candidates+=("$mp")
    done < <(find /local -maxdepth 1 -type d -name 'tmp.*' 2>/dev/null | sort)
  fi

  [[ ${#candidates[@]} -gt 0 ]] || die "No /local/tmp.* directory found"

  for mp in "${candidates[@]}"; do
    if [[ -d "$mp" && -w "$mp" ]]; then
      printf '%s\n' "$mp"
      return 0
    fi
  done

  die "Found /local/tmp.* entries, but none are writable"
}

# -------- checks --------
[[ -f "$MODULE_SETUP" ]] || die "Module setup script not found: $MODULE_SETUP"
[[ -x "$INSTALL_TORCH" ]] || die "Torch installer not executable: $INSTALL_TORCH"

# -------- load modules --------
log "Loading module environment"
# shellcheck disable=SC1090
source "$MODULE_SETUP"

log "Disk usage snapshot"
df -h

# -------- resolve dynamic venv path --------
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-$(pick_local_tmp)}"
VENV_DIR="${VENV_DIR:-$LOCAL_TMP_ROOT/$USER_TAG/$VENV_NAME}"

log "Using local tmp root: $LOCAL_TMP_ROOT"
log "Using venv dir: $VENV_DIR"

mkdir -p "$(dirname "$VENV_DIR")"

# -------- local tmp/cache environment --------
# VENV_DIR is usually /local/tmp.xxxxx/$USER/torch_venv
# so dirname(VENV_DIR) is /local/tmp.xxxxx/$USER
OOTMP="${OOTMP:-$(dirname "$VENV_DIR")}"

mkdir -p \
  "$OOTMP/tmp" \
  "$OOTMP/uv-cache" \
  "$OOTMP/pip-cache" \
  "$OOTMP/torch-extensions" \
  "$OOTMP/hf-cache" \
  "$OOTMP/xdg-cache" \
  "$OOTMP/xdg-config" \
  "$OOTMP/vllm-cache" \
  "$OOTMP/vllm-config" \
  "$OOTMP/nccl-link"

export OOTMP
export TMPDIR="$OOTMP/tmp"
export UV_CACHE_DIR="$OOTMP/uv-cache"
export PIP_CACHE_DIR="$OOTMP/pip-cache"
export TORCH_EXTENSIONS_DIR="$OOTMP/torch-extensions"

export XDG_CACHE_HOME="$OOTMP/xdg-cache"
export XDG_CONFIG_HOME="$OOTMP/xdg-config"

export HF_HOME="$OOTMP/hf-cache"
export HF_HUB_CACHE="$OOTMP/hf-cache/hub"
export TRANSFORMERS_CACHE="$OOTMP/hf-cache/transformers"

export VLLM_CACHE_ROOT="$OOTMP/vllm-cache"
export VLLM_CONFIG_ROOT="$OOTMP/vllm-config"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

# Useful only if building vLLM from source later.
export FETCHCONTENT_BASE_DIR="$OOTMP/vllm-deps"

# -------- install torch venv if needed --------
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  log "Creating torch venv"
  "$INSTALL_TORCH" "$VENV_DIR"
else
  log "Torch venv already exists"
fi

# -------- activate venv --------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PYTHON_BIN="$VENV_DIR/bin/python3.13"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$VENV_DIR/bin/python"
[[ -x "$PYTHON_BIN" ]] || die "No Python found in venv"

# -------- install/check vLLM --------
INSTALL_VLLM="${INSTALL_VLLM:-1}"
VLLM_VER="${VLLM_VER:-0.25.0}"

if [[ "$INSTALL_VLLM" == "1" ]]; then
  if ! "$PYTHON_BIN" -c 'import vllm' >/dev/null 2>&1; then
    log "Installing vLLM==${VLLM_VER}"
    uv pip install "vllm==${VLLM_VER}"
  else
    log "vLLM already installed"
  fi

  "$PYTHON_BIN" - <<'PY'
import vllm
from pathlib import Path
print("vllm:", getattr(vllm, "__version__", "unknown"))
print("vllm path:", Path(vllm.__file__).resolve())
PY
fi

# -------- vLLM/NCCL paths from active venv --------
NCCL_PKG_DIR="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import nvidia.nccl

paths = list(getattr(nvidia.nccl, "__path__", []))
if not paths:
    raise RuntimeError("nvidia.nccl has no __path__; is nvidia-nccl-cu12 installed?")
print(Path(paths[0]).resolve())
PY
)"

export NCCL_PKG_DIR
export VLLM_NCCL_INCLUDE_PATH="$NCCL_PKG_DIR/include"
export VLLM_NCCL_SO_PATH="$NCCL_PKG_DIR/lib/libnccl.so.2"

ln -sf "$NCCL_PKG_DIR/lib/libnccl.so.2" "$OOTMP/nccl-link/libnccl.so"

export LIBRARY_PATH="$OOTMP/nccl-link:$NCCL_PKG_DIR/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$OOTMP/nccl-link:$NCCL_PKG_DIR/lib:${LD_LIBRARY_PATH:-}"

log "NCCL package dir: $NCCL_PKG_DIR"
log "VLLM_NCCL_SO_PATH: $VLLM_NCCL_SO_PATH"

########## CHECL torch

log "Checking torch"
"$PYTHON_BIN" -c 'import torch; print("torch:", torch.__version__)'

TORCH_PREFIX_PATH="$("$PYTHON_BIN" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
log "Torch CMake prefix path: $TORCH_PREFIX_PATH"

#######patch
"$PYTHON_BIN" patch_gcc_typename.py

# -------- prepare build dir --------
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

RESOLVED_BUILD_DIR="$(pwd -P)"
[[ "$RESOLVED_BUILD_DIR" != "/" ]] || die "Refusing to clean /"
[[ "$RESOLVED_BUILD_DIR" == *"/ooverlap/build" ]] || die "Refusing to clean unexpected dir: $RESOLVED_BUILD_DIR"

if [[ "$CLEAN_BUILD" == "1" ]]; then
  log "Cleaning build dir: $RESOLVED_BUILD_DIR"
  shopt -s dotglob nullglob
  files=( *)
  if (( ${#files[@]} > 0 )); then
    rm -rf -- ./*
  fi
  shopt -u dotglob nullglob
fi

# -------- configure --------
log "Running cmake configure"
cmake -S .. -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX_PATH" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DOOVERLAP_BUILD_TORCH_COLLECTIVES=ON \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DPython3_FIND_IMPLEMENTATIONS=CPython \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# -------- build --------
log "Building with $JOBS jobs"
cmake --build . -j "$JOBS"

export VLLM_OOVERLAP_TORCH_EXT="$BUILD_DIR/lib/ooverlap_torch_ext.so"
log "VLLM_OOVERLAP_TORCH_EXT: $VLLM_OOVERLAP_TORCH_EXT"

log "Done"
log "Venv used: $VENV_DIR"

ENV_OUT="${ENV_OUT:-$OOTMP/ooverlap_vllm_env.sh}"
cat > "$ENV_OUT" <<EOF
source "$VENV_DIR/bin/activate"
export OOTMP="$OOTMP"
export TMPDIR="$TMPDIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export XDG_CONFIG_HOME="$XDG_CONFIG_HOME"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export VLLM_CONFIG_ROOT="$VLLM_CONFIG_ROOT"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export NCCL_PKG_DIR="$NCCL_PKG_DIR"
export VLLM_NCCL_INCLUDE_PATH="$VLLM_NCCL_INCLUDE_PATH"
export VLLM_NCCL_SO_PATH="$VLLM_NCCL_SO_PATH"
export LIBRARY_PATH="$LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export VLLM_OOVERLAP_TORCH_EXT="$VLLM_OOVERLAP_TORCH_EXT"
EOF

log "Wrote runtime env file: $ENV_OUT"
log "Use later with: source $ENV_OUT"
