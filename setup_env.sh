#!/usr/bin/env bash
set -Eeuo pipefail

# -------- user-tunable defaults --------
MODULE_SETUP="${MODULE_SETUP:-$HOME/modules-conf/setup_env_gcore13-2.sh}"
#INSTALL_TORCH="${INSTALL_TORCH:-$HOME/scripts/install_torch_14.sh}"
INSTALL_TORCH="${INSTALL_TORCH:-$HOME/scripts/install_torch.sh}"
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

log "Checking torch"
"$PYTHON_BIN" -c 'import torch; print("torch:", torch.__version__)'

TORCH_PREFIX_PATH="$("$PYTHON_BIN" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
log "Torch CMake prefix path: $TORCH_PREFIX_PATH"

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
  -DPython3_FIND_STRATEGY=LOCATION \
  -DPython3_FIND_IMPLEMENTATIONS=CPython \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# -------- build --------
log "Building with $JOBS jobs"
cmake --build . -j "$JOBS"

log "Done"
log "Venv used: $VENV_DIR"
