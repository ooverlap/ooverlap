#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  run_external_p2p_collective_bench.sh [OPTIONS] {2|4} {AR|RS|AG} NUMEL [NUMEL ...]

Run test/test_external_p2p_collective.py for one or more tensor sizes.

Options:
  --nccl-mode MODE       default or local-tma (default: default)
  --nccl-so PATH         Local NCCL shared library. Default:
                         <repo>/build/third_party/nccl/lib/libnccl.so.2
  --backend BACKEND      all, ooverlap, nccl, or nccl_symmetric (default: all)
  --sym-ctas N           NCCL symmetric CTA count in local-tma mode (default: 16)
  --tma-kernel NAME      Override the collective-specific TMA kernel name
  --iters N              Measured iterations (default: $ITERS or 100)
  --warmup N             Warmup iterations (default: $WARMUP or 20)
  --output-dir DIR       Override result directory
  --no-nccl-preflight    Skip the local NCCL version/path verification
  -h, --help             Show this help

Examples:
  # Use the environment/default NCCL.
  ./evalution/run_external_p2p_collective_bench.sh \
    --nccl-mode default 4 RS 1048576 2097152 8388608

  # Preload the patched NCCL built by ooverlap and force symmetric TMA.
  ./evalution/run_external_p2p_collective_bench.sh \
    --nccl-mode local-tma 4 RS 1048576 2097152 8388608

  # Run only the symmetric NCCL backend with a separately built NCCL clone.
  ./evalution/run_external_p2p_collective_bench.sh \
    --nccl-mode local-tma \
    --nccl-so /home/keyvand/nccl/build/lib/libnccl.so.2 \
    --backend nccl_symmetric \
    --sym-ctas 16 \
    4 AR 2097152
USAGE
  exit 2
}

NCCL_MODE="${NCCL_MODE:-default}"
BACKEND="${BACKEND:-all}"
SYM_CTAS="${NCCL_SYM_CTAS:-16}"
ITERS_VALUE="${ITERS:-100}"
WARMUP_VALUE="${WARMUP:-20}"
NCCL_SO="${NCCL_SO:-}"
TMA_KERNEL_OVERRIDE=""
OUT_DIR_OVERRIDE=""
RUN_NCCL_PREFLIGHT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nccl-mode)
      [[ $# -ge 2 ]] || usage
      NCCL_MODE="$2"
      shift 2
      ;;
    --nccl-so)
      [[ $# -ge 2 ]] || usage
      NCCL_SO="$2"
      shift 2
      ;;
    --backend)
      [[ $# -ge 2 ]] || usage
      BACKEND="$2"
      shift 2
      ;;
    --sym-ctas)
      [[ $# -ge 2 ]] || usage
      SYM_CTAS="$2"
      shift 2
      ;;
    --tma-kernel)
      [[ $# -ge 2 ]] || usage
      TMA_KERNEL_OVERRIDE="$2"
      shift 2
      ;;
    --iters)
      [[ $# -ge 2 ]] || usage
      ITERS_VALUE="$2"
      shift 2
      ;;
    --warmup)
      [[ $# -ge 2 ]] || usage
      WARMUP_VALUE="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || usage
      OUT_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --no-nccl-preflight)
      RUN_NCCL_PREFLIGHT=0
      shift
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      usage
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -ge 3 ]] || usage
WORLD_SIZE="$1"
COLLECTIVE_ARG="${2^^}"
shift 2
NUMELS=("$@")

case "$COLLECTIVE_ARG" in
  AR)
    COLLECTIVE="allreduce"
    DEFAULT_TMA_KERNEL="AllReduce_RSxTmaLD_AGxTmaST"
    ;;
  RS)
    COLLECTIVE="reduce_scatter"
    DEFAULT_TMA_KERNEL="ReduceScatter_TmaLD"
    ;;
  AG)
    COLLECTIVE="all_gather"
    DEFAULT_TMA_KERNEL="AllGather_TmaST"
    ;;
  *)
    echo "error: collective must be AR, RS, or AG; got: $COLLECTIVE_ARG" >&2
    usage
    ;;
esac

TMA_KERNEL="${TMA_KERNEL_OVERRIDE:-$DEFAULT_TMA_KERNEL}"

case "$WORLD_SIZE" in
  2)
    DEVICES="0,1"
    DEFAULT_OOVERLAP_CTAS=8
    DEFAULT_REDUCE_TASK_CTAS=8
    ;;
  4)
    DEVICES="0,1,2,3"
    DEFAULT_OOVERLAP_CTAS=9
    DEFAULT_REDUCE_TASK_CTAS=3
    ;;
  *)
    echo "error: world size must be 2 or 4; got: $WORLD_SIZE" >&2
    usage
    ;;
esac

case "$NCCL_MODE" in
  default|local-tma) ;;
  *)
    echo "error: --nccl-mode must be default or local-tma; got: $NCCL_MODE" >&2
    usage
    ;;
esac

case "$BACKEND" in
  all|ooverlap|nccl|nccl_symmetric) ;;
  *)
    echo "error: --backend must be all, ooverlap, nccl, or nccl_symmetric; got: $BACKEND" >&2
    usage
    ;;
esac

for value_name in SYM_CTAS ITERS_VALUE; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: $value_name must be a positive integer; got: $value" >&2
    exit 2
  }
done

[[ "$WARMUP_VALUE" =~ ^[0-9]+$ ]] || {
  echo "error: warmup must be a non-negative integer; got: $WARMUP_VALUE" >&2
  exit 2
}

for numel in "${NUMELS[@]}"; do
  [[ "$numel" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: NUMEL must be a positive integer; got: $numel" >&2
    exit 2
  }
  if [[ "$COLLECTIVE" != "allreduce" ]] && (( numel % WORLD_SIZE != 0 )); then
    echo "error: $COLLECTIVE requires NUMEL divisible by world size $WORLD_SIZE; got: $numel" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../test/test_external_p2p_collective.py" ]]; then
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/test/test_external_p2p_collective.py" ]]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  echo "error: could not locate the ooverlap repository root from $SCRIPT_DIR" >&2
  exit 1
fi

DRIVER="$REPO_ROOT/test/test_external_p2p_collective.py"
EXTENSION="$REPO_ROOT/build/lib/ooverlap_ext.so"
PYTHON_BIN="${PYTHON_BIN:-python}"
OOVERLAP_CTAS="${OOVERLAP_MAX_CTAS:-$DEFAULT_OOVERLAP_CTAS}"
REDUCE_TASK_CTAS="${OOVERLAP_MAX_CTAS_PER_REDUCE_TASK:-$DEFAULT_REDUCE_TASK_CTAS}"

TUNING_POLICY="${OOVERLAP_TUNING_POLICY:-$REPO_ROOT/results/policies/tp${WORLD_SIZE}_policy.json}"
if [[ ! -f "$TUNING_POLICY" && -f "$REPO_ROOT/results/policies/tp4_policy.json" ]]; then
  TUNING_POLICY="$REPO_ROOT/results/policies/tp4_policy.json"
fi

if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
  OUT_DIR="$OUT_DIR_OVERRIDE"
else
  OUT_DIR="$REPO_ROOT/results/evalution/external_p2p/direct_bench/tp${WORLD_SIZE}/${NCCL_MODE}/${COLLECTIVE}"
fi
SUMMARY="$OUT_DIR/summary.tsv"

[[ -f "$DRIVER" ]] || {
  echo "error: benchmark driver not found: $DRIVER" >&2
  exit 1
}
[[ -f "$EXTENSION" ]] || {
  echo "error: Python extension not found: $EXTENSION" >&2
  echo "       build ooverlap first with: cmake --build build -j" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "error: Python executable not found: $PYTHON_BIN" >&2
  exit 1
}

ENV_UNSETS=(
  -u NCCL_DEBUG
  -u NCCL_DEBUG_SUBSYS
)
ENV_ASSIGNMENTS=(
  "CUDA_VISIBLE_DEVICES=$DEVICES"
  "OOVERLAP_MAX_CTAS=$OOVERLAP_CTAS"
  "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK=$REDUCE_TASK_CTAS"
)

if [[ -f "$TUNING_POLICY" ]]; then
  ENV_ASSIGNMENTS+=("OOVERLAP_TUNING_POLICY=$TUNING_POLICY")
  TUNING_POLICY_LABEL="$TUNING_POLICY"
else
  ENV_UNSETS+=(-u OOVERLAP_TUNING_POLICY)
  TUNING_POLICY_LABEL="not found (runtime fallback)"
fi

if [[ "$BACKEND" == "all" ]]; then
  ENV_UNSETS+=(-u OOVERLAP_BENCH_ONLY)
else
  ENV_ASSIGNMENTS+=("OOVERLAP_BENCH_ONLY=$BACKEND")
fi

NCCL_SO_REAL=""
NCCL_RUNTIME_LABEL="environment/default"

if [[ "$NCCL_MODE" == "default" ]]; then
  ENV_UNSETS+=(
    -u NCCL_SYM_TMA_ENABLE
    -u NCCL_SYM_KERNEL
    -u NCCL_SYM_CTAS
  )
  if [[ "${LD_PRELOAD:-}" == *libnccl* ]]; then
    echo "[evalution] warning: default mode inherited an LD_PRELOAD containing libnccl:" >&2
    echo "[evalution]          $LD_PRELOAD" >&2
  fi
else
  if [[ -z "$NCCL_SO" ]]; then
    NCCL_SO="$REPO_ROOT/build/third_party/nccl/lib/libnccl.so.2"
  fi

  if [[ ! -e "$NCCL_SO" ]]; then
    NCCL_LIB_SEARCH_DIR="$(dirname -- "$NCCL_SO")"
    candidate="$(find -L "$NCCL_LIB_SEARCH_DIR" -maxdepth 1 -type f -name 'libnccl.so.2.*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "$candidate" ]]; then
      NCCL_SO="$candidate"
    fi
  fi

  [[ -e "$NCCL_SO" ]] || {
    echo "error: local NCCL library not found: $NCCL_SO" >&2
    echo "       expected the bundled build under build/third_party/nccl/lib," >&2
    echo "       or pass --nccl-so /path/to/libnccl.so.2" >&2
    exit 1
  }

  NCCL_SO_REAL="$(readlink -f "$NCCL_SO")"
  NCCL_LIBDIR="$(dirname -- "$NCCL_SO_REAL")"

  GCC_LIBDIR=""
  if command -v g++ >/dev/null 2>&1; then
    GCC_LIBSTDCPP="$(g++ -print-file-name=libstdc++.so.6 2>/dev/null || true)"
    if [[ -n "$GCC_LIBSTDCPP" && "$GCC_LIBSTDCPP" != "libstdc++.so.6" ]]; then
      GCC_LIBSTDCPP="$(readlink -f "$GCC_LIBSTDCPP")"
      GCC_LIBDIR="$(dirname -- "$GCC_LIBSTDCPP")"
    fi
  fi

  LOCAL_LIBRARY_PATH="$NCCL_LIBDIR"
  if [[ -n "$GCC_LIBDIR" ]]; then
    LOCAL_LIBRARY_PATH="$LOCAL_LIBRARY_PATH:$GCC_LIBDIR"
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    LOCAL_LIBRARY_PATH="$LOCAL_LIBRARY_PATH:$LD_LIBRARY_PATH"
  fi

  LOCAL_PRELOAD="$NCCL_SO_REAL"
  if [[ -n "${LD_PRELOAD:-}" ]]; then
    LOCAL_PRELOAD="$LOCAL_PRELOAD:$LD_PRELOAD"
  fi

  ENV_ASSIGNMENTS+=(
    "LD_LIBRARY_PATH=$LOCAL_LIBRARY_PATH"
    "LD_PRELOAD=$LOCAL_PRELOAD"
    "NCCL_SYM_TMA_ENABLE=1"
    "NCCL_SYM_KERNEL=$TMA_KERNEL"
    "NCCL_SYM_CTAS=$SYM_CTAS"
    "NCCL_NET=${NCCL_NET:-Socket}"
    "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}"
  )
  NCCL_RUNTIME_LABEL="$NCCL_SO_REAL"
fi

run_env() {
  env "${ENV_UNSETS[@]}" "${ENV_ASSIGNMENTS[@]}" "$@"
}

if [[ "$NCCL_MODE" == "local-tma" && "$RUN_NCCL_PREFLIGHT" -eq 1 ]]; then
  run_env "$PYTHON_BIN" - "$NCCL_SO_REAL" <<'PY'
import ctypes
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()
lib = ctypes.CDLL(str(expected), mode=ctypes.RTLD_GLOBAL)
version = ctypes.c_int()
status = lib.ncclGetVersion(ctypes.byref(version))
if status != 0:
    raise SystemExit(f"ncclGetVersion failed with status {status}")

import torch  # noqa: E402

mapped = set()
for line in Path("/proc/self/maps").read_text().splitlines():
    if "libnccl.so" not in line:
        continue
    candidate = Path(line.split()[-1])
    try:
        mapped.add(candidate.resolve())
    except OSError:
        mapped.add(candidate)

print(f"[evalution] preloaded NCCL: {expected}")
print(f"[evalution] ncclGetVersion: {version.value}")
print(f"[evalution] torch NCCL version: {torch.cuda.nccl.version()}")
for path in sorted(mapped, key=str):
    print(f"[evalution] mapped NCCL: {path}")

if version.value != 23007:
    raise SystemExit(
        f"expected NCCL version 23007 (2.30.7), got {version.value}"
    )
if expected not in mapped:
    raise SystemExit(
        f"expected preloaded NCCL {expected} was not found in /proc/self/maps"
    )
PY
fi

mkdir -p "$OUT_DIR"
{
  echo "# collective=$COLLECTIVE"
  echo "# collective_argument=$COLLECTIVE_ARG"
  echo "# world_size=$WORLD_SIZE"
  echo "# devices=$DEVICES"
  echo "# backend=$BACKEND"
  echo "# nccl_mode=$NCCL_MODE"
  echo "# nccl_runtime=$NCCL_RUNTIME_LABEL"
  if [[ "$NCCL_MODE" == "local-tma" ]]; then
    echo "# nccl_sym_tma_enable=1"
    echo "# nccl_sym_kernel=$TMA_KERNEL"
    echo "# nccl_sym_ctas=$SYM_CTAS"
  fi
  echo "# ooverlap_max_ctas=$OOVERLAP_CTAS"
  echo "# ooverlap_max_ctas_per_reduce_task=$REDUCE_TASK_CTAS"
  echo "# tuning_policy=$TUNING_POLICY_LABEL"
  echo "# iters=$ITERS_VALUE warmup=$WARMUP_VALUE"
  printf "numel\tbytes\tooverlap_ms\tnccl_ms\tnccl_symmetric_ms\n"
} > "$SUMMARY"

extract_metric() {
  local key="$1"
  local file="$2"
  awk -F': ' -v wanted="$key" '
    {
      name=$1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      if (name == wanted) {
        print $2
        exit
      }
    }
  ' "$file"
}

cd "$REPO_ROOT"

echo "[evalution] external-P2P direct benchmark"
echo "[evalution] world_size=$WORLD_SIZE devices=$DEVICES collective=$COLLECTIVE"
echo "[evalution] backend=$BACKEND nccl_mode=$NCCL_MODE"
echo "[evalution] nccl_runtime=$NCCL_RUNTIME_LABEL"
if [[ "$NCCL_MODE" == "local-tma" ]]; then
  echo "[evalution] tma_kernel=$TMA_KERNEL sym_ctas=$SYM_CTAS"
fi
echo "[evalution] ooverlap_max_ctas=$OOVERLAP_CTAS"
echo "[evalution] ooverlap_max_ctas_per_reduce_task=$REDUCE_TASK_CTAS"
echo "[evalution] tuning_policy=$TUNING_POLICY_LABEL"
echo "[evalution] output=$OUT_DIR"

for numel in "${NUMELS[@]}"; do
  LOG_FILE="$OUT_DIR/numel${numel}.log"
  echo "[evalution] running numel=$numel log=$LOG_FILE"

  run_env "$PYTHON_BIN" "$DRIVER" \
    --mode bench \
    --collective "$COLLECTIVE" \
    --numel "$numel" \
    --iters "$ITERS_VALUE" \
    --warmup "$WARMUP_VALUE" \
    --devices "$DEVICES" \
    2>&1 | tee "$LOG_FILE"

  ooverlap_ms="$(extract_metric ooverlap_ms "$LOG_FILE" || true)"
  nccl_ms="$(extract_metric nccl_ms "$LOG_FILE" || true)"
  nccl_symmetric_ms="$(extract_metric nccl_symmetric_ms "$LOG_FILE" || true)"

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$numel" \
    "$((numel * 2))" \
    "${ooverlap_ms:-NA}" \
    "${nccl_ms:-NA}" \
    "${nccl_symmetric_ms:-NA}" \
    | tee -a "$SUMMARY"
done

echo "[evalution] wrote summary: $SUMMARY"
