#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./evalution/run_external_p2p_collective_bench.sh [--local-tma] {2|4} {AR|RS|AG} NUMEL [NUMEL ...]
#
# Default mode uses the NCCL already available in the environment.
# --local-tma preloads build/third_party/nccl/lib/libnccl.so.2 and enables
# NCCL symmetric TMA kernel eligibility. It does not force a specific kernel.

ITERS=300
WARMUP=20

LOCAL_TMA=0
if [[ "${1:-}" == "--local-tma" ]]; then
  LOCAL_TMA=1
  shift
fi

[[ $# -ge 3 ]] || {
  echo "usage: $0 [--local-tma] {2|4} {AR|RS|AG} NUMEL [NUMEL ...]" >&2
  exit 2
}

WORLD_SIZE="$1"
COLLECTIVE_ARG="${2^^}"
shift 2
NUMELS=("$@")

case "$COLLECTIVE_ARG" in
  AR) COLLECTIVE="allreduce" ;;
  RS) COLLECTIVE="reduce_scatter" ;;
  AG) COLLECTIVE="all_gather" ;;
  *)
    echo "error: collective must be AR, RS, or AG" >&2
    exit 2
    ;;
esac

case "$WORLD_SIZE" in
  2)
    DEVICES="0,1"
    OOVERLAP_CTAS=8
    REDUCE_TASK_CTAS=8
    ;;
  4)
    DEVICES="0,1,2,3"
    OOVERLAP_CTAS=9
    REDUCE_TASK_CTAS=3
    ;;
  *)
    echo "error: world size must be 2 or 4" >&2
    exit 2
    ;;
esac

for numel in "${NUMELS[@]}"; do
  [[ "$numel" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: NUMEL must be a positive integer: $numel" >&2
    exit 2
  }

  if [[ "$COLLECTIVE" != "allreduce" ]] &&
     (( numel % WORLD_SIZE != 0 )); then
    echo "error: $COLLECTIVE requires NUMEL divisible by $WORLD_SIZE: $numel" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DRIVER="$REPO_ROOT/test/test_external_p2p_collective.py"
EXTENSION="$REPO_ROOT/build/lib/ooverlap_ext.so"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -f "$DRIVER" ]] || {
  echo "error: benchmark driver not found: $DRIVER" >&2
  exit 1
}

[[ -f "$EXTENSION" ]] || {
  echo "error: Python extension not found: $EXTENSION" >&2
  exit 1
}

TUNING_POLICY="${OOVERLAP_TUNING_POLICY:-$REPO_ROOT/results/policies/tp${WORLD_SIZE}_policy.json}"
if [[ ! -f "$TUNING_POLICY" &&
      -f "$REPO_ROOT/results/policies/tp4_policy.json" ]]; then
  TUNING_POLICY="$REPO_ROOT/results/policies/tp4_policy.json"
fi

if (( LOCAL_TMA )); then
  MODE="local-tma"
else
  MODE="default"
fi

OUT_DIR="$REPO_ROOT/results/evalution/external_p2p/direct_bench/tp${WORLD_SIZE}/${MODE}/${COLLECTIVE}"
SUMMARY="$OUT_DIR/summary.tsv"
mkdir -p "$OUT_DIR"

ENV_UNSETS=(
  -u NCCL_DEBUG
  -u NCCL_DEBUG_SUBSYS
  -u OOVERLAP_BENCH_ONLY
)

ENV_ASSIGNMENTS=(
  "CUDA_VISIBLE_DEVICES=$DEVICES"
  "OOVERLAP_MAX_CTAS=${OOVERLAP_MAX_CTAS:-$OOVERLAP_CTAS}"
  "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK=${OOVERLAP_MAX_CTAS_PER_REDUCE_TASK:-$REDUCE_TASK_CTAS}"
  "NCCL_NET=${NCCL_NET:-Socket}"
  "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}"
)

if [[ -f "$TUNING_POLICY" ]]; then
  ENV_ASSIGNMENTS+=("OOVERLAP_TUNING_POLICY=$TUNING_POLICY")
  TUNING_POLICY_LABEL="$TUNING_POLICY"
else
  ENV_UNSETS+=(-u OOVERLAP_TUNING_POLICY)
  TUNING_POLICY_LABEL="not found"
fi

NCCL_RUNTIME="environment/default"

if (( LOCAL_TMA )); then
  NCCL_SO="$REPO_ROOT/build/third_party/nccl/lib/libnccl.so.2"

  [[ -e "$NCCL_SO" ]] || {
    echo "error: local NCCL library not found: $NCCL_SO" >&2
    exit 1
  }

  NCCL_SO="$(readlink -f "$NCCL_SO")"
  NCCL_LIBDIR="$(dirname -- "$NCCL_SO")"

  GCC_LIBDIR=""
  if command -v g++ >/dev/null 2>&1; then
    GCC_LIBSTDCPP="$(g++ -print-file-name=libstdc++.so.6 2>/dev/null || true)"
    if [[ -n "$GCC_LIBSTDCPP" &&
          "$GCC_LIBSTDCPP" != "libstdc++.so.6" ]]; then
      GCC_LIBDIR="$(dirname -- "$(readlink -f "$GCC_LIBSTDCPP")")"
    fi
  fi

  LOCAL_LIBRARY_PATH="$NCCL_LIBDIR"
  [[ -n "$GCC_LIBDIR" ]] &&
    LOCAL_LIBRARY_PATH="$LOCAL_LIBRARY_PATH:$GCC_LIBDIR"
  [[ -n "${LD_LIBRARY_PATH:-}" ]] &&
    LOCAL_LIBRARY_PATH="$LOCAL_LIBRARY_PATH:$LD_LIBRARY_PATH"

  LOCAL_PRELOAD="$NCCL_SO"
  [[ -n "${LD_PRELOAD:-}" ]] &&
    LOCAL_PRELOAD="$LOCAL_PRELOAD:$LD_PRELOAD"

  # Leave NCCL_SYM_KERNEL unset so NCCL performs automatic kernel selection.
  # NCCL_SYM_TMA_ENABLE=1 makes TMA kernels eligible; it does not guarantee
  # that the tuner will choose one.
  ENV_UNSETS+=(-u NCCL_SYM_KERNEL)
  ENV_ASSIGNMENTS+=(
    "LD_LIBRARY_PATH=$LOCAL_LIBRARY_PATH"
    "LD_PRELOAD=$LOCAL_PRELOAD"
    "NCCL_SYM_TMA_ENABLE=1"
  )

  NCCL_RUNTIME="$NCCL_SO"
else
  ENV_UNSETS+=(
    -u NCCL_SYM_TMA_ENABLE
    -u NCCL_SYM_KERNEL
    -u NCCL_SYM_CTAS
  )
fi

run_env() {
  env "${ENV_UNSETS[@]}" "${ENV_ASSIGNMENTS[@]}" "$@"
}

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

{
  echo "# collective=$COLLECTIVE"
  echo "# world_size=$WORLD_SIZE"
  echo "# devices=$DEVICES"
  echo "# mode=$MODE"
  echo "# nccl_runtime=$NCCL_RUNTIME"
  echo "# nccl_sym_tma_enable=$LOCAL_TMA"
  echo "# nccl_sym_kernel=automatic"
  echo "# nccl_sym_ctas=${NCCL_SYM_CTAS:-default}"
  echo "# tuning_policy=$TUNING_POLICY_LABEL"
  echo "# iters=$ITERS warmup=$WARMUP"
  printf "numel\tbytes\tooverlap_ms\tnccl_ms\tnccl_symmetric_ms\n"
} > "$SUMMARY"

cd "$REPO_ROOT"

echo "[evalution] collective=$COLLECTIVE world_size=$WORLD_SIZE mode=$MODE"
echo "[evalution] nccl_runtime=$NCCL_RUNTIME"
echo "[evalution] output=$OUT_DIR"

for numel in "${NUMELS[@]}"; do
  LOG_FILE="$OUT_DIR/numel${numel}.log"
  echo "[evalution] running numel=$numel"

  run_env "$PYTHON_BIN" "$DRIVER" \
    --mode bench \
    --collective "$COLLECTIVE" \
    --numel "$numel" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
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
