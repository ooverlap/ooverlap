#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {2|4} {AR|RS|AG} NUMEL [NUMEL ...]" >&2
  echo "Example: $0 4 AG 65536 2097152 16777216 67108864" >&2
  exit 2
}

[[ $# -ge 3 ]] || usage
WORLD_SIZE="$1"
COLLECTIVE_ARG="${2^^}"
shift 2
NUMELS=("$@")

case "$COLLECTIVE_ARG" in
  AR)
    COLLECTIVE="allreduce"
    ;;
  RS)
    COLLECTIVE="reduce_scatter"
    ;;
  AG)
    COLLECTIVE="all_gather"
    ;;
  *)
    usage
    ;;
esac

# Match the unrestricted settings in run_external_p2p_bandwidth_latency.sh.
case "$WORLD_SIZE" in
  2)
    DEVICES="0,1"
    OOVERLAP_CTAS="8"
    MAX_CTAS_PER_REDUCE_TASK="8"
    ;;
  4)
    DEVICES="0,1,2,3"
    OOVERLAP_CTAS="9"
    MAX_CTAS_PER_REDUCE_TASK="3"
    ;;
  *)
    usage
    ;;
esac

for numel in "${NUMELS[@]}"; do
  [[ "$numel" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: NUMEL must be a positive integer; got: $numel" >&2
    exit 2
  }
  if [[ "$COLLECTIVE" != "allreduce" ]] && ((numel % WORLD_SIZE != 0)); then
    echo "error: $COLLECTIVE requires NUMEL divisible by world size $WORLD_SIZE; got: $numel" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DRIVER="$REPO_ROOT/test/test_external_p2p_collective.py"
ANALYZER="$REPO_ROOT/tool/nsys_sm_usage.py"
TUNING_POLICY="$REPO_ROOT/results/policies/tp4_policy.json"
PYTHON_BIN="${PYTHON_BIN:-python}"
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-100}"
TIMING_ITERS="${TIMING_ITERS:-300}"
TIMING_WARMUP="${TIMING_WARMUP:-20}"
PROFILE_TO_TIMING_COOLDOWN_SECONDS="${PROFILE_TO_TIMING_COOLDOWN_SECONDS:-10}"
NCCL_SO="$REPO_ROOT/build/third_party/nccl/lib/libnccl.so.2"

OUT_DIR="$REPO_ROOT/results/evalution/external_p2p/sm_usage/tp${WORLD_SIZE}/unres/${COLLECTIVE}"
SM_FILE="$OUT_DIR/sm_usage.txt"
TIMING_FILE="$OUT_DIR/timing_speedup.txt"
SUMMARY_FILE="$OUT_DIR/sm_usage_speedup.txt"

[[ -f "$DRIVER" ]] || {
  echo "error: benchmark driver not found: $DRIVER" >&2
  exit 1
}
[[ -f "$ANALYZER" ]] || {
  echo "error: analyzer not found: $ANALYZER" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "error: Python executable not found: $PYTHON_BIN" >&2
  exit 1
}
command -v nsys >/dev/null 2>&1 || {
  echo "error: nsys not found" >&2
  exit 1
}
command -v awk >/dev/null 2>&1 || {
  echo "error: awk not found" >&2
  exit 1
}
[[ -e "$NCCL_SO" ]] || {
  echo "error: patched local NCCL library not found: $NCCL_SO" >&2
  exit 1
}

NCCL_SO="$(readlink -f "$NCCL_SO")"
NCCL_LIBDIR="$(dirname -- "$NCCL_SO")"

GCC_LIBDIR=""
if command -v g++ >/dev/null 2>&1; then
  GCC_LIBSTDCPP="$(g++ -print-file-name=libstdc++.so.6 2>/dev/null || true)"
  if [[ -n "$GCC_LIBSTDCPP" && "$GCC_LIBSTDCPP" != "libstdc++.so.6" ]]; then
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

for pair in \
  "ITERS:$ITERS" \
  "TIMING_ITERS:$TIMING_ITERS"; do
  if ! [[ "${pair#*:}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${pair%%:*} must be a positive integer; got: ${pair#*:}" >&2
    exit 2
  fi
done

for pair in \
  "WARMUP:$WARMUP" \
  "TIMING_WARMUP:$TIMING_WARMUP"; do
  if ! [[ "${pair#*:}" =~ ^[0-9]+$ ]]; then
    echo "error: ${pair%%:*} must be a non-negative integer; got: ${pair#*:}" >&2
    exit 2
  fi
done

if ! [[ "$PROFILE_TO_TIMING_COOLDOWN_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "error: PROFILE_TO_TIMING_COOLDOWN_SECONDS must be non-negative; got: $PROFILE_TO_TIMING_COOLDOWN_SECONDS" >&2
  exit 2
fi

unset OOVERLAP_BENCH_ONLY
unset OOVERLAP_MAX_CTAS
unset OOVERLAP_MAX_CTAS_PER_REDUCE_TASK
unset NCCL_MIN_CTAS
unset NCCL_MAX_CTAS
unset NCCL_SYM_TMA_ENABLE
unset NCCL_SYM_KERNEL
unset NCCL_SYM_CTAS
unset OOVERLAP_TUNING_POLICY

export CUDA_VISIBLE_DEVICES="$DEVICES"

if [[ -f "$TUNING_POLICY" ]]; then
  export OOVERLAP_TUNING_POLICY="$TUNING_POLICY"
  TUNING_POLICY_LABEL="$TUNING_POLICY"
else
  echo "[evalution] warning: tuning policy not found: $TUNING_POLICY; continuing without it" >&2
  TUNING_POLICY_LABEL="not found (runtime fallback)"
fi

SMS_PER_DEVICE="$(
  "$PYTHON_BIN" - "$WORLD_SIZE" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != expected:
    raise SystemExit(
        f"expected {expected} visible GPUs, found {torch.cuda.device_count()}"
    )
counts = [
    torch.cuda.get_device_properties(device).multi_processor_count
    for device in range(expected)
]
if len(set(counts)) != 1:
    raise SystemExit(f"profiled GPUs have different SM counts: {counts}")
print(counts[0])
PY
)"

DEFAULT_ENV_UNSETS=(
  -u NCCL_DEBUG
  -u NCCL_DEBUG_SUBSYS
  -u NCCL_SYM_TMA_ENABLE
  -u NCCL_SYM_KERNEL
  -u NCCL_SYM_CTAS
)

TMA_ENV_UNSETS=(
  -u NCCL_DEBUG
  -u NCCL_DEBUG_SUBSYS
  -u NCCL_SYM_KERNEL
)

COMMON_ENV_ASSIGNMENTS=(
  "CUDA_VISIBLE_DEVICES=$DEVICES"
  "NCCL_NET=${NCCL_NET:-Socket}"
  "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}"
)

TMA_ENV_ASSIGNMENTS=(
  "LD_LIBRARY_PATH=$LOCAL_LIBRARY_PATH"
  "LD_PRELOAD=$LOCAL_PRELOAD"
  "NCCL_SYM_TMA_ENABLE=1"
)

run_default_env() {
  env "${DEFAULT_ENV_UNSETS[@]}" "${COMMON_ENV_ASSIGNMENTS[@]}" "$@"
}

run_local_tma_env() {
  env \
    "${TMA_ENV_UNSETS[@]}" \
    "${COMMON_ENV_ASSIGNMENTS[@]}" \
    "${TMA_ENV_ASSIGNMENTS[@]}" \
    "$@"
}

write_common_metadata() {
  local output_kind="$1"
  echo "# output=$output_kind"
  echo "# collective=$COLLECTIVE"
  echo "# collective_argument=$COLLECTIVE_ARG"
  echo "# world_size=$WORLD_SIZE"
  echo "# devices=$DEVICES"
  echo "# dtype=fp16"
  echo "# bytes=numel*2"
  echo "# ooverlap_max_ctas=$OOVERLAP_CTAS"
  echo "# max_ctas_per_reduce_task=$MAX_CTAS_PER_REDUCE_TASK"
  echo "# nccl_ctas=default"
  echo "# tuning_policy=$TUNING_POLICY_LABEL"
  echo "# sms_per_device=$SMS_PER_DEVICE"
  echo "# profile_iters=$ITERS profile_warmup=$WARMUP"
  echo "# timing_iters=$TIMING_ITERS timing_warmup=$TIMING_WARMUP"
  echo "# nccl_tma_mode=symmetric NCCL with NCCL_SYM_TMA_ENABLE=1"
  echo "# nccl_tma_runtime=$NCCL_SO"
  echo "# nccl_sym_kernel=automatic"
  echo "# nccl_sym_ctas=${NCCL_SYM_CTAS:-default}"
  echo "# execution_order=all nsys SM-usage runs, cooldown, then all unprofiled timing runs"
}

extract_metric() {
  local log_file="$1"
  local metric_name="$2"
  local value

  value="$(awk -v metric="${metric_name}:" '
    $1 == metric { value = $2 }
    END {
      if (value == "") {
        exit 1
      }
      print value
    }
  ' "$log_file")" || {
    echo "error: metric $metric_name not found in $log_file" >&2
    exit 1
  }

  printf '%s\n' "$value"
}

speedup_ratio() {
  local baseline_ms="$1"
  local ooverlap_ms="$2"

  "$PYTHON_BIN" - "$baseline_ms" "$ooverlap_ms" <<'PY'
import math
import sys

baseline = float(sys.argv[1])
ooverlap = float(sys.argv[2])
if not math.isfinite(baseline) or not math.isfinite(ooverlap):
    raise SystemExit("timing must be finite")
if baseline <= 0.0 or ooverlap <= 0.0:
    raise SystemExit("timing must be positive")
print(f"{baseline / ooverlap:.6f}")
PY
}

mkdir -p "$OUT_DIR"

{
  write_common_metadata "average active SM usage from nsys GPU metrics"
  printf "numel\tbytes\tooverlap_avg_gpu_median_active_sms\tnccl_avg_gpu_median_active_sms\tnccl_symmetric_avg_gpu_median_active_sms\tnccl_tma_avg_gpu_median_active_sms\n"
} > "$SM_FILE"

{
  write_common_metadata "unprofiled latency and T-CCL speedup"
  printf "numel\tbytes\tooverlap_ms\tnccl_ms\tnccl_symmetric_ms\tnccl_tma_ms\tooverlap_speedup_over_nccl\tooverlap_speedup_over_nccl_symmetric\tooverlap_speedup_over_nccl_tma\n"
} > "$TIMING_FILE"

{
  write_common_metadata "paper table input: SM usage plus unprofiled latency and speedup"
  printf "numel\tbytes\tooverlap_avg_gpu_median_active_sms\tnccl_avg_gpu_median_active_sms\tnccl_symmetric_avg_gpu_median_active_sms\tnccl_tma_avg_gpu_median_active_sms\tooverlap_ms\tnccl_ms\tnccl_symmetric_ms\tnccl_tma_ms\tooverlap_speedup_over_nccl\tooverlap_speedup_over_nccl_symmetric\tooverlap_speedup_over_nccl_tma\n"
} > "$SUMMARY_FILE"

cd "$REPO_ROOT"

declare -A SM_RESULT=()
PROFILE_BACKENDS=(ooverlap nccl nccl_symmetric nccl_tma)

echo "[evalution] phase=sm_usage begin"
echo "[evalution] all nsys profiling completes before unprofiled timing begins"

for numel in "${NUMELS[@]}"; do
  for backend in "${PROFILE_BACKENDS[@]}"; do
    report="$OUT_DIR/${backend}_numel${numel}"
    selected_backend="$backend"

    if [[ "$backend" == "nccl_tma" ]]; then
      selected_backend="nccl_symmetric"
    fi

    echo "[evalution] phase=sm_usage collective=$COLLECTIVE backend=$backend numel=$numel report=$report"

    profile_env=(
      "OOVERLAP_BENCH_ONLY=$selected_backend"
    )
    if [[ "$backend" == "ooverlap" ]]; then
      profile_env+=(
        "OOVERLAP_MAX_CTAS=$OOVERLAP_CTAS"
        "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK=$MAX_CTAS_PER_REDUCE_TASK"
      )
    fi

    if [[ "$backend" == "nccl_tma" ]]; then
      profile_target=(
        env
        "${TMA_ENV_UNSETS[@]}"
        "${COMMON_ENV_ASSIGNMENTS[@]}"
        "${TMA_ENV_ASSIGNMENTS[@]}"
        "${profile_env[@]}"
      )
    else
      profile_target=(
        env
        "${DEFAULT_ENV_UNSETS[@]}"
        "${COMMON_ENV_ASSIGNMENTS[@]}"
        "${profile_env[@]}"
      )
    fi

    nsys profile \
      --force-overwrite=true \
      --trace=cuda \
      --gpu-metrics-devices=cuda-visible \
      --gpu-metrics-frequency=20000 \
      --output="$report" \
      "${profile_target[@]}" \
      "$PYTHON_BIN" "$DRIVER" \
        --mode bench \
        --collective "$COLLECTIVE" \
        --numel "$numel" \
        --iters "$ITERS" \
        --warmup "$WARMUP" \
        --devices "$DEVICES"

    nsys export \
      --type sqlite \
      --force-overwrite=true \
      --output="${report}.sqlite" \
      "${report}.nsys-rep"

    SM_RESULT["$backend:$numel"]="$($PYTHON_BIN "$ANALYZER" \
      "${report}.sqlite" \
      --sms-per-device "$SMS_PER_DEVICE" \
      --value-only)"
  done

  bytes=$((numel * 2))
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$numel" \
    "$bytes" \
    "${SM_RESULT[ooverlap:$numel]}" \
    "${SM_RESULT[nccl:$numel]}" \
    "${SM_RESULT[nccl_symmetric:$numel]}" \
    "${SM_RESULT[nccl_tma:$numel]}" | tee -a "$SM_FILE"
done

echo "[evalution] phase=sm_usage complete"
echo "[evalution] cooling down ${PROFILE_TO_TIMING_COOLDOWN_SECONDS}s before timing"
sleep "$PROFILE_TO_TIMING_COOLDOWN_SECONDS"

echo "[evalution] phase=timing begin"

for numel in "${NUMELS[@]}"; do
  default_log="$OUT_DIR/timing_default_numel${numel}.log"
  tma_log="$OUT_DIR/timing_nccl_tma_numel${numel}.log"

  echo "[evalution] phase=timing collective=$COLLECTIVE backends=ooverlap,nccl,nccl_symmetric numel=$numel"
  run_default_env \
    OOVERLAP_BENCH_ONLY=all \
    OOVERLAP_MAX_CTAS="$OOVERLAP_CTAS" \
    OOVERLAP_MAX_CTAS_PER_REDUCE_TASK="$MAX_CTAS_PER_REDUCE_TASK" \
    "$PYTHON_BIN" "$DRIVER" \
      --mode bench \
      --collective "$COLLECTIVE" \
      --numel "$numel" \
      --iters "$TIMING_ITERS" \
      --warmup "$TIMING_WARMUP" \
      --devices "$DEVICES" \
      2>&1 | tee "$default_log"

  ooverlap_ms="$(extract_metric "$default_log" ooverlap_ms)"
  nccl_ms="$(extract_metric "$default_log" nccl_ms)"
  nccl_symmetric_ms="$(extract_metric "$default_log" nccl_symmetric_ms)"
  speedup_nccl="$(speedup_ratio "$nccl_ms" "$ooverlap_ms")"
  speedup_nccl_symmetric="$(speedup_ratio "$nccl_symmetric_ms" "$ooverlap_ms")"

  echo "[evalution] phase=timing collective=$COLLECTIVE backend=nccl_tma numel=$numel"
  run_local_tma_env \
    OOVERLAP_BENCH_ONLY=nccl_symmetric \
    "$PYTHON_BIN" "$DRIVER" \
      --mode bench \
      --collective "$COLLECTIVE" \
      --numel "$numel" \
      --iters "$TIMING_ITERS" \
      --warmup "$TIMING_WARMUP" \
      --devices "$DEVICES" \
      2>&1 | tee "$tma_log"

  nccl_tma_ms="$(extract_metric "$tma_log" nccl_symmetric_ms)"
  speedup_nccl_tma="$(speedup_ratio "$nccl_tma_ms" "$ooverlap_ms")"
  bytes=$((numel * 2))

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$numel" \
    "$bytes" \
    "$ooverlap_ms" \
    "$nccl_ms" \
    "$nccl_symmetric_ms" \
    "$nccl_tma_ms" \
    "$speedup_nccl" \
    "$speedup_nccl_symmetric" \
    "$speedup_nccl_tma" | tee -a "$TIMING_FILE"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$numel" \
    "$bytes" \
    "${SM_RESULT[ooverlap:$numel]}" \
    "${SM_RESULT[nccl:$numel]}" \
    "${SM_RESULT[nccl_symmetric:$numel]}" \
    "${SM_RESULT[nccl_tma:$numel]}" \
    "$ooverlap_ms" \
    "$nccl_ms" \
    "$nccl_symmetric_ms" \
    "$nccl_tma_ms" \
    "$speedup_nccl" \
    "$speedup_nccl_symmetric" \
    "$speedup_nccl_tma" | tee -a "$SUMMARY_FILE"
done

echo "[evalution] phase=timing complete"
echo "[evalution] wrote SM usage: $SM_FILE"
echo "[evalution] wrote timing/speedup: $TIMING_FILE"
echo "[evalution] wrote combined paper table input: $SUMMARY_FILE"
