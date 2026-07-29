#!/usr/bin/env bash
set -euo pipefail

CSV_DIR="${CSV_DIR:-$HOME/csv_out_h100}"
PROFILE_SCRIPT="${PROFILE_SCRIPT:-./profile_h100_dense_f16_gemm.sh}"

mkdir -p "$CSV_DIR/logs"

# Same shape grid as the old csv_out_h100 directory.
M_LIST=(4096 8192 16384 32768 49152)
N_LIST=(2048 4096 8192)
K_LIST=(1024 2048 4096 8192)

FAIL_LOG="$CSV_DIR/failed_shapes_pow2.txt"
DONE_LOG="$CSV_DIR/done_shapes_pow2.txt"

touch "$FAIL_LOG" "$DONE_LOG"

is_good_csv() {
  local csv="$1"
  [[ -f "$csv" ]] || return 1

  python - "$csv" <<'PY'
import sys
import pandas as pd

p = sys.argv[1]

try:
    df = pd.read_csv(p)
except Exception:
    raise SystemExit(1)

if len(df) < 1:
    raise SystemExit(1)

if "Runtime" not in df.columns:
    raise SystemExit(1)

if "Operation" not in df.columns:
    raise SystemExit(1)

bad = df["Operation"].astype(str).str.contains(
    "sptensorop|spgemm|bf16|e4m3|e5m2",
    case=False,
    regex=True
).any()

if bad:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

total=0
skipped=0
ok=0
failed=0

echo "[info] CSV_DIR=$CSV_DIR"
echo "[info] PROFILE_SCRIPT=$PROFILE_SCRIPT"
echo "[info] M_LIST=${M_LIST[*]}"
echo "[info] N_LIST=${N_LIST[*]}"
echo "[info] K_LIST=${K_LIST[*]}"

if [[ ! -x "$PROFILE_SCRIPT" ]]; then
  echo "[error] Profile script is not executable: $PROFILE_SCRIPT"
  echo "Run: chmod +x $PROFILE_SCRIPT"
  exit 1
fi

for M in "${M_LIST[@]}"; do
  for N in "${N_LIST[@]}"; do
    for K in "${K_LIST[@]}"; do
      total=$((total + 1))

      csv="$CSV_DIR/m${M}n${N}k${K}.gemm.csv"
      log="$CSV_DIR/logs/m${M}n${N}k${K}.log"

      if is_good_csv "$csv"; then
        echo "[skip] M=$M N=$N K=$K already has good CSV"
        skipped=$((skipped + 1))
        continue
      fi

      echo "[run] M=$M N=$N K=$K"
      echo "[run] output: $csv"
      echo "[run] log: $log"

      set +e
      "$PROFILE_SCRIPT" "$M" "$N" "$K" "$CSV_DIR" >"$log" 2>&1
      rc=$?
      set -e

      if [[ $rc -ne 0 ]]; then
        echo "[fail] M=$M N=$N K=$K rc=$rc"
        echo "$M $N $K rc=$rc" >> "$FAIL_LOG"
        failed=$((failed + 1))
        continue
      fi

      if is_good_csv "$csv"; then
        echo "[ok] M=$M N=$N K=$K"
        echo "$M $N $K" >> "$DONE_LOG"
        ok=$((ok + 1))
      else
        echo "[bad] M=$M N=$N K=$K produced missing/invalid CSV"
        echo "$M $N $K bad_csv" >> "$FAIL_LOG"
        failed=$((failed + 1))
      fi

      echo "[progress] total_seen=$total ok=$ok skipped=$skipped failed=$failed"
    done
  done
done

echo "[summary] total_seen=$total ok=$ok skipped=$skipped failed=$failed"
echo "[summary] CSV_DIR=$CSV_DIR"
echo "[summary] failures=$FAIL_LOG"
