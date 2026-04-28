#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <ctas>"
  echo "example: $0 4"
  echo "example: $0 4,8,16"
  exit 1
fi

CTAS_ARG="$1"

OUT="results/tma_allreduce_sweep.jsonl"

NUMELS="131072,262144,524288,1048576,2097152,4194304,8388608,12582912,16777216,20971520,25165824,29360128,33554432,37748736,41943040,46137344,50331648,54525952,58720256,62914560,67108864,100663296,134217728,201326592,268435456"
KERNELS="tma_copy,seq_fast_gmem,overlap_fast_gmem"
THREADS="128,256,512,1024"
WINDOW_CHUNKS="8,16,32,64,128,256,512"

# Must match the variants instantiated in src/comm/tma_variant_config.h.
VARIANTS="4096:32,8192:16,16384:8,32768:4,65536:2"

ITERS=50
WARMUP=10
DEV0=0
DEV1=1

mkdir -p results

IFS=',' read -ra CTAS_VALUES <<< "$CTAS_ARG"

for CTAS in "${CTAS_VALUES[@]}"; do
  echo "[run] CTAS=${CTAS}"

  python test/sweep_tma_allreduce.py \
    --numels "${NUMELS}" \
    --kernels "${KERNELS}" \
    --threads "${THREADS}" \
    --max-ctas "${CTAS}" \
    --window-chunks "${WINDOW_CHUNKS}" \
    --variants "${VARIANTS}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    --dev0 "${DEV0}" \
    --dev1 "${DEV1}" \
    --nccl-max-ctas "${CTAS}" \
    --out "${OUT}"
done

echo "PASS: wrote ${OUT}"
