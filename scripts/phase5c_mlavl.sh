#!/usr/bin/env bash
# Phase 5c: MLAVL baseline as in the paper on WDSF judge scores (GPU 0). One model per score, like the paper's
# per-score models on FS1000 / Fis-V: total (3 seeds) and the 5 criteria (seed 0). Runs in parallel (CPU-bound MAG²).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # shared server: 7 runs x 4 threads
mkdir -p logs/wdsf
run() { [ -f logs/wdsf/$1/summary.json ] || $MLAVL_PY ours/train_wdsf.py --name $1 "${@:2}" > logs/wdsf/$1.log 2>&1; }
run wdsf_total_s1 --target total --seed 1 &
run wdsf_total_s2 --target total --seed 2 &
for c in Technique Vocabulary Originality Execution Musicality; do
  run wdsf_${c}_s0 --target $c --seed 0 &
done
wait
echo "[$(date)] done"
