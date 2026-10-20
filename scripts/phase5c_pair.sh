#!/usr/bin/env bash
# Phase 5c improvement 1: pairwise MLAVL (shared weights, loss on the round margin), GPU 0, 4 threads per run.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs/wdsf
run() { [ -f logs/wdsf/$1/summary.json ] || $MLAVL_PY ours/train_wdsf_pair.py --name $1 "${@:2}" > logs/wdsf/$1.log 2>&1; }
for s in 0 1 2; do run pair_total_s$s --target total --seed $s & done
run pair_Technique_s0 --target Technique --seed 0 &
run pair_Execution_s0 --target Execution --seed 0 &
wait
echo "[$(date)] done"
