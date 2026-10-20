#!/usr/bin/env bash
# Phase 5c: pairwise MLAVL with the epoch chosen on a held-out training event (no test-fold selection),
# from scratch and from the reproduced RG Ribbon checkpoint. GPU 0, 4 threads per run.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RG=ckpt/rg_Ribbon_s2_1011_1935_best.pkl
run() { [ -f logs/wdsf/$1/summary.json ] || $MLAVL_PY ours/train_wdsf_pair.py --name $1 "${@:2}" > logs/wdsf/$1.log 2>&1; }
for s in 0 1 2; do run pairv_total_s$s --target total --seed $s & done
run pairv_Technique_s0 --target Technique --seed 0 &
run pairv_Execution_s0 --target Execution --seed 0 &
run pairv_total_rg_s0 --target total --seed 0 --init $RG &
run pairv_Technique_rg_s0 --target Technique --seed 0 --init $RG &
wait
echo "[$(date)] done"
