#!/usr/bin/env bash
# Phase 5: siamese MLAVL battle winner prediction on BRACE, leave-one-year-out, 3 seeds per setting.
#   bash scripts/run_battle.sh <gpu> <tag> [train_battle.py args...]
#   e.g. bash scripts/run_battle.sh 0 pcs --init ckpt/fisv_PCS_1011_1554_best.pkl
# Runs seeds 0,1,2 sequentially; results in logs/battle/{tag}_s{seed}/ and logs/stdout/battle_{tag}_s{seed}.log
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
GPU=${1:?gpu id (0, 1 or 3)}; TAG=${2:?tag}; shift 2
[[ "$GPU" == "2" ]] && { echo "GPU 2 is banned"; exit 1; }
mkdir -p logs/stdout
for SEED in 0 1 2; do
  LOG=logs/stdout/battle_${TAG}_s${SEED}.log
  echo "[$(date)] commit $(git rev-parse --short HEAD) gpu $GPU :: $*" | tee "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU "$MLAVL_PY" ours/train_battle.py --name "${TAG}_s${SEED}" --seed "$SEED" "$@" 2>&1 \
    | grep --line-buffered -v -i 'warn' | tee -a "$LOG"
done
