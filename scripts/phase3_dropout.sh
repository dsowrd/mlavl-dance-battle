#!/usr/bin/env bash
# Phase 3, experiment 1: stronger regularisation on RG. Everything in the reproduction over-fits early
# (train rho 0.99, best test rho at epoch 40-350), so raise the Transformer dropout (S2CE encoder, AVCF and DPG decoders)
# from the paper's 0.3 to 0.5 and 0.7. 3 seeds each, to compare with the 3-seed baseline (rho 0.775 / MSE 5.10).
#
#   bash scripts/phase3_dropout.sh            # launches tmux session mlavl_p3_dropout: 24 runs on GPUs 0, 1, 3
#
# Layout: GPU 0 = seed 0, GPU 1 = seed 1, GPU 3 = seed 2. On each GPU the 4 apparatus run concurrently (one tmux window
# each, ~1.8 GB VRAM per run) and each window runs dropout 0.5 then dropout 0.7 sequentially (~2 h + ~2 h).
# Logs: logs/stdout/rg_{App}_s{seed}_dp{0.5,0.7}_*.log ; summarise with
#   $MLAVL_PY scripts/summarize_runs.py --seeds logs/stdout/rg_*_dp0.5_*.log
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=mlavl_p3_dropout
declare -A GPU_OF_SEED=([0]=0 [1]=1 [2]=3)
tmux new-session -d -s "$SESSION" -n placeholder
for SEED in 0 1 2; do
  GPU=${GPU_OF_SEED[$SEED]}
  for APP in Ball Clubs Hoop Ribbon; do
    tmux new-window -t "$SESSION" -n "${APP}_s${SEED}" \
      "SEED=$SEED TAG=dp0.5 bash scripts/run_rg.sh $APP $GPU --dropout 0.5; SEED=$SEED TAG=dp0.7 bash scripts/run_rg.sh $APP $GPU --dropout 0.7; sleep 5"
  done
done
tmux kill-window -t "$SESSION:placeholder"
tmux list-windows -t "$SESSION"
