#!/usr/bin/env bash
# Generic 3-seed RG sweep launcher (Phase 3). Each setting runs on all 4 apparatus x seeds 0,1,2 = 12 runs;
# settings are run one after another in the same tmux windows.
#
#   bash scripts/rg_sweep.sh <session> <tag1> "<extra args 1>" [<tag2> "<extra args 2>" ...]
#   e.g. bash scripts/rg_sweep.sh mlavl_p3_wd wd1e-3 "--weight-decay 1e-3" wd1e-2 "--weight-decay 1e-2"
#
# Layout: GPU 0 = seed 0, GPU 1 = seed 1, GPU 3 = seed 2 (GPU 2 is banned). On each GPU the 4 apparatus run
# concurrently (one tmux window each, ~1.8 GB VRAM per run, ~2 h for the longest setting).
# Logs: logs/stdout/rg_{App}_s{seed}_{tag}_*.log ; summarise one setting with
#   $MLAVL_PY scripts/summarize_runs.py --seeds logs/stdout/rg_*_<tag>_*.log
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=${1:?tmux session name}; shift
[ $# -ge 2 ] && [ $(( $# % 2 )) -eq 0 ] || { echo "need <tag> \"<args>\" pairs"; exit 1; }
declare -A GPU_OF_SEED=([0]=0 [1]=1 [2]=3)
tmux new-session -d -s "$SESSION" -n placeholder
for SEED in 0 1 2; do
  GPU=${GPU_OF_SEED[$SEED]}
  for APP in Ball Clubs Hoop Ribbon; do
    CHAIN=""
    set -- "$@"
    for ((i = 1; i <= $#; i += 2)); do
      TAG=${!i}; j=$((i + 1)); ARGS=${!j}
      CHAIN+="SEED=$SEED TAG=$TAG bash scripts/run_rg.sh $APP $GPU $ARGS; "
    done
    tmux new-window -t "$SESSION" -n "${APP}_s${SEED}" "${CHAIN}sleep 5"
  done
done
tmux kill-window -t "$SESSION:placeholder"
tmux list-windows -t "$SESSION"
