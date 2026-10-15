#!/usr/bin/env bash
# One seed per GPU in parallel (GPUs 0, 1, 3 = seeds 0, 1, 2), for long leave-one-year-out runs (--test_on all).
#   bash scripts/run_battle_parallel.sh <tag> [train_battle.py args...]     -> tmux sessions <tag>_s{0,1,2}
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?tag}; shift
i=0
for G in 0 1 3; do
  tmux new-session -d -s "${TAG}_s$i" "source scripts/env.sh; echo \"[\$(date)] commit \$(git rev-parse --short HEAD) gpu $G :: $*\" > logs/stdout/battle_${TAG}_s$i.log; CUDA_VISIBLE_DEVICES=$G \$MLAVL_PY ours/train_battle.py --name ${TAG}_s$i --seed $i $* 2>&1 | grep --line-buffered -v -i warn >> logs/stdout/battle_${TAG}_s$i.log"
  i=$((i + 1))
done
