#!/usr/bin/env bash
# Seeds 0, 1, 2 in parallel, by default all on GPU 0 (GPU policy 2026-10-16; GPUS="0 1 3" to spread when allowed).
#   bash scripts/run_battle_parallel.sh <tag> [train_battle.py args...]     -> tmux sessions <tag>_s{0,1,2}
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?tag}; shift
i=0
for G in ${GPUS:-0 0 0}; do   # default: all seeds on GPU 0 (GPU policy 2026-10-16)
  tmux new-session -d -s "${TAG}_s$i" "source scripts/env.sh; echo \"[\$(date)] commit \$(git rev-parse --short HEAD) gpu $G :: $*\" > logs/stdout/battle_${TAG}_s$i.log; CUDA_VISIBLE_DEVICES=$G \$MLAVL_PY ours/train_battle.py --name ${TAG}_s$i --seed $i $* 2>&1 | grep --line-buffered -v -i warn >> logs/stdout/battle_${TAG}_s$i.log"
  i=$((i + 1))
done
