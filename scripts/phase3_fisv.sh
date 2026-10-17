#!/usr/bin/env bash
# Phase 3 on Fis-V: single-task control, multi-task, LDS, multi-task + LDS; 3 seeds each; all on GPU 0, at most
# $PAR runs at a time (GPU policy 2026-10-16). Logs: logs/stdout/<name>.log, results: logs/fisv/<name>.json
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
PAR=${PAR:-6}
jobs=()
for s in 0 1 2; do
  jobs+=("fisv_st_TES_s$s --score TES" "fisv_st_PCS_s$s --score PCS" "fisv_mt_s$s --multitask")
done
for s in 0 1 2; do
  jobs+=("fisv_lds_TES_s$s --score TES --lds" "fisv_lds_PCS_s$s --score PCS --lds" "fisv_mtlds_s$s --multitask --lds")
done
for j in "${jobs[@]}"; do
  name=${j%% *}; args=${j#* }; seed=${name##*_s}
  while [ "$(pgrep -fc 'ours/train_fisv.py')" -ge "$PAR" ]; do sleep 30; done
  [ -f "logs/fisv/$name.json" ] && continue
  echo "[$(date)] commit $(git rev-parse --short HEAD) gpu 0 :: $name $args" > "logs/stdout/$name.log"
  CUDA_VISIBLE_DEVICES=0 nohup "$MLAVL_PY" ours/train_fisv.py --name "$name" --seed "$seed" $args 2>&1 \
    | grep --line-buffered -v -i warn >> "logs/stdout/$name.log" &
  sleep 20
done
wait
echo "[$(date)] all Phase 3 Fis-V runs finished"
