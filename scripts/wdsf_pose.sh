#!/usr/bin/env bash
# Phase 5c: appearance-free pose input for the WDSF turns (GPU 0). Keypoint R-CNN every 3rd frame of the turns
# (3 shards), then per-clip pose + kick-beat features with the BRACE-trained dancer selector.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$MLAVL_PY -I tools/battle/wdsf_pose.py turns
for K in 0 1 2; do
  $MLAVL_PY -I tools/battle/estimate_pose.py --set wdsf --stride 3 --shard $K --nshards 3 &
done
wait
echo "[$(date)] pose_est: $(ls $MLAVL_DATA/features/WDSF/pose_est | grep -c 'npz$')"
$MLAVL_PY -I tools/battle/wdsf_pose.py features
echo "[$(date)] pose features: $(ls $MLAVL_DATA/features/WDSF/pose | grep -c 'npz$')"
