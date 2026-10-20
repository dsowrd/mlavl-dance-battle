#!/usr/bin/env bash
# Phase 5c: inputs for turn segmentation of the WDSF battle videos (GPU 0).
#   1) clips/       Video Swin K400 + HF AST clip features, the space the BRACE dancing detector
#                   (tools/battle/segment_turns.py) was trained in; 3 shards in parallel
#   2) appearance/  dancer crop (Faster R-CNN person, large-and-central) CLIP ViT-B/16 embedding per clip centre
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
V=$MLAVL_DATA/videos/wdsf; F=$MLAVL_DATA/features/WDSF
# (ids such as -JT8WXpilmU cannot be passed through --ids: both tools take every finished video by default)
for K in 0 1 2; do
  $MLAVL_PY -I tools/battle/extract_features.py --videos $V --out $F --shard $K --nshards 3 &
done
wait
echo "[$(date)] clips: $(ls $F/clips | grep -c 'npz$')"
$MLAVL_PY -I tools/battle/appearance.py detect --videos $V --features $F
echo "[$(date)] appearance: $(ls $F/appearance | grep -c 'npz$')"
