#!/usr/bin/env bash
# Phase 5b step 2: battle features from the fine-tuned backbones (GPU 0, 4 shards per set).
#   clips_ft   = dancer-crop Video Swin K600 fine-tuned on BRACE moves (ckpt/vst_move.pt) + PAMFN-matched AST audio
#   clips_ftva = clips_ft video + AST fine-tuned on beat position (ckpt/ast_beat_b6.pt)
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
for S in brace extra; do
  for K in 0 1 2 3; do
    $MLAVL_PY -I tools/battle/finetune_vst.py extract --name vst_move --set $S --shard $K --nshards 4 &
  done
  wait
done
for S in brace extra; do
  $MLAVL_PY -I tools/battle/finetune_ast.py extract --name ast_beat_b6 --set $S --src clips_ft --out clips_ftva
done
echo '[all extracted]'
