#!/usr/bin/env bash
# Phase 4: clip features (Video Swin-B 1024-d + AST 768-d) for all BRACE videos.
#   bash scripts/extract_brace.sh 3        # <gpu id: 0|1|3>; ~25 s per video, ~7 GiB VRAM, resumable
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
GPU=${1:?gpu id (0, 1 or 3)}; shift
[[ "$GPU" == "2" ]] && { echo "GPU 2 is banned"; exit 1; }
export TORCH_HOME=${MLAVL_DATA}/cache/torch HF_HOME=${MLAVL_DATA}/cache/hf   # keep model weights off the root SSD
mkdir -p logs/stdout
LOG=logs/stdout/extract_brace_$(date +%m%d_%H%M).log
echo "[$(date)] commit $(git rev-parse --short HEAD) gpu $GPU" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU /usr/bin/time -v "$MLAVL_PY" -I tools/battle/extract_features.py \
  --videos "$MLAVL_DATA/videos/brace" --out "$MLAVL_DATA/features/BRACE" "$@" 2>&1 | grep --line-buffered -v -i 'warn\|waveform = ' | tee -a "$LOG"
