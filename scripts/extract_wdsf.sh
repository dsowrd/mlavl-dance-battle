#!/usr/bin/env bash
# Phase 5c: PAMFN-style clip features (Video Swin-B K600 + AST, round-12 audio setting) for the WDSF battle videos,
# GPU 0, 2 shards. Re-runs while the download (tools/battle/download_brace_videos.sh, 4 shards) is still going,
# until every video in video_ids.csv that downloaded has its clips_pamfn/{id}.npz.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
V=$MLAVL_DATA/videos/wdsf
F=$MLAVL_DATA/features/WDSF/clips_pamfn
N=$(($(wc -l < $MLAVL_DATA/datasets/wdsf/video_ids.csv) - 1))
while true; do
  downloading=$(pgrep -fc "yt_dlp.*videos/wdsf")
  for K in 0 1; do
    $MLAVL_PY -I tools/battle/extract_features_pamfn.py extract --set wdsf --tm_dim 64 --audio_sr 32000 \
      --fbank_sr 32000 --shard $K --nshards 2 &
  done
  wait
  done_n=$(ls $F 2>/dev/null | grep -c '\.npz$')
  echo "[$(date)] extracted $done_n / $N (downloaders running: $downloading)"
  [ "$downloading" -eq 0 ] && break
  sleep 60
done
echo '[all extracted]'
