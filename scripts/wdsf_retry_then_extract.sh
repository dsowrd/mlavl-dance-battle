#!/usr/bin/env bash
# Phase 5c follow-up: after the 4 download shards end, retry the failed WDSF ids once (YouTube 403s are usually
# transient), wait for scripts/extract_wdsf.sh to finish, then run one more extraction pass for the retried videos.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
V=$MLAVL_DATA/videos/wdsf; W=$MLAVL_DATA/datasets/wdsf
while pgrep -f "python -m yt_dlp.*videos/wdsf" >/dev/null || pgrep -f "[d]ownload_brace_videos.sh" >/dev/null; do sleep 60; done
sort -u $V/failed.txt | grep -v -x -F -f <(awk '{print $2}' $V/downloaded.txt) > $W/retry_ids.txt
echo "[$(date)] retry $(wc -l < $W/retry_ids.txt) ids"
mv $V/failed.txt $V/failed_first_pass.txt
(echo video_id; cat $W/retry_ids.txt) > $W/retry.csv
INFO=$W/retry.csv OUT=$V HEIGHT=720 bash tools/battle/download_brace_videos.sh
while pgrep -f "[e]xtract_wdsf.sh" >/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
$MLAVL_PY -I tools/battle/extract_features_pamfn.py extract --set wdsf --tm_dim 64 --audio_sr 32000 --fbank_sr 32000
echo "[$(date)] done: $(ls $MLAVL_DATA/features/WDSF/clips_pamfn | grep -c 'npz$') extracted, $(grep -c . $V/downloaded.txt) downloaded"
