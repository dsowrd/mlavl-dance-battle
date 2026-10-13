#!/usr/bin/env bash
# Download the 81 BRACE (Moltisanti et al., ECCV 2022) Red Bull BC One videos from YouTube for research use only.
# Videos are NOT redistributed and never committed (S-Lab License 1.0, non-commercial; YouTube ToS).
#
#   bash tools/battle/download_brace_videos.sh            # all videos in videos_info.csv
#   bash tools/battle/download_brace_videos.sh 3rIk56dcBTM # only these ids
#
# Output: $MLAVL_DATA/videos/brace/{video_id}.mp4 (1080p, BRACE keypoints are in 1920x1080 pixel coordinates).
# Codec: H.264 preferred, then VP9; AV1 is excluded because the server ffmpeg 4.4 has no software AV1 decoder.
# Frame rate is whatever YouTube serves (25 or 29.97 fps) and matches the fps column of BRACE videos_info.csv.
# Resumable: finished ids are listed in downloaded.txt; failures go to failed.txt.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
INFO=${MLAVL_DATA}/datasets/brace/repo/videos_info.csv
OUT=${MLAVL_DATA}/videos/brace
mkdir -p "$OUT"
if [ $# -gt 0 ]; then IDS=("$@"); else mapfile -t IDS < <(awk -F, 'NR>1{print $1}' "$INFO"); fi
echo "[$(date)] ${#IDS[@]} videos -> $OUT"
for id in "${IDS[@]}"; do
  "$MLAVL_PY" -m yt_dlp --no-progress --ffmpeg-location /home/irl/anaconda3/bin \
    -f "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080][vcodec^=vp9]+ba[ext=m4a]/bv*[height<=1080][vcodec!^=av01]+ba/b[height<=1080]" \
    --merge-output-format mp4 --download-archive "$OUT/downloaded.txt" \
    -o "$OUT/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=$id" \
    || { echo "$id" >> "$OUT/failed.txt"; echo "FAILED $id"; }
  sleep 2
done
echo "[$(date)] done: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l) mp4 files, $(sort -u "$OUT/failed.txt" 2>/dev/null | wc -l) failed ids"
