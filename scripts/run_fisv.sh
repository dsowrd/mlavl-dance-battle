#!/usr/bin/env bash
# Phase 2: Fis-V reproduction with the paper's settings (MLAVL Sec. 4.1 + supplement).
#   source scripts/env.sh
#   bash scripts/run_fisv.sh TES 1            # <score: TES|PCS> <gpu id: 0|1|3> [extra args...]
#
# Paper: lr 3e-4, dropout 0.7, lambda(TL/CE/MSE) = 0.5/0.1/10, xi = TES 45 / PCS 40, batch 64, Adam wd 1e-4, cosine 0.01
# epochs: TES 420, PCS 440.  clip-num 124 (= shortest video in PAMFN FISV_rgb_VST.npy; T ranges 124..302)
# Verified 2026-10-11: FISV_rgb_VST.npy is (T, 1024) -> --in_dim 1024; FISV_audio_AST.npy (T, 768). 400 train / 100 test.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

SCORE=${1:?score: TES|PCS}
GPU=${2:?gpu id (0, 1 or 3)}
shift 2
[[ "$GPU" == "2" ]] && { echo "GPU 2 is banned"; exit 1; }

declare -A EPOCHS=([TES]=420 [PCS]=440)
FEAT=${MLAVL_DATA}/features/FISV
IN_DIM=${IN_DIM:-1024}
SEED=${SEED:-0}            # e.g. SEED=1 bash scripts/run_rg.sh Ball 3
NAME=fisv_${SCORE}_s${SEED}_$(date +%m%d_%H%M)
mkdir -p logs/stdout

CMD=(env CUDA_VISIBLE_DEVICES=$GPU "$MLAVL_PY" main.py
  --dataset fisv --action-type "$SCORE"
  --video-path "$FEAT" --audio-path "$FEAT"
  --train-label-path "$FEAT/train.txt" --test-label-path "$FEAT/test.txt"
  --action_list ./action-label_FS.csv --text_feature ./FS_text_feature.npy --n_text 42
  --clip-num 124 --in_dim "$IN_DIM" --hidden_dim 256 --n_head 8 --n_encoder 1 --n_decoder 3 --n_query 6
  --lr 3e-4 --epoch "${EPOCHS[$SCORE]}" --batch 64 --optim adam --weight-decay 1e-4 --lr-decay cos --decay-rate 0.01
  --dropout 0.7 --alpha 0.5 --alpha_ce 0.1 --margin 1.0 --seed "$SEED"
  --model-name "$NAME" "$@")

echo "[$(date)] commit $(git rev-parse --short HEAD) gpu $GPU :: ${CMD[*]}" | tee -a logs/stdout/${NAME}.log
/usr/bin/time -v "${CMD[@]}" 2>&1 | tee -a logs/stdout/${NAME}.log
