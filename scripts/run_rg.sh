#!/usr/bin/env bash
# Phase 1: RG reproduction with the paper's settings (MLAVL Sec. 4.1 + supplement).
#   source scripts/env.sh
#   bash scripts/run_rg.sh Ball 0            # <apparatus> <gpu id: 0|1|3> [extra args...]
#   bash scripts/run_rg.sh Hoop 1 --lr 1e-2  # e.g. README lr for the comparison run
#
# Paper: lr 1e-4, dropout 0.3, lambda(TL/CE/MSE) = 1.0/0.1/10, xi = 25, batch 64, Adam wd 1e-4, cosine decay 0.01
# epochs: Ball 620, Clubs 340, Hoop 450, Ribbon 710
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

APP=${1:?apparatus: Ball|Clubs|Hoop|Ribbon}
GPU=${2:?gpu id (0, 1 or 3)}
shift 2
[[ "$GPU" == "2" ]] && { echo "GPU 2 is banned"; exit 1; }

declare -A EPOCHS=([Ball]=620 [Clubs]=340 [Hoop]=450 [Ribbon]=710)
FEAT=${MLAVL_DATA}/features/RG
# Verified 2026-10-11 with scripts/inspect_features.py: PAMFN {App}_rgb_VST.npy is (T, 1024), audio AST (T, 768),
# T in [68, 77], 200 train / 50 test videos per apparatus. README's --in_dim 768 does NOT fit these files.
IN_DIM=${IN_DIM:-1024}
SEED=${SEED:-0}            # e.g. SEED=1 bash scripts/run_rg.sh Ball 3
NAME=rg_${APP}_s${SEED}_$(date +%m%d_%H%M)
mkdir -p logs/stdout

CMD=(env CUDA_VISIBLE_DEVICES=$GPU "$MLAVL_PY" main.py
  --dataset rg --action-type "$APP" --score-type Total_Score
  --video-path "$FEAT" --audio-path "$FEAT"
  --train-label-path "$FEAT/train.txt" --test-label-path "$FEAT/test.txt"
  --action_list ./action-label_RG.csv --text_feature ./RG_text_feature.npy --n_text 32
  --clip-num 68 --in_dim "$IN_DIM" --hidden_dim 256 --n_head 8 --n_encoder 1 --n_decoder 3 --n_query 6
  --lr 1e-4 --epoch "${EPOCHS[$APP]}" --batch 64 --optim adam --weight-decay 1e-4 --lr-decay cos --decay-rate 0.01
  --dropout 0.3 --alpha 1.0 --alpha_ce 0.1 --margin 1.0 --seed "$SEED"
  --model-name "$NAME" "$@")

echo "[$(date)] commit $(git rev-parse --short HEAD) gpu $GPU :: ${CMD[*]}" | tee -a logs/stdout/${NAME}.log
/usr/bin/time -v "${CMD[@]}" 2>&1 | tee -a logs/stdout/${NAME}.log
