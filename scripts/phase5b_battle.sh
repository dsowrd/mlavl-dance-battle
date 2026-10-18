#!/usr/bin/env bash
# Phase 5b step 2: battle runs on the features from the fine-tuned backbones (waits for scripts/extract_ft.sh).
# Same setting as pamfn_all / d0_extra08_all: RG Ribbon s2 init, draft0 vocabulary, BRACE + consistent extra, test on all 170.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "all extracted" logs/feat_ft.log 2>/dev/null; do sleep 300; done
COMMON="--init ckpt/rg_Ribbon_s2_1011_1935_best.pkl --action_list ours/vocab/action-label_BREAKING_draft0.csv --text_feature ours/vocab/BREAKING_draft0_text_feature.npy --extra --extra_min_consistency 0.8 --test_on all"
bash scripts/run_battle_parallel.sh ft_all $COMMON --clips clips_ft
bash scripts/run_battle_parallel.sh ftva_all $COMMON --clips clips_ftva
bash scripts/run_battle_parallel.sh ftva_lt_all $COMMON --clips clips_ftva --learn_text
