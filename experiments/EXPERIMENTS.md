# Experiment log

One line per run. Fill in every column; if a value is unknown write `?` and fix it later.
Numbers that differ from the paper are recorded as they are, with a guess at the cause in *Notes*.

Columns: Date · Commit · Dataset/score · Key args · GPU · Wall time · Max VRAM · best ρ (epoch) · best MSE (epoch) · Notes

## Paper targets (MLAVL, CVPR 2025)

| Data | ρ | MSE |
| --- | --- | --- |
| RG | Ball 0.826 / Clubs 0.829 / Hoop 0.871 / Ribbon 0.866 / Avg 0.849 | 5.57 / 4.20 / 4.11 / 3.99 / Avg 4.47 |
| FS1000 | TES 0.92 / PCS 0.89 / SS 0.90 / TR 0.90 / PE 0.88 / CO 0.89 / IN 0.88 / Avg 0.90 | 64.89 / 6.39 / 0.23 / 0.24 / 0.50 / 0.25 / 0.26 / Avg 10.39 |
| Fis-V | TES 0.766 / PCS 0.863 / Avg 0.823 | 19.44 / 7.17 / 13.31 |

Average ρ: report both arithmetic mean and Fisher-z mean.

## Phase 0 · setup

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Phase 1 · RG reproduction

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-11 | d092717 | RG / Ball | `scripts/run_rg.sh Ball 0` (paper: lr 1e-4, dropout 0.3, α 1.0, α_ce 0.1, clip 68, in_dim 1024, 620 ep) | 0 (4 runs concurrently) | 1:32:54 | ~1.8 GB (RSS 1.6 GB) | 0.754 (129), MSE at that ckpt 7.01 | 5.42 (145), ρ at that ckpt 0.753 | paper 0.826 / 5.57. Train ρ 0.99 from ep ~300: overfits, test ρ oscillates 0.66–0.72 after ep 200 |
| 2026-10-11 | d092717 | RG / Clubs | `scripts/run_rg.sh Clubs 0` (340 ep) | 0 (4 concurrent) | 0:52:58 | ~1.8 GB (RSS 1.6 GB) | 0.785 (226), MSE 5.29 | 4.47 (225), ρ 0.779 | paper 0.829 / 4.20 |
| 2026-10-11 | d092717 | RG / Hoop | `scripts/run_rg.sh Hoop 0` (450 ep) | 0 (4 concurrent) | 1:07:27 | ~1.8 GB (RSS 1.7 GB) | 0.780 (163), MSE 6.39 | 4.68 (288), ρ 0.766 | paper 0.871 / 4.11. Largest gap (−0.09) |
| 2026-10-11 | d092717 | RG / Ribbon | `scripts/run_rg.sh Ribbon 0` (710 ep) | 0 (4 concurrent) | 1:41:04 | ~1.8 GB (RSS 1.6 GB) | 0.792 (355), MSE 5.39 | 5.36 (354), ρ 0.789 | paper 0.866 / 3.99 |
| 2026-10-11 | d092717 | **RG / Avg** | paper settings, seed 0, 1 run each | 0 | 4 runs in 1:41 wall | | **0.778** (arithmetic) / **0.778** (Fisher z) | **4.98** (mean of best-MSE ckpts); 6.02 at the best-ρ ckpts | paper 0.849 / 4.47 → ρ −0.071, MSE +0.51. Logs `logs/stdout/rg_*_1011_1548.log`, ckpts backed up to `/data/mlavl/ckpt_backup/`. Suspected causes below |
| 2026-10-11 | 394eed5 | RG / Ball,Clubs,Hoop,Ribbon | as above + `--lr 1e-2` (README lr) | 0 (4 concurrent) | 17:31–17:50, **stopped at 72–80 of 340–710 epochs** | ~1.8 GB | partial: Ball 0.255 / Clubs 0.385 / Hoop 0.412 / Ribbon 0.401 (test ρ oscillates around 0, often negative) | – | README lr 1e-2 does not train with the paper's other settings: at the same epoch lr 1e-4 reaches 0.69–0.78. Stopped early by 정현도 to free GPU 0. Conclusion: use the paper lr; README value is wrong or assumed a different setup. Logs `logs/stdout/rg_*_1011_1731.log` |
| 2026-10-11 | 1c0f97c | RG / Ball s1 | paper settings, `SEED=1` | 3 (4 concurrent) | 1:51:08 | ~1.8 GB | 0.768 (37), MSE 8.39 | 5.43 (87), ρ 0.763 | |
| 2026-10-11 | 2a2a346 | RG / Ball s2 | `SEED=2` | 3 | 1:28:42 | ~1.8 GB | 0.751 (42), MSE 12.02 | 5.72 (185), ρ 0.746 | |
| 2026-10-11 | 1c0f97c | RG / Clubs s1 | `SEED=1` | 3 | 1:00:17 | ~1.8 GB | 0.758 (128), MSE 5.42 | 5.09 (156), ρ 0.735 | |
| 2026-10-11 | 2a2a346 | RG / Clubs s2 | `SEED=2` | 3 | 0:59:28 | ~1.8 GB | 0.704 (229), MSE 6.51 | 5.69 (179), ρ 0.698 | worst RG run |
| 2026-10-11 | 1c0f97c | RG / Hoop s1 | `SEED=1` | 3 | 1:18:03 | ~1.8 GB | 0.824 (45), MSE 9.44 | 4.29 (151), ρ 0.803 | |
| 2026-10-11 | 2a2a346 | RG / Hoop s2 | `SEED=2` | 3 | 1:11:19 | ~1.8 GB | 0.755 (305), MSE 5.30 | 5.22 (307), ρ 0.747 | |
| 2026-10-11 | 1c0f97c | RG / Ribbon s1 | `SEED=1` | 3 | 1:59:03 | ~1.8 GB | 0.789 (255), MSE 5.99 | 5.44 (222), ρ 0.779 | |
| 2026-10-11 | 2a2a346 | RG / Ribbon s2 | `SEED=2` | 3 | 1:34:40 | ~1.8 GB | 0.842 (561), MSE 4.66 | 4.36 (380), ρ 0.839 | best RG run |

**RG, 3 seeds (0, 1, 2), paper settings.** `scripts/summarize_runs.py --seeds logs/stdout/rg_*_1011_1548.log logs/stdout/rg_*_s[12]_*.log`

| Apparatus | paper ρ | ours ρ, mean ± std | paper MSE | ours best MSE, mean ± std | ours MSE at best-ρ ckpt |
| --- | --- | --- | --- | --- | --- |
| Ball | 0.826 | 0.758 ± 0.009 | 5.57 | 5.52 ± 0.17 | 9.14 ± 2.59 |
| Clubs | 0.829 | 0.749 ± 0.041 | 4.20 | 5.08 ± 0.61 | 5.74 ± 0.67 |
| Hoop | 0.871 | 0.786 ± 0.035 | 4.11 | 4.73 ± 0.47 | 7.04 ± 2.14 |
| Ribbon | 0.866 | 0.808 ± 0.030 | 3.99 | 5.05 ± 0.60 | 5.35 ± 0.67 |
| **Avg** | **0.849** | **0.775** (Fisher z 0.776) | **4.47** | **5.10** | 6.82 |

- Seed spread is large: ±0.03–0.04 ρ for Clubs/Hoop/Ribbon, single runs range 0.704 (Clubs s2) to 0.842 (Ribbon s2). The paper's per-apparatus numbers are within about 1–2 std of our best single runs for Hoop/Ribbon but not for Ball/Clubs.
- Seed 0 alone gave 0.778; the 3-seed mean 0.775 is essentially the same, so the −0.07 gap to the paper is not a seed artefact.
- Best-ρ epochs are often very early (37–45 for Ball s1/s2, Hoop s1) with a high MSE at that checkpoint (8–12): ρ and MSE optima do not coincide; the paper probably reports ρ and MSE from separate checkpoints too (the code tracks both), but note the early peaks are noisy.
- Per-run ckpts (`_best`, `_best_mse`) backed up to `/data/mlavl/ckpt_backup/`.

Suspected causes of the RG gap (to check in order):

1. **Features differ from the authors'.** MLAVL's README uses `--in_dim 768` for RG, but the only public RG features (PAMFN) are 1024-d Video Swin. The authors probably extracted their own 768-d RGB features (and possibly their own audio features); we cannot match them. This alone could explain most of the gap.
2. **Over-fitting on 200 training videos.** Train ρ reaches 0.99 while test ρ plateaus at ep 150–350; the best epochs (129–355) are well before the paper's epoch counts (340–710). Paper numbers are probably best-epoch numbers as well (the code saves `_best.pkl`), so this is not a reporting difference, but a stronger regulariser (dropout 0.5, weight decay) or fewer epochs may help.
3. **Single seed (0).** Test set is 50 videos per apparatus; ρ varies by ±0.03 between neighbouring epochs. The paper may report the best of several seeds. Repeat with seeds 1, 2 during the 10/26–10/31 trip.
4. **Trainable params 6.72M vs paper 3.82M**: the released code is not exactly the paper's configuration (to be itemised per module).
5. **README vs paper hyper-parameters**: README says lr 1e-2 and epochs 250/400/500/150 (Ball/Clubs/Hoop/Ribbon), the paper says lr 1e-4 and 620/340/450/710. The lr 1e-2 run above tests the first; if it is better, the README is what the authors actually ran.

## Phase 2 · Fis-V reproduction (FS1000 if the features are obtained)

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-11 | 93a141b | Fis-V / TES | `scripts/run_fisv.sh TES 1` (paper: lr 3e-4, dropout 0.7, α 0.5, α_ce 0.1, clip 124, in_dim 1024, 420 ep, seed 0) | 1 (2 concurrent) | 3:36:15 | ~2.7 GB (RSS 2.0 GB) | 0.723 (56), MSE at that ckpt 25.73 | 20.65 (165), ρ at that ckpt 0.667 | paper 0.766 / 19.44. Best ρ at ep 56 of 420, afterwards 0.63–0.67 (train ρ 0.99): over-fits early |
| 2026-10-11 | 93a141b | Fis-V / PCS | `scripts/run_fisv.sh PCS 1` (440 ep, seed 0) | 1 (2 concurrent) | 3:45:42 | ~2.7 GB (RSS 1.9 GB) | 0.852 (48), MSE 8.00 | 8.00 (48), ρ 0.852 | paper 0.863 / 7.17. Close to the paper; best at ep 48, then 0.78–0.83 |
| 2026-10-11 | 93a141b | **Fis-V / Avg** | seed 0 | 1 | | | **0.787** (arithmetic) / **0.796** (Fisher z) | **14.33** (mean of best-MSE ckpts); 16.86 at the best-ρ ckpts | paper 0.823 / 13.31 → ρ −0.036, MSE +1.0. Logs `logs/stdout/fisv_*_1011_1554.log`, ckpts in `/data/mlavl/ckpt_backup/`. Fis-V gap is smaller than RG's (−0.071): Fis-V has 400 training videos vs 200, and the feature question (in_dim 768 vs 1024) applies here too since the README also says 768 |
| 2026-10-11 | f6fc37b | Fis-V / TES s1 | paper settings, `SEED=1` | 0 (2 concurrent) | 3:28:04 | ~2.7 GB (RSS 2.1 GB) | 0.736 (16), MSE 36.59 | 21.34 (63), ρ 0.684 | best ρ at epoch 16 (!), very noisy early peak; logs `logs/stdout/fisv_*_s1_1011_1747.log` |
| 2026-10-11 | f6fc37b | Fis-V / PCS s1 | `SEED=1` | 0 (2 concurrent) | 3:36:22 | ~2.7 GB (RSS 2.0 GB) | 0.847 (65), MSE 10.59 | 8.29 (64), ρ 0.826 | |
| 2026-10-11 | f9d91f2 | Fis-V / TES s2 | `SEED=2` | 3 (2 concurrent) | 2:57:23 | ~2.7 GB (RSS 1.8 GB) | 0.711 (45), MSE 27.34 | 20.30 (123), ρ 0.683 | started 21:10 after the RG seed runs freed GPU 3; logs `logs/stdout/fisv_*_s2_1011_2110.log` |
| 2026-10-11 | f9d91f2 | Fis-V / PCS s2 | `SEED=2` | 3 (2 concurrent) | 3:04:31 | ~2.7 GB (RSS 1.9 GB) | 0.858 (50), MSE 9.01 | 7.52 (248), ρ 0.834 | best Fis-V PCS run |

**Fis-V, 3 seeds (0, 1, 2), paper settings.** `scripts/summarize_runs.py --seeds logs/stdout/fisv_*_1011_1554.log logs/stdout/fisv_*_s[12]_*.log`

| Score | paper ρ | ours ρ, mean ± std | paper MSE | ours best MSE, mean ± std | ours MSE at best-ρ ckpt |
| --- | --- | --- | --- | --- | --- |
| TES | 0.766 | 0.723 ± 0.013 | 19.44 | 20.77 ± 0.53 | 29.89 ± 5.86 |
| PCS | 0.863 | 0.852 ± 0.006 | 7.17 | 7.93 ± 0.39 | 9.20 ± 1.30 |
| **Avg** | **0.823** | **0.788** (Fisher z 0.797) | **13.31** | **14.35** | 19.54 |

- Seed spread is much smaller than on RG (±0.006–0.013 ρ): 400 training / 100 test videos vs 200 / 50.
- PCS reproduces within 0.011 ρ; TES is 0.043 below the paper in every seed. TES (technical element score) depends on recognising specific jumps/spins, which is where the authors' text-guided features would matter most and where a feature mismatch (README `--in_dim 768` vs public 1024-d VST) would hurt most.
- Best-ρ epochs are very early (16–65 of 420/440) and the MSE at those checkpoints is poor (TES 26–37); after epoch ~100 the model has train ρ 0.99 and test ρ settles 0.03–0.08 below the early peak. Fewer epochs or stronger regularisation is the first thing to try in Phase 3; the early peaks also mean the "best ρ" protocol is optimistic and noisy.
- All `_best` / `_best_mse` ckpts backed up to `/data/mlavl/ckpt_backup/` (975 MB total so far).

## Phase 3 · improvements (public features only)

### 3-1 · dropout (regularisation), RG

Motivation: every reproduction run over-fits early (train ρ 0.99; best test ρ at epoch 16–355 of 340–710). `--dropout` is the dropout inside the Transformer layers (S²CE encoder, AVCF cross-temporal decoder, DPG grading decoder; attention + FFN). Paper value 0.3 (RG). Baseline (3 seeds): ρ 0.775 / MSE 5.10.

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12 | 585983c | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--dropout 0.5` (`scripts/phase3_dropout.sh`) | 0/1/3 = seed 0/1/2; 4 concurrent per GPU | 12:04–14:21, 1:07–2:17 per run | ~1.8 GB | avg 0.776 (see table) | avg 5.33 | 12 runs, logs `logs/stdout/rg_*_s?_dp0.5_1012_1204.log`. Final train ρ still 0.98–0.99 |
| 2026-10-12 | 9b9d21c | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--dropout 0.7` | 0/1/3 = seed 0/1/2 | 13:11–15:59, 1:07–1:41 per run | ~1.8 GB | avg 0.751 | avg 6.40 | 12 runs, logs `logs/stdout/rg_*_s?_dp0.7_1012_*.log`. Final train ρ 0.93–0.96. Requested by 정현도 ("dropout 올려서 Phase 3 첫 실험 돌려") |

**Result, 3 seeds each (mean ± std of per-apparatus best test ρ / best test MSE):**

| Apparatus | dropout 0.3 (paper, baseline) | dropout 0.5 | dropout 0.7 |
| --- | --- | --- | --- |
| Ball | 0.758 ± 0.009 / 5.52 ± 0.17 | 0.771 ± 0.016 / 5.93 ± 0.73 | 0.735 ± 0.021 / 7.39 ± 0.36 |
| Clubs | 0.749 ± 0.041 / 5.08 ± 0.61 | 0.745 ± 0.024 / 5.24 ± 0.43 | 0.722 ± 0.037 / 6.83 ± 0.85 |
| Hoop | 0.786 ± 0.035 / 4.73 ± 0.47 | 0.792 ± 0.006 / 4.79 ± 0.28 | 0.785 ± 0.008 / 5.18 ± 0.10 |
| Ribbon | 0.808 ± 0.030 / 5.05 ± 0.60 | 0.798 ± 0.030 / 5.37 ± 0.63 | 0.763 ± 0.031 / 6.20 ± 0.80 |
| **Avg** | **0.775 / 5.10** | **0.776 / 5.33** | **0.751 / 6.40** |
| final train ρ | 0.99 | 0.98–0.99 | 0.93–0.96 |

Conclusion: **Transformer dropout is not the lever.** 0.5 is indistinguishable from the baseline (ρ +0.001, MSE +0.23, all within seed std); 0.7 hurts (ρ −0.024, MSE +1.3) and only then does the train ρ drop below 0.99. The model memorises the 200 training videos through something dropout inside the Transformer layers does not touch (frozen 1024-d clip features → 1×1 Conv projections, the graph module, the grading queries). Next regularisation candidates, in order: (a) fewer epochs / early-stopping-style evaluation protocol, (b) weight decay 1e-4 → 1e-3/1e-2, (c) dropout on the input features / after the projection (`PositionalEncoding` has a dropout=0 slot), (d) temporal augmentation (random crop range is only T−68 ≤ 9 clips on RG; try random clip masking or mixup across videos). Keep 0.3 for all later Phase 3 experiments.

### 3-2 · weight decay, RG

Motivation: dropout did nothing (3-1). The optimiser is plain Adam with coupled L2 `weight_decay` (main.py `get_optim`), paper value 1e-4. Try 1e-3 and 1e-2. Baseline (3 seeds): ρ 0.775 / MSE 5.10. Launcher: `scripts/rg_sweep.sh mlavl_p3_wd wd1e-3 "--weight-decay 1e-3" wd1e-2 "--weight-decay 1e-2"`.

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12 | 5f35949 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--weight-decay 1e-3` | 0/1/3 = seed 0/1/2; 4 concurrent per GPU | 18:04–20:22 | ~1.8 GB | avg 0.772 | avg 5.21 | 12 runs, logs `logs/stdout/rg_*_s?_wd1e-3_1012_1804.log`. Final train ρ 0.99 |
| 2026-10-12 | 3e573ea | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--weight-decay 1e-2` | 0/1/3 = seed 0/1/2 | 19:10–22:05 | ~1.8 GB | avg 0.773 | avg 5.30 | 12 runs, logs `logs/stdout/rg_*_s?_wd1e-2_1012_*.log`. Final train ρ 0.98–0.99. Requested by 정현도 ("weight decay 올려서 다음 실험 돌려") |

**Result, 3 seeds each (mean ± std of per-apparatus best test ρ / best test MSE):**

| Apparatus | wd 1e-4 (paper, baseline) | wd 1e-3 | wd 1e-2 |
| --- | --- | --- | --- |
| Ball | 0.758 ± 0.009 / 5.52 ± 0.17 | 0.767 ± 0.010 / 5.55 ± 0.27 | 0.780 ± 0.015 / 5.21 ± 0.25 |
| Clubs | 0.749 ± 0.041 / 5.08 ± 0.61 | 0.748 ± 0.034 / 4.96 ± 0.45 | 0.749 ± 0.028 / 5.03 ± 0.35 |
| Hoop | 0.786 ± 0.035 / 4.73 ± 0.47 | 0.787 ± 0.036 / 4.81 ± 0.45 | 0.795 ± 0.021 / 4.61 ± 0.31 |
| Ribbon | 0.808 ± 0.030 / 5.05 ± 0.60 | 0.784 ± 0.019 / 5.52 ± 0.41 | 0.769 ± 0.024 / 6.34 ± 0.28 |
| **Avg** | **0.775 / 5.10** | **0.772 / 5.21** | **0.773 / 5.30** |
| final train ρ | 0.99 | 0.99 | 0.98–0.99 |

Conclusion: **no effect.** Averages move by ≤0.003 ρ, within seed std, and train ρ stays at 0.99 even at 1e-2 (100× the paper value). Ball/Hoop tick up and Ribbon ticks down with larger decay; these are opposite-signed, apparatus-level noise. Note that this is *coupled* L2 inside Adam, which Adam's per-parameter scaling largely neutralises (the AdamW argument); the code already has `--optim adamw`, so a decoupled-decay test (AdamW, wd 0.05–0.1) is the proper follow-up if weight decay is pursued further. Together with 3-1 the picture is that generic parameter regularisation does not address this over-fitting; the next candidates act on the data/input side (fewer epochs, input-feature dropout, temporal augmentation) or on the evaluation protocol.

### 3-3 · input-side regularisation (ours), RG

Motivation: parameter regularisation (3-1 dropout, 3-2 L2) did nothing; the model memorises the fixed clip features. Two input-side knobs (`feat:` commit, `models/model.py` forward, both off by default):
- `--clip_mask p`: during training each clip is zeroed with probability p, same mask for video and audio (keeps them aligned). RG random-crop gives ≤9 clips of augmentation, so this is the first real temporal augmentation.
- `--input_dropout p`: `nn.Dropout` on the projected 256-d V/A tokens right before the shared S²CE encoder.

Settings: clip_mask 0.1, clip_mask 0.25, input_dropout 0.3, input_dropout 0.5; 3 seeds each (48 runs). Launcher:
`scripts/rg_sweep.sh mlavl_p3_input cm0.1 "--clip_mask 0.1" cm0.25 "--clip_mask 0.25" id0.3 "--input_dropout 0.3" id0.5 "--input_dropout 0.5"`. Baseline (3 seeds): ρ 0.775 / MSE 5.10.

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-13 | bf5dec5 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--clip_mask 0.1` | 0/1/3 = seed 0/1/2; 4 concurrent per GPU | 10:29–11:54 | ~1.8 GB | avg 0.765 | avg 5.33 | 12 runs, logs `logs/stdout/rg_*_s?_cm0.1_1013_1029.log` |
| 2026-10-13 | bf5dec5 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--clip_mask 0.25` | 0/1/3 | ~11:55–13:45 | ~1.8 GB | avg 0.753 | avg 5.60 | 12 runs, logs `logs/stdout/rg_*_s?_cm0.25_1013_*.log` |
| 2026-10-13 | bf5dec5 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--input_dropout 0.3` | 0/1/3 | ~13:45–16:05 | ~1.8 GB | avg 0.774 | avg 5.11 | 12 runs, logs `logs/stdout/rg_*_s?_id0.3_1013_*.log` |
| 2026-10-13 | bf5dec5 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--input_dropout 0.5` | 0/1/3 | ~16:05–18:20 | ~1.8 GB | avg 0.771 | avg 5.17 | 12 runs, logs `logs/stdout/rg_*_s?_id0.5_1013_*.log`. Requested by 정현도 ("1번 가자") |

**Result, 3 seeds each (mean ± std of per-apparatus best test ρ / best test MSE):**

| Apparatus | baseline (paper) | clip_mask 0.1 | clip_mask 0.25 | input_dropout 0.3 | input_dropout 0.5 |
| --- | --- | --- | --- | --- | --- |
| Ball | 0.758 ± 0.009 / 5.52 | 0.758 ± 0.008 / 5.70 | 0.737 ± 0.016 / 6.16 | 0.750 ± 0.006 / 5.74 | 0.757 ± 0.011 / 5.72 |
| Clubs | 0.749 ± 0.041 / 5.08 | 0.739 ± 0.024 / 5.14 | 0.702 ± 0.030 / 5.89 | 0.745 ± 0.030 / 5.07 | 0.738 ± 0.016 / 5.11 |
| Hoop | 0.786 ± 0.035 / 4.73 | 0.789 ± 0.024 / 4.71 | 0.791 ± 0.017 / 4.72 | 0.796 ± 0.025 / 4.69 | 0.794 ± 0.005 / 4.70 |
| Ribbon | 0.808 ± 0.030 / 5.05 | 0.774 ± 0.027 / 5.78 | 0.782 ± 0.035 / 5.62 | 0.804 ± 0.033 / 4.93 | 0.795 ± 0.033 / 5.17 |
| **Avg** | **0.775 / 5.10** | **0.765 / 5.33** | **0.753 / 5.60** | **0.774 / 5.11** | **0.771 / 5.17** |
| final train ρ | 0.99 | 0.99 | 0.99 | 0.99 | 0.98–0.99 |

Conclusion: **no gain from input-side regularisation either.** Input dropout 0.3/0.5 is a wash (−0.001 / −0.004 ρ, within seed std); clip masking *hurts* monotonically (−0.010 at 0.1, −0.022 at 0.25, driven by Ball/Clubs/Ribbon) and train ρ stays 0.99 regardless, so masking 25 % of clips still does not stop memorisation — the model fits through whichever clips remain. Across 3-1/3-2/3-3 (dropout, L2, input dropout, clip masking; 96 runs) nothing moves the RG average by more than seed noise, which strongly suggests the −0.07 gap to the paper is not a regularisation problem but an input problem (the public 1024-d PAMFN features vs whatever 768-d features the authors used). **Phase 3 hyper-parameter sweeps stop here** (정현도, 2026-10-13); the improvement story for the mid-term presentation moves to the task-level changes (multi-task head, high-score weighting) and to the battle domain (Phase 4/5), where the baseline is ours anyway. Keep `--clip_mask 0 --input_dropout 0` (defaults).

### 3-4 · multi-task TES+PCS head and high-score weighting, Fis-V (2026-10-17/18)

Diagnosis first (`tools/score_bias.py` on the reproduction checkpoints): test predictions are compressed toward the mean (slope prediction ~ label 0.51–0.54 for TES, 0.61–0.81 for PCS); the bottom fifth of TES is over-predicted by +8 to +10 points, which dominates the MSE. Trainer `ours/train_fisv.py` (upstream protocol: paper settings, test every epoch, best-ρ / best-MSE epoch per score; + linear calibration fitted on the training-set predictions of the best-ρ checkpoint → "calibrated MSE"). Queue `scripts/phase3_fisv.sh`: 6 settings × seeds 0–2 = 18 runs, all on GPU 0, 6 at a time, 2026-10-17 19:37 → 2026-10-18 ~21:00 (GPU 0 shared with the audio re-extraction and the battle runs), ~1.5 GB each.

| setting | score | best ρ (mean ± std) | MSE at best ρ | calibrated MSE | best MSE | slope | last-epoch ρ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single-task (control, `st`) | TES | 0.716 ± 0.008 | 31.95 | 20.48 | 20.57 | 0.45 | 0.617 |
| multi-task (`mt`) | TES | **0.752 ± 0.007** | 40.34 | 22.17 | 19.01 | 0.45 | 0.629 |
| LDS (`lds`) | TES | 0.701 ± 0.011 | 59.77 | 23.96 | 20.55 | 0.85 | 0.631 |
| multi-task + LDS (`mtlds`) | TES | 0.739 ± 0.017 | 32.95 | **19.13** | **18.54** | 0.81 | 0.631 |
| single-task (control, `st`) | PCS | 0.846 ± 0.003 | 9.10 | 7.92 | 7.96 | 0.63 | 0.802 |
| multi-task (`mt`) | PCS | 0.832 ± 0.006 | 8.92 | 7.53 | 7.59 | 0.64 | 0.797 |
| LDS (`lds`) | PCS | **0.849 ± 0.005** | 8.56 | **7.01** | **6.73** | 0.72 | 0.809 |
| multi-task + LDS (`mtlds`) | PCS | 0.844 ± 0.006 | 8.04 | 7.51 | 7.01 | 0.72 | 0.796 |
| paper (Table 3) | TES / PCS | 0.766 / 0.863 | | | 19.44 / 7.17 | | |
| our reproduction (upstream `main.py`, 3 seeds) | TES / PCS | 0.723 / 0.852 | | | 20.77 / 7.93 | | |

Reading:
- **Multi-task helps TES ranking: +0.036 ρ over the control (≈ 5 seed std), 0.752 vs paper 0.766**, the closest we have come. Predicting PCS from the same grade embeddings regularises the TES ranking. It costs PCS −0.014 (the PCS head is a second regressor on embeddings shaped by the TES loss).
- **LDS fixes the compression** (slope 0.45 → 0.81–0.85 on TES, 0.63 → 0.72 on PCS) and lowers the MSE: PCS best MSE 6.73 and TES 18.54 (mtlds) are below the paper's 7.17 / 19.44. It does not improve ranking on its own (TES −0.015).
- Post-hoc calibration alone gives most of the MSE gain for the unweighted models (TES 31.95 → 20.48), so part of the MSE gap is a scale problem, not a ranking problem.
- Best per score: TES ρ from `mt`, PCS ρ and MSE from `lds`; `mtlds` is the best single compromise (ρ 0.739 / 0.844, avg 0.792 vs reproduction 0.788, best MSE 18.54 / 7.01).
- Caveat (as for the reproduction): best-ρ / best-MSE epochs are chosen on the test set, as in the upstream protocol; last-epoch numbers (0.63 / 0.80) are much lower and show the same ordering is not guaranteed without early stopping.


## Phase 4/6 · battle winner prediction (BRACE, 81 battles)

Labels: `ours/battle/winners.csv` (Wikipedia brackets, 31/81 checked by 정현도 against the videos, all correct).
Metric: pairwise accuracy over the 81 battles, 95% bootstrap CI (10,000 resamples of battles). A = dancer who danced first.

### Training-free baselines (2026-10-13, `tools/battle/baselines.py`)

Pose statistics from the BRACE keypoints, aggregated over all of a dancer's segments in the battle; prediction = the dancer with the larger value wins.

| baseline | accuracy | 95% CI | 2011 | 2013 | 2014 | 2017 | 2018 | 2020 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random (expected) | 50.0% | | | | | | | |
| first dancer wins | 59.3% | 48.1–70.4% | 38% | 60% | 50% | 60% | 60% | 79% |
| more dancing time | 50.6% | 39.5–61.7% | 50% | 53% | 50% | 47% | 47% | 57% |
| higher mean joint speed (energy) | 58.0% | 46.9–69.1% | 75% | 67% | 29% | 73% | 53% | 57% |
| higher 90th-pct joint speed | 56.8% | 45.7–67.9% | 75% | 67% | 36% | 80% | 40% | 50% |
| larger powermove share | 59.3% | 48.1–70.4% | 88% | 47% | 50% | 67% | 47% | 71% |

- Every rule's CI contains 50%: with 81 battles, anything below ~61% is not distinguishable from chance. A learned model needs ≥ ~65% to be clearly better than the first-dancer rule.
- Winners and losers are almost identical on these statistics (mean energy 0.0476 vs 0.0464, powermove share 0.39 vs 0.37, dancing time 78.6 s vs 78.9 s): judges are not rewarding "more" or "faster", which is the motivation for MLAVL-style assessment (musicality, vocabulary) rather than motion energy.
- Year-to-year swings (energy rule 29% in 2014, 75% in 2011) are large because a year has only 8–15 battles; leave-one-year-out results must be pooled over all 81 battles, never reported per fold alone.

### Siamese MLAVL, first run (2026-10-13, `scripts/run_battle.sh`, commit 43f98ef)

P(A wins) = σ(k·(s_A − s_B)), BCE + 0.1 × MLAVL aux losses, clip_num 64, 40 epochs, lr 1e-4, batch 16, random A/B swap, leave-one-year-out (6 folds), model after the last epoch (no selection on the test year). Breaking vocabulary = unreviewed draft, vanilla-CLIP text features. GPU 0/1/3, ~35 min per seed (6 folds), ~2 GB VRAM.

| init | acc (mean ± std over 3 seeds) | AUC | log loss | seed-ensemble acc (95% CI) | seed agreement |
| --- | --- | --- | --- | --- | --- |
| scratch | 55.6% ± 2.0 | 0.544 | 1.058 | 58.0% (46.9–69.1) | 0.71 |
| Fis-V PCS ckpt | 54.7% ± 1.5 | 0.569 | 0.924 | 55.6% (44.4–65.4) | 0.91 |
| RG Ribbon ckpt | **60.9% ± 1.2** | **0.605** | 0.987 | **61.7% (50.6–72.8)** | 0.84 |
| *first dancer wins (rule)* | *59.3%* | | | *(48.1–70.4)* | |
| *higher joint-speed energy (rule)* | *58.0%* | | | *(46.9–69.1)* | |

- Best setting (RG init) is the first one whose CI lower bound clears 50%, but it is not distinguishable from the first-dancer rule (59.3%); none of the three is a demonstrated improvement over the training-free baselines.
- Initialising from an AQA model helps only from RG (rhythmic gymnastics: whole-body motion to music), not from Fis-V PCS; scratch is noisiest (seed agreement 0.71).
- Over-confidence / over-fitting: final train loss 0.14–0.42 while test log loss is 0.92–1.06 (> 0.693 of a coin flip). The model memorises ~70 training battles in 40 epochs.
- 2018 is the hardest year in every setting (33–60%).
- Next: fewer epochs / smaller lr for the init runs, freeze the backbone part (S²CE + MAG²) and train only the grading head, add the first-dancer prior as a learned bias (the model is order-invariant by design), and the reviewed vocabulary.

### Siamese MLAVL, round 2: less fitting (2026-10-13/14, RG Ribbon init, 3 seeds each)

| setting | acc (mean ± std) | AUC | log loss | seed-ensemble acc (95% CI) | final train loss |
| --- | --- | --- | --- | --- | --- |
| RG init, 40 ep, lr 1e-4 (round 1) | **60.9% ± 1.2** | **0.605** | 0.987 | **61.7% (50.6–72.8)** | 0.14–0.42 |
| 10 epochs | 56.0% ± 2.3 | 0.580 | 0.845 | 58.0% (46.9–69.1) | 0.53–0.76 |
| 20 epochs, lr 5e-5 | 55.6% ± 2.7 | 0.580 | 0.881 | 56.8% (45.7–67.9) | 0.44–0.66 |
| freeze proj/S²CE/MAG²/AVCF, train text proj + grading decoder (2.97 M params) | 48.6% ± 1.2 | 0.511 | 0.696 | 50.6% (39.5–61.7) | 0.83–0.93 |
| 10 epochs + first-dancer bias (rerun after fix) | 56.4% ± 2.1 | 0.579 | 0.871 | 56.8% (45.7–67.9) | 0.54–0.76 |
| 40 epochs + first-dancer bias (rerun after fix) | 60.9% ± 0.6 | 0.610 | 1.169 | 61.7% (50.6–72.8) | |

- Training less makes the predictions less over-confident (log loss 0.99 → 0.85) but does not raise accuracy; freezing the RG-trained front end leaves the model at chance (log loss 0.696 = it outputs ~0.5), so the RG features are not directly usable for battles and the front end has to adapt.
- All settings so far lie in 55–61%, inside one another's CIs: with 81 battles, settings cannot be ranked yet. The round-1 RG-init number is the best point estimate, not a proven winner.
- **Bug found (fixed in 5a825e8):** the logit scale k and the order bias were trained at the model lr (1e-4); with Adam a scalar moves ~lr per step, so in ≤ 200 steps they stayed at their initial values (bias printed 0.00 in every fold, results identical to "10 epochs"). They now have their own lr (`--scalar_lr 1e-2`). The invalid bias run was discarded (logs moved to `logs/stdout/old/`). Earlier accuracies are unaffected (k does not change the sign of s_A − s_B); their log losses were computed with k ≈ 10 throughout. Rerunning: `rg_e10bias` and `initRG_bias` (40 ep).
- **Order-bias rerun (2026-10-14 00:40):** the learned first-dancer offset settles at b = 0.03–0.29 (≈ +1–7 points of probability for the first dancer), i.e. the model treats the first-dancer effect as small once it sees the performances; accuracy is unchanged (60.9% with 40 epochs, 56.4% with 10). With the scale now learnable, k grows to 16–18 in the 40-epoch runs and the log loss gets worse (1.17): the 40-epoch model is over-confident, which the fixed k = 10 had been hiding.
- **Status after rounds 1–2:** best = RG-initialised siamese MLAVL, 60.9% ± 0.6–1.2 (seed-ensemble 61.7%, CI 50.6–72.8%), AUC 0.61. That is above chance but not distinguishable from the first-dancer rule (59.3%). Hyper-parameter changes move results only within 55–61%. The remaining levers are information the model does not have yet (pose / kick-beat, the reviewed breaking vocabulary) and more training pairs (round-level pairs: every battle has 2–4 rounds per dancer).


### Round 3: vocabulary v1 and round-level pairs (2026-10-14, RG Ribbon init, 40 epochs, 3 seeds)

Reference changed (정현도, 2026-10-14): the task is judging who danced better, so results are compared with chance (50%) and content-based rules, not with the dancing order (first-dancer win rate), which the order-invariant model cannot use anyway. `tools/battle/compare_runs.py` now reports p(acc ≤ 50%) and the paired difference to the energy rule.

| run | vocab | unit | acc mean ± std | AUC | log loss | ensemble acc (95% CI) | p(acc≤50%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| initRG (round 1) | draft0 | battle | 60.9% ± 1.2 | 0.605 | 0.987 | 61.7% (50.6–72.8) | 0.015 |
| v1_battle | v1 | battle | 57.2% ± 1.5 | 0.620 | 1.156 | 58.0% (46.9–69.1) | 0.071 |
| d0_round | draft0 | round pairs | 46.5% ± 0.6 | 0.500 | 1.483 | 46.9% (35.8–58.0) | 0.711 |
| v1_round | v1 | round pairs | 48.1% ± 1.7 | 0.492 | 1.500 | 43.2% (32.1–54.3) | 0.891 |

- Round-level pairs (each round of A vs the same-index round of B, labelled with the battle winner) remove the signal: judges vote on the whole battle, single rounds do not carry it, and the extra (noisy) pairs make the model fit noise (train loss 0.08, test log loss 1.5). Dropped.
- Vocabulary v1 vs draft0: −3.7 points accuracy, +0.015 AUC; indistinguishable with 81 battles. Both are kept as options; the decision needs more battles.
- With 81 battles every comparison between settings is inside the noise. Next: more battles (203 extra World Final battles being downloaded and segmented automatically), and the pose / kick-beat branch on BRACE (running).

### Round 4: pose / kick-beat branch (2026-10-14, BRACE, RG Ribbon init, 40 epochs, 3 seeds)

`--pose`: 115-d clip features from BRACE keypoints + kick-beat alignment, added to the video tokens through a gate that starts at 0 (`ours/pose_branch.py`).

| run | vocab | acc mean ± std | AUC | log loss | ensemble acc (95% CI) | p(acc≤50%) |
| --- | --- | --- | --- | --- | --- | --- |
| initRG | draft0 | 60.9% ± 1.2 | 0.605 | 0.987 | 61.7% (50.6–72.8) | 0.015 |
| d0_pose | draft0 | 58.0% ± 1.7 | 0.581 | 1.297 | 61.7% (50.6–71.6) | 0.016 |
| v1_battle | v1 | 57.2% ± 1.5 | 0.620 | 1.156 | 58.0% (46.9–69.1) | 0.071 |
| v1_pose | v1 | 58.0% ± 1.0 | 0.596 | 1.328 | 60.5% (49.4–70.4) | 0.029 |

- No gain. The learned pose gate ends between −0.21 and +0.19 with a sign that changes across folds and seeds: the model does not find a consistent use for pose with ~70 training battles. Kick-beat alignment alone does not separate winners either (winner 0.468 vs loser 0.472; "higher BAS wins" 44%).
- Pose stays BRACE-only (the extra battles have no keypoints); it is revisited only if the larger dataset shows headroom.

### Data extension: 203 more World Final battles (2026-10-14)

정현도: "합당한 결과가 나올때까지 계속 진행해. 데이터를 더 모으고, 라벨링은 알아서 해줘".

- **Videos and labels** (`tools/battle/collect_extra_battles.py`, `ours/battle/extra_battles.csv`): official Red Bull BC One channel (2,107 videos) matched to the Wikipedia brackets 2004–2025 → 284 battles, 81 = BRACE, **203 new** (201 downloaded at 720p, 13 GB; 2 permanently 403). Winner from the bracket (agrees with BRACE labels on 81/81). Dancing order: first name in the YouTube title (= BRACE first dancer on 81/81); the Wikipedia `<sup>1</sup>` order marks agree on only 41/81 and are not used. First dancer won 61.6% of the new battles.
- **Features**: Video Swin + AST for all 201 (`/data/mlavl/features/BCONE_EXTRA`).
- **Who dances when** (no annotation outside BRACE), `tools/battle/segment_turns.py`, leave-one-year-out on BRACE:
  - BiGRU dancing detector: 92.0% clip accuracy.
  - 2-means on whole-frame Swin features to split the two dancers: 50% (chance) – frames are mostly crowd/stage.
  - Turn-start head + joint decoding of 4/6/8 alternating turns (DP, BRACE turn-count prior): dancer assignment 83.9% mean, 92.5% median; but some videos come out phase-flipped (whole battle A↔B). Confidence (weakest chosen turn start ≥ 0.4) keeps 70% of BRACE at 89%, and 66 of the 201 new battles.
- **Cost of segmentation errors** (`--brace_auto_turns`, BRACE trained/tested with held-out-year automatic turns, RG init, draft0, 3 seeds): **47.3% ± 2.3** (ensemble 44.4%, CI 34.5–55.6) vs 60.9% with the human round annotation. Turn errors, especially phase flips that invert the winner label for a battle, wipe out the signal; adding the 203 battles with these turns would add more noise than signal.
- **Appearance identity** (`tools/battle/appearance.py`): CLIP ViT-B/16 embedding of the dancer crop at each clip centre. With the BRACE annotated dancer boxes (oracle), 2-means per clip separates the dancers only 64.8% (median 60%); breaking poses (inverted, spinning, blurred) dominate the embedding. Single clips are not enough; next: use it at the turn level (mean of ~20 clips per turn) to verify/repair the alternating labelling and filter flipped videos.

### Round 5: BRACE + 89/128 extra battles, and linear probes (2026-10-14)

Turn-level appearance consistency (`segment_turns.py turn_consistency`): mean crop embedding of each turn vs the other A / B turns. On BRACE, consistency ≥ 0.8 keeps 49/81 videos, dancer assignment 91.7%, **no phase-flipped video** (12% without the filter). Applied to the extra battles: 89 kept (≥ 0.8), 128 (≥ 0.6).

Siamese MLAVL, RG init, draft0, 40 epochs, 3 seeds:

| train | test | acc mean ± std | ensemble acc (95% CI) | p(acc≤50%) |
| --- | --- | --- | --- | --- |
| BRACE 81 (round 1) | BRACE 81 | 60.9% ± 1.2 | 61.7% (50.6–72.8) | 0.015 |
| BRACE + 89 extra | BRACE 81 | 56.4% ± 2.3 | 54.3% (43.2–65.4) | 0.217 |
| BRACE + 128 extra | BRACE 81 | 54.7% ± 4.5 | 51.9% (40.7–63.0) | 0.370 |
| BRACE + 89 extra | all 170 (seed 0) | 53.5% | (45.9–60.6) | |

Linear probes (`tools/battle/linear_probe.py`, `tools/battle/probe_sources.py`): per-dancer mean/std of the features, antisymmetric logistic regression on f(A) − f(B), PCA in-fold, leave-one-year-out, 170 battles (81 BRACE + 89 extra):

| source | all 170 | unanimous 5–0 (n=61) | split 3–2 (n=44) |
| --- | --- | --- | --- |
| Video Swin, k=32 | 55.3% (47.6–62.9) | 63.9% (50.8–75.4) | 59.1% |
| AST audio, k=32 | 53.5% (45.9–61.2) | 44.3% | 54.5% |
| Swin + AST, k=32 | 48.8–49.4% | | |
| dancer-crop CLIP, k=8 | 55.3% (47.6–62.9) | 65.6% (52.5–77.0) | 38.6% |
| BRACE pose (81 only), k=16 | 58.0% (46.9–69.1) | 52.2% (n=23) | 61.9% (n=21) |
| crop + video, trained on 4–1/5–0 only, k=16 | 57.1% (50.0–64.7) | 60.7% | 54.5% |

**Conclusion of rounds 1–5.** No model or probe gets a convincing winner signal from the available representations: on 170 battles everything lies in 53–58% with CIs from ~47% to ~64%. The best single BRACE number (60.9%) did not survive more data and is best read as a selection effect over ~15 tried settings. There is a weak, consistent hint that clear (unanimous) battles are predictable above chance (60–66% across sources, CI lower bounds ≈ 47–53%) while split decisions are not. The bottleneck is the input representation (frozen Kinetics/AudioSet/CLIP features pooled over clips carry little "quality" information), not the head, the regulariser or the amount of BRACE-style data.

All-battle siamese run (`d0_extra08_all`, 3 seeds): 53.5% for every seed on 170 battles (BRACE 53.1–58.0%, extra 49.4–53.9%).

### Round 6: estimated pose for all battles + judging-criteria features (2026-10-15)

- **Pose** (`tools/battle/estimate_pose.py`): Keypoint R-CNN (COCO-17) every 3rd frame of the dancing clips of all 282 battle videos (BRACE + extra), up to 6 persons per frame. The dancer is among the detections in 93% of BRACE frames. Dancer selection: hand-made rule (tracking + size + centre + appearance box) picks the wrong person in 20.6% of frames; a learned selector (gradient boosting on candidate features + Viterbi tracking, trained on BRACE keypoints, never on winners) 12.0% (held-out years).
- **Judging-criteria features** (`tools/battle/judge_features.py`), per dancer over their own turns, normalised by bbox height and per second: dancing time, joint speed (mean, p90), inverted share, floor share, spin speed (mean, p90), freezes (rate, time share, inverted share), jerk, kick-beat BAS, freeze-on-beat, pose-cluster entropy / clusters used, travel, extension.

Winner vs loser, 170 battles (81 BRACE + 89 extra with consistent turns), paired sign test: no feature differs (all p ≥ 0.077; closest: freeze time, winner higher in 57.1% of battles, p = 0.077). Means are almost identical (inverted 0.166 vs 0.155, BAS 0.478 vs 0.479, pose entropy 2.771 vs 2.752, speed 4.13 vs 4.27).

Leave-one-year-out antisymmetric logistic regression on the 17 differences: 46.5–50.6% on 170 battles; 47.9–53.6% on all 280 (no turn filter).

**Human reference.** From the bracket vote counts of 259 battles: 5–0 113, 4–1 81, 3–2 64 (24.8% split decisions); a single judge agrees with the panel majority 83.8% of the time. That is the ceiling a model of "the judges' decision" could approach, and it shows the label itself is partly subjective.

**Conclusion after rounds 1–6.** Across the siamese MLAVL (frozen Swin/AST/CLIP features, ~15 settings), linear probes of every feature source, and interpretable judging-criteria statistics from estimated pose, battle outcomes are not predictable above chance on 170–280 battles; winners and losers are statistically indistinguishable on motion, difficulty, musicality-alignment and variety measures. What decides breaking battles (originality, character, responding to the opponent, round-by-round context) is not captured by these signals.

### Round 7: diagnosis and dancer-centred features (2026-10-15)

**Is a battle outcome predictable at all? Dancer-history Elo** (`tools/battle/elo_baseline.py`; only who the dancers are and their earlier World Final results, never the video): 60.4% (55.3–65.6) on 331 bracket matches 2004–2025; "more previous World Final wins wins" 66.5% (n = 173); on our 170 video battles with both dancers seen before 57.1% (49.4–64.7). Outcomes carry a dancer-strength signal of roughly 60–66%; the video models (53–57%) do not reach it, so the gap is in the representation.

**Dancer-centred Video Swin** (`tools/battle/extract_crop_features.py`): each 32-frame clip cropped to a square of 1.4 × the clip's median dancer box (learned selector on estimated pose), 224 px, same Swin-B. Cosine to the whole-frame feature 0.875.
- Linear probe on 170 battles: 46.5–52.4% (whole frame 50.6–55.3%); unanimous 50.8–59.0%.
- Siamese MLAVL (RG init, draft0, BRACE + 89 extra, test on all 170, 3 seeds in parallel): 52.9 / 54.1 / 55.3%, ensemble 52.4% (44.7–60.0); whole frame 53.5 ×3, ensemble 54.7% (47.1–61.8).
- No gain: giving Kinetics-pretrained Video Swin a close-up of the dancer does not make it encode performance quality.

### Round 8: label noise (2026-10-15/16; whole-frame features, RG init, draft0, BRACE + 89 extra, test on all 170, 3 seeds)

| run | seeds | AUC | ensemble acc (95% CI) | unanimous 5–0 (n=61) | 4–1 (n=50) | split 3–2 (n=44) |
| --- | --- | --- | --- | --- | --- | --- |
| hard labels (d0_extra08_all) | 53.5 / 53.5 / 53.5% | 0.563 | 54.7% (47.1–61.8) | | | |
| `--soft_labels` (target = A's vote share) | 54.1 / 53.5 / 55.3% | 0.573 | 54.7% (47.1–61.8) | 57.4% | 62.0% | 43.2% |
| `--train_min_votes 4` (train on 5–0 / 4–1 only) | 54.1 / 56.5 / 53.5% | 0.549 | 54.7% (47.1–62.4) | 55.7% | 54.0% | 50.0% |

No gain overall. The pattern seen in the probes repeats: clear decisions 55–62%, split decisions at or below chance.

**Summary of the improvement attempts (rounds 7–8).** (1) Outcomes are predictable from dancer strength (Elo 60%, past wins 66%), (2) a dancer close-up does not make Kinetics features encode it, (3) cleaner / softer labels do not help. Remaining lever: expert round-level labels (정현도) and/or a representation trained for dance quality.

### Round 9: expert evidence from real judges + fixed expert rubric (2026-10-16)

**What real judges decide on** (`tools/battle/olympics_judging.py`, Paris 2024 results book, 144 rounds × 9 judges = 1,296 votes, per-criterion signed percentages; `ours/battle/olympics_criteria.csv`):

| criterion | agrees with the judge's overall pick | share of the overall margin | summed over judges, predicts the round winner |
| --- | --- | --- | --- |
| Execution | 87.8% | 25.2% | 88.2% |
| Vocabulary | 82.3% | 22.5% | 83.3% |
| Technique | 77.0% | 19.4% | 72.9% |
| Musicality | 75.7% | 17.9% | 70.8% |
| Originality | 71.3% | 15.0% | 68.1% |

Judges on the majority side of a round: 80.9%; 5–4 / 4–5 rounds 13.9%.

**Fixed rubric** (`tools/battle/rubric_scores.py`): each criterion measured by observable cues on the dancer's own turns (CLIP zero-shot prompts on the dancer crops + estimated-pose statistics + beat alignment), standardised, weighted by the judges' margin shares above. Nothing fitted on our winners.

| score | 170 (BRACE + consistent extra) | BRACE 81 | all 280 | unanimous (n=61) |
| --- | --- | --- | --- | --- |
| Execution proxy | 48.8% | 45.7% | 51.1% | 49.2% |
| Vocabulary proxy | 54.1% | 56.8% | 52.1% | 60.7% |
| Technique proxy | 55.3% | 59.3% | 55.7% | 63.9% |
| Musicality proxy | 51.8% | 55.6% | 51.1% | 41.0% |
| Originality proxy | 60.0% (52.4–67.1) | 58.0% | 50.7% | 54.1% |
| **Weighted rubric** | 55.9% (48.2–62.9) | **63.0% (51.9–72.8)** | 54.6% (48.9–60.4) | 50.8% |
| 5 criteria, logistic LOYO | 55.9% | 54.3% | 52.5% | 63.9% |

- The rubric reaches 63.0% on BRACE (human turns), but not on the extra battles (automatic turns) or the full 280; the originality signal on the 170 set (60%) does not hold on 280 (50.7%). No result is robust across subsets.
- The most decisive criterion for real judges, **Execution (88%)**, is exactly the one our proxies cannot measure (48.8%, chance): crashes, slips and control show up as brief events that whole-clip CLIP similarity and pose statistics average away. This is the clearest statement of the measurement gap.

### Round 10: back to the paper's formulation – score regression (2026-10-16)

정현도: "원래 논문의 점수 방식으로 하는걸로 돌아가볼까". Breaking battles have no performance score, so the score is the dancer's **tournament placement** in that year (5 champion, 4 runner-up, 3 semifinal, 2 quarterfinal, 1 top-16 exit, from the Wikipedia brackets); one sample = one dancer's turns in one battle (340 performances from the 170 battles). Confound: the battle's round alone gives Spearman ρ 0.60 (a finalist is ≥ 4), so the metrics that count hold the round fixed: ρ within (year, round) groups, ρ on Top-16 battles (every dancer once; null 95% interval −0.15..0.15), and within-battle pair accuracy.

| model (leave-one-year-out) | ρ within round | ρ Top-16 | pair accuracy |
| --- | --- | --- | --- |
| linear probe, Swin / AST / both, PCA 16–64 (`tools/battle/placement_probe.py`) | −0.00 … 0.10 | −0.07 … 0.02 | 44.7–55.3% |
| MLAVL regression, RG init, 40 ep (`ours/train_placement.py`), seeds 0/1/2 | −0.02 / −0.02 / 0.03 | −0.07 / −0.03 / −0.06 | 51.2 / 53.5 / 55.3% |
| same, seed ensemble | 0.02 | −0.06 | 55.9% |

No signal: the score formulation of the paper does not change the outcome; predicting how far a dancer goes from one performance is as hard as predicting the battle. (Dancer-grouped CV gives ρ −0.2…−0.3 in the probe, the known regression-to-the-training-mean artefact of grouped CV; not evidence of an inverse signal.) Runs on GPU 0 only (policy since 2026-10-16).

### Round 11: zero-shot scoring with the reproduced models (2026-10-17)

정현도: "점수만 뽑고 점수 비교로 그냥 승패를 검증하는건 어때?" – the original model's input/output, no training on battles (`tools/battle/zero_shot_scores.py`). Each of the 18 reproduction checkpoints (RG Ball/Clubs/Hoop/Ribbon, Fis-V TES/PCS × seeds 0–2, `_best.pkl`) with its own vocabulary and settings scores each dancer's turns (same 1024-d Swin + 768-d AST clip features; ≤ 136 clips in one pass at real length, longer sequences on windows of the training clip number, averaged); higher score wins. GPU 0, 9 min for 280 battles × 18 checkpoints.

| scorer | all 280 | 170 (BRACE + consistent extra) | BRACE 81 | unanimous (n=61) |
| --- | --- | --- | --- | --- |
| single checkpoints | 44.3–56.8% | 41.8–56.5% | 39.5–63.0% | 41.0–62.3% |
| RG ensemble (12) | 51.8% (45.7–57.5) | 49.4% | 53.1% | 57.4% |
| Fis-V TES ensemble (3) | 50.7% | 51.8% | 49.4% | 59.0% |
| Fis-V PCS ensemble (3) | 50.4% | 50.0% | 51.9% | 54.1% |
| all 18 ensemble | 51.1% (45.0–56.8) | 50.0% | 54.3% | 57.4% |

Agreement between scorers (Spearman of the A − B score difference over the 280 battles): seeds of the same model 0.39–0.74 (Ball 0.69, Ribbon 0.69, TES 0.69, PCS 0.74), RG vs Fis-V models −0.12, all pairs 0.19.

Reading: the AQA models produce consistent, non-random judgements of breaking (seeds agree), but what a rhythmic-gymnastics model and a figure-skating model call "better" disagree with each other and neither agrees with breaking judges (≈ 50%). Quality as learned from gymnastics / skating scores does not transfer to battles; this also separates the earlier battle-training results from over-fitting: without any training on battles the result is the same chance level.

### Round 12: zero-shot again with PAMFN-matched audio (2026-10-18)

Phase 5b step 1. Audio for all 282 battles re-extracted the PAMFN way (`tools/battle/extract_features_pamfn.py extract --audio_only --tm_dim 64 --audio_sr 32000 --fbank_sr 32000` → `clips_pamfn/`; per-dimension mean-profile correlation with PAMFN 0.89 vs 0.45 before). Video features kept (K400 and K600 Swin-B share a feature space, see commit 3ac7ed7). GPU 0 shared with the Phase 3 Fis-V queue: ~2–3 min per video, finished overnight. Then round 11 repeated unchanged (`zero_shot_scores.py clips_pamfn`).

| scorer | all 280 | 170 | BRACE 81 | unanimous (n=61) |
| --- | --- | --- | --- | --- |
| single checkpoints | 43.9–58.2% | 40.6–58.2% | 38.3–61.7% | 44.3–63.9% |
| RG ensemble (12) | 51.4% (45.7–57.1) | 48.8% | 53.1% | 59.0% |
| Fis-V TES ensemble (3) | 50.7% | 51.8% | 46.9% | 59.0% |
| Fis-V PCS ensemble (3) | 51.1% | 51.2% | 50.6% | 49.2% |
| all 18 ensemble | 51.1% (45.4–57.1) | 50.6% | 54.3% | 57.4% |

Reading: matching the audio input to the training distribution changes nothing (all within ±1 point of round 11). The input mismatch was not why the reproduced models fail on battles; what they learned as "quality" does not transfer. Next: fine-tune with the matched features (`pamfn_all`, same setting as `d0_extra08_all`), then step 2 (backbones and prompts adapted to breaking).

Fine-tuning with the matched features (`pamfn_all`: RG Ribbon s2 init, draft0 vocabulary, BRACE + 89 consistent extra battles, test on all 170, leave-one-year-out, 40 epochs, seeds 0–2 on GPU 0, 2026-10-18 13:39 → ~21:00, ~7 h per run because GPU 0 was shared):

| features | seeds | seed ensemble (170) | p(acc ≤ 50%) | BRACE 81 | extra 89 |
| --- | --- | --- | --- | --- | --- |
| old `clips` (`d0_extra08_all`, round 5) | 53.5 / 53.5 / 53.5% | 54.7% (47.1–61.8) | 0.13 | 53.1% | 56.2% |
| PAMFN-matched audio `clips_pamfn` | 54.1 / 55.3 / 54.1% | 52.9% (45.3–60.0) | 0.24 | 50.6% | 55.1% |

Same chance level. Step 1 of Phase 5b (input matching) is closed: neither zero-shot nor fine-tuning moves. Moving on to step 2 (adapting the backbones and the prompts to breaking).

### Round 13: adapting the backbones and the prompts to breaking (Phase 5b step 2, 2026-10-18/19)

**Video: Video Swin-B K600 fine-tuned on BRACE move types** (`tools/battle/finetune_vst.py`, ckpt `ckpt/vst_move.pt`). Dancer-centred crops of the 32-frame grid clips inside the 1,352 annotated segments (cache: 256 px JPEG at box scale 1.6, 4.6 GB on /data; train: random 75–100 % crop + flip, eval: centre = box scale 1.4 = the extraction crop). BRACE's own sequence-level split: 6,363 train clips / 905 segments, 2,733 test clips / 413 segments. Trained: last stage (features.6) + norm, 25.4M params, AdamW lr 5e-5 (head 5e-4), one-cycle, 15 epochs, batch 16, fp16, GPU 0, ~2 h.

| model | test clip acc | test segment acc |
| --- | --- | --- |
| frozen K400 whole frame, linear probe (leave-one-year-out) | 0.643 | 0.736 |
| frozen K400 dancer crop, linear probe (leave-one-year-out) | 0.699 | 0.819 |
| fine-tuned K600 dancer crop, epoch 1 (kept) | 0.815 | **0.903** |
| same, epoch 14 | 0.783 | 0.869 |

Best after 1–2 epochs, then over-fits (train loss keeps falling). Move labels only; winner labels are never used and the extra battles have none.

**Audio: AST fine-tuned toward beat position and tempo** (`tools/battle/finetune_ast.py`). PAMFN preprocessing (each clip sees 0.64 s = 64 fbank frames; frozen output = `clips_pamfn` audio, r = 0.99999998). Targets from librosa beats on all 282 videos: beat in each of 8 bins of 80 ms (BCE), log2 tempo from the median inter-beat interval within ±5 s (MSE). Held out: BRACE 2020 + every 10th extra video (66,766 / 9,083 clips). AdamW lr 5e-5 (heads ×20), batch 64, ~10–20 min per run.

| trained AST layers | params | best held-out beat AUC (epoch) | tempo within 4 % |
| --- | --- | --- | --- |
| heads only (frozen = linear probe) | 0 | 0.619 (7) | 0.18 |
| last 2 | 14.2M | 0.664 (7) | 0.16 |
| last 4 | 28.4M | 0.698 (6) | 0.13 |
| last 6 (kept, `ckpt/ast_beat_b6.pt`) | 42.5M | **0.735 (5)** | 0.14 |

Frozen AudioSet AST barely encodes where the beat falls inside a clip; fine-tuning half the network brings it to 0.735. Tempo does not improve (librosa's tempo is ambiguous between half and double time, a label problem). AST's time resolution (16-frame patches, stride 10 → 5 time positions per 0.64 s) bounds the 80 ms task.

Battle features from these backbones (`scripts/extract_ft.sh`, 4 shards on GPU 0): `clips_ft` (fine-tuned dancer-crop video + PAMFN audio), `clips_ftva` (+ fine-tuned AST audio).
