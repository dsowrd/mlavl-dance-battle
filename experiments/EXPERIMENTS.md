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
| 2026-10-11 | f6fc37b | Fis-V / TES, PCS | paper settings, `SEED=1` | 0 (2 concurrent) | running since 17:47 | | | | seed 1 moved to the freed GPU 0 (정현도: "GPU 0 비었으니 Fis-V seed 1 거기서 돌려"), tmux `mlavl_fisv_s1`, logs `logs/stdout/fisv_*_s1_1011_1747.log` |
| 2026-10-11 | 32277bf | Fis-V / TES, PCS | paper settings, `SEED=2` | 3 (queued: starts when the RG seed runs on GPU 3 are all finished, expected ~22:00) | queued 18:15 | | | | moved from the GPU 1 queue (정현도: "GPU 3 비면 Fis-V seed 2도 거기서 돌려"), tmux `mlavl_fisv_s2`, logs `logs/stdout/fisv_*_s2_*.log`. Approved by 정현도 ("Fis-V도 seed 1, 2 추가해") |
