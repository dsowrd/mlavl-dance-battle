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
| 2026-10-11 | 394eed5 | RG / Ball,Clubs,Hoop,Ribbon | as above + `--lr 1e-2` (README lr) | 0 (4 concurrent) | running since 17:31 | | | | lr comparison run, logs `logs/stdout/rg_*_1011_1731.log` |
| 2026-10-11 | 1c0f97c | RG / Ball,Clubs,Hoop,Ribbon | paper settings, `SEED=1` then `SEED=2` (`--seed`) | 3 (4 concurrent, seeds sequential) | running since 17:36 | | | | seed repeats, logs `logs/stdout/rg_*_s{1,2}_1011_*.log`. Approved by 정현도 ("GPU 3에서 seed 1, 2도 돌려") |

Suspected causes of the RG gap (to check in order):

1. **Features differ from the authors'.** MLAVL's README uses `--in_dim 768` for RG, but the only public RG features (PAMFN) are 1024-d Video Swin. The authors probably extracted their own 768-d RGB features (and possibly their own audio features); we cannot match them. This alone could explain most of the gap.
2. **Over-fitting on 200 training videos.** Train ρ reaches 0.99 while test ρ plateaus at ep 150–350; the best epochs (129–355) are well before the paper's epoch counts (340–710). Paper numbers are probably best-epoch numbers as well (the code saves `_best.pkl`), so this is not a reporting difference, but a stronger regulariser (dropout 0.5, weight decay) or fewer epochs may help.
3. **Single seed (0).** Test set is 50 videos per apparatus; ρ varies by ±0.03 between neighbouring epochs. The paper may report the best of several seeds. Repeat with seeds 1, 2 during the 10/26–10/31 trip.
4. **Trainable params 6.72M vs paper 3.82M**: the released code is not exactly the paper's configuration (to be itemised per module).
5. **README vs paper hyper-parameters**: README says lr 1e-2 and epochs 250/400/500/150 (Ball/Clubs/Hoop/Ribbon), the paper says lr 1e-4 and 620/340/450/710. The lr 1e-2 run above tests the first; if it is better, the README is what the authors actually ran.

## Phase 2 · Fis-V reproduction (FS1000 if the features are obtained)

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-11 | 93a141b | Fis-V / TES, PCS | `scripts/run_fisv.sh {TES,PCS} 1` (paper: lr 3e-4, dropout 0.7, α 0.5, α_ce 0.1, clip 124, in_dim 1024, 420/440 ep) | 1 (2 concurrent) | running since 15:54 | ~2.7 GB each | | | logs `logs/stdout/fisv_*_1011_1554.log` |
