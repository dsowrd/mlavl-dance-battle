# breaking-judge: reproducing MLAVL and extending it to breaking battle judging

This repository builds on **MLAVL** (Xu et al., "Language-Guided Audio-Visual Learning for Long-Term Sports Assessment", CVPR 2025, [code](https://github.com/XuHuangbiao/MLAVL)). Most of the model code is theirs. This project reproduces their results and extends the method to 1-vs-1 dance battle judging as a course project (Deep Learning, Dept. of Intelligent Robot Engineering, CBNU, Fall 2026). The upstream commit history is preserved; `main` tracks upstream and our work lives on `dev`.

## What is ours vs. upstream

Upstream (unchanged unless listed): `models/`, `clip/`, `datasets.py`, `train.py`, `test.py`, `utils.py`, `action-label_*.csv`, `*_text_feature.npy`, `Framework.png`.

Our changes to upstream files (each file carries a `# Modified from MLAVL` header; commits prefixed `fix(upstream):`):

- `options.py`: `--dataset {fs1000,fisv,rg,logo}`, `--text_feature`, `--use_clip`, `--alpha_ce`.
- `main.py`: picks the Dataset class from `--dataset` (upstream hard-coded `FS1000Dataset`); passes `--alpha_ce`.
- `models/model.py`: loads the precomputed CLIP text features instead of building ViFi-CLIP (`--use_clip` restores it); removed hard-coded `.cuda()`.
- `models/loss.py`: CE-loss weight is an argument (paper uses 0.01 for FS1000, 0.1 for Fis-V/RG).

Ours (new files; commits prefixed `feat:` / `data:` / `exp:` / `docs:`):

- `scripts/`: environment and run scripts, `smoke_test.py`.
- `experiments/EXPERIMENTS.md`: every run, one line each.
- `ours/vocab/`: breaking move vocabulary for the MAG² graph (draft).
- `ours/`, `tools/battle/` (planned): pose branch, kick-beat feature, battle dataset, pairwise (siamese) judging.

## Reproduction results

Full per-run details (commit, GPU, wall time, best epochs, suspected causes of gaps) are in `experiments/EXPERIMENTS.md`.
Ours = paper hyper-parameters, public PAMFN features (1024-d Video Swin RGB + 768-d AST audio); each row is mean ± std over seeds 0, 1, 2 (one run per seed, best test epoch per run as tracked by the upstream code).
ρ is the best test Spearman correlation over epochs; MSE is the best test MSE over epochs (both as tracked by the upstream code).

| Data | Paper ρ / MSE | Ours ρ / MSE | Note |
| --- | --- | --- | --- |
| RG Ball | 0.826 / 5.57 | 0.758 ± 0.009 / 5.52 ± 0.17 | 3 seeds |
| RG Clubs | 0.829 / 4.20 | 0.749 ± 0.041 / 5.08 ± 0.61 | 3 seeds |
| RG Hoop | 0.871 / 4.11 | 0.786 ± 0.035 / 4.73 ± 0.47 | 3 seeds |
| RG Ribbon | 0.866 / 3.99 | 0.808 ± 0.030 / 5.05 ± 0.60 | 3 seeds |
| **RG avg** | **0.849 / 4.47** | **0.775 / 5.10** | mean over seeds 0, 1, 2 then over apparatus; Fisher-z mean ρ 0.776. README's `--in_dim 768` suggests the authors used different RGB features from the public 1024-d ones |
| Fis-V TES | 0.766 / 19.44 | 0.723 ± 0.013 / 20.77 ± 0.53 | 3 seeds |
| Fis-V PCS | 0.863 / 7.17 | 0.852 ± 0.006 / 7.93 ± 0.39 | 3 seeds |
| **Fis-V avg** | **0.823 / 13.31** | **0.788 / 14.35** | Fisher-z mean ρ 0.797 |
| FS1000 (avg) | 0.90 / 10.39 | – | features are Baidu-only; not obtained |

## Acknowledgements

| Work | Used for | Notes |
| --- | --- | --- |
| [MLAVL](https://github.com/XuHuangbiao/MLAVL) (Xu et al., CVPR 2025) | model code, text features, action label CSVs | base of this repository |
| [GDLT](https://github.com/xuangch/CVPR22_GDLT) (Xu et al., CVPR 2022) | code base of MLAVL | |
| [PAMFN](https://github.com/qinghuannn/PAMFN) (Zeng & Zheng, TIP 2024) | RG / Fis-V features and labels | |
| [Skating-Mixer / FS1000](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) (Xia et al., AAAI 2023) | FS1000 features and labels | |
| Rhythmic Gymnastics dataset (ACTION-NET, Zeng et al., ACM MM 2020) | RG dataset | |
| [BRACE](https://github.com/dmoltisanti/brace) (Moltisanti et al., ECCV 2022) | battle video list, 2D keypoints, beats | S-Lab License 1.0, non-commercial; videos are used for research only and are not redistributed |
| CLIP / ViFi-CLIP, Video Swin, AST | text / video / audio features | |
| Red Bull BC One official results | winner labels | per-year source URLs recorded in `winners.csv` |

Full BibTeX entries are copied from each project's official page as they are used (see **Citation**). Upstream's original README follows.

---

# MLAVL: Language-Guided Audio-Visual Learning for Long-Term Sports Assessment

This is the code for CVPR2025 paper "Language-Guided Audio-Visual Learning for Long-Term Sports Assessment".

## Abstract
Long-term sports assessment is a challenging task in video understanding since it requires judging complex movement variations and action-music coordination. However, there is no direct correlation between the diverse background music and movements in sporting events. Previous works require a large number of model parameters to learn potential associations between actions and music. To address this issue, we propose a language-guided audio-visual learning (MLAVL) framework that models" audio-action-visual" correlations guided by low-cost language modality. In our framework, multidimensional domain-based actions form action knowledge graphs, motivating audio-visual modalities to focus on task-relevant actions. We further design a shared-specific context encoder to integrate deep multimodal semantics, and an audio-visual cross-modal fusion module to evaluate action-music consistency. To match the sport's rules, we then propose a dual-branch prompt-guided grading module to weigh both visual and audio-visual performance. Extensive experiments demonstrate that our approach achieves state-of-the-art on four public long-term sports benchmarks while maintaining low parameters.

![Framework](Framework.png)

## Environments

- RTX 3090
- CUDA: 12.4
- Python: 3.8.19
- PyTorch: 2.4.1+cu124

## Features

The features (RGB, Audio, Flow) and label files of Rhythmic Gymnastics and Fis-V dataset can be download from the [PAMFN](https://github.com/qinghuannn/PAMFN) repository.

The features (RGB, Audio) and label files of FS1000 dataset can be download from the [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) repository.

The features (RGB) and label files of LOGO dataset can be download from the [UIL-AQA](https://github.com/dx199771/Interpretability-AQA/tree/main) repository.

## Pretrained Model
If you wish to extract your own action text labels, please download the [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) pretrained model and place it in:
```
weights/k400_clip_complete_finetuned_30_epochs.pth
```

## Running
### The following are examples only, more details coming soon!

**Note:** Both the FS1000 and Fis-V datasets feature figure skating scenes and share the same action text features: “action-label_FS.csv” and “FS_text_feature.npy”.

Please fill in or select the args enclosed by {} first. For example, on the **FS1000** dataset:

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {Ball/Clubs/Hoop/Ribbon} --lr 1e-2 --epoch {250/400/500/150} --in_dim 768 --n_head 8 --n_encoder 1 --n_decoder 3 --n_query 6 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {Ball/Clubs/Hoop/Ribbon} --in_dim 768 --n_head 8 --n_encoder 1 --n_decoder 3 --n_query 6 --test --ckpt {the name of the used checkpoint}
```

Additionally, you can select a result that balances SRCC and MSE based on the specific requirements of your application.

Be patient and persistent in tuning the code to achieve new state-of-the-art results.

## Citation
If our project is helpful for your research, please consider citing:
```
@InProceedings{Xu_2025_CVPR,
    author    = {Xu, Huangbiao and Ke, Xiao and Wu, Huanqi and Xu, Rui and Li, Yuezhou and Guo, Wenzhong},
    title     = {Language-Guided Audio-Visual Learning for Long-Term Sports Assessment},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {23967-23977}
}
```

## Acknowledgement
This repository builds upon [GDLT (CVPR 2022)](https://github.com/xuangch/CVPR22_GDLT).

We thank the authors for their contributions to the research community.