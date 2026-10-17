"""Is there high-score underestimation in our Fis-V reproduction? (ours, Phase 3 diagnostic)

Loads the reproduction checkpoints (ckpt/fisv_{TES,PCS}*_best.pkl), predicts the Fis-V test set and reports, per
score and per checkpoint: Spearman rho, MSE, the slope of prediction vs label (1 = calibrated, < 1 = compressed toward
the mean), and the mean residual (prediction - label, in points) in the top / middle / bottom fifth of the labels.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY tools/score_bias.py
"""
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import options  # noqa: E402
from datasets import FisVDataset  # noqa: E402
from models.model import MLAVL  # noqa: E402

FEAT = '/data/mlavl/features/FISV'


def args_for(score):
    return options.parser.parse_args([
        '--dataset', 'fisv', '--action-type', score, '--video-path', FEAT, '--audio-path', FEAT,
        '--train-label-path', f'{FEAT}/train.txt', '--test-label-path', f'{FEAT}/test.txt',
        '--action_list', './action-label_FS.csv', '--text_feature', './FS_text_feature.npy', '--n_text', '42',
        '--clip-num', '124', '--in_dim', '1024', '--hidden_dim', '256', '--n_head', '8', '--n_encoder', '1',
        '--n_decoder', '3', '--n_query', '6', '--dropout', '0.7'])


@torch.no_grad()
def predict(score, ckpt, device):
    a = args_for(score)
    ds = FisVDataset(a.video_path, a.audio_path, a.test_label_path, a.action_list, a.clip_num, score, train=False, args=a)
    names = ds.classes
    m = MLAVL(a.in_dim, a.hidden_dim, a.n_head, a.n_encoder, a.n_decoder, a.n_query, a.dropout, a, names).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    P, Y = [], []
    for i in range(len(ds)):
        v, au, y = ds[i]
        P.append(m(v[None].to(device), au[None].to(device))['output'].item())
        Y.append(float(y))
    return np.array(P) * ds.score_range, np.array(Y) * ds.score_range


def main():
    device = torch.device('cuda')
    print('| ckpt | rho | MSE | slope pred~label | residual bottom 20% | middle | top 20% |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for score in ['TES', 'PCS']:
        for ck in sorted(glob.glob(f'ckpt/fisv_{score}*_best.pkl')):
            p, y = predict(score, ck, device)
            q20, q80 = np.percentile(y, [20, 80])
            r = p - y
            slope = np.polyfit(y, p, 1)[0]
            print(f'| {os.path.basename(ck)} | {spearmanr(p, y)[0]:.3f} | {np.mean(r ** 2):.2f} | {slope:.2f} | '
                  f'{r[y <= q20].mean():+.2f} | {r[(y > q20) & (y < q80)].mean():+.2f} | {r[y >= q80].mean():+.2f} |')


if __name__ == '__main__':
    main()
