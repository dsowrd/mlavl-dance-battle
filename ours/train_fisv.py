"""Fis-V improvements over the reproduction (ours, Phase 3): multi-task TES+PCS head, LDS-weighted MSE, and post-hoc
linear calibration.

Reproduction diagnosis (tools/score_bias.py): predictions are compressed toward the mean (slope pred~label 0.51-0.54
for TES, 0.61-0.81 for PCS); the bottom fifth of TES is over-predicted by +8 to +10 points, which dominates the MSE.

  --multitask   one MLAVL predicts TES and PCS: the second score uses its own regressors / mixing weight on the same
                grade embeddings (grade_AV, grade_V); loss = MLAVL loss on TES + 10 * MSE on PCS.
  --lds         label-distribution smoothing (Yang et al., ICML 2021): per training sample weight 1/sqrt(density of its
                label, Gaussian kernel sigma = 0.05 on the normalised scale), mean 1, on the MSE term(s).
  calibration   after training, a linear map pred -> label fitted on the training-set predictions of the evaluated
                checkpoint is applied to the test predictions (reported as MSE_cal; rho is unchanged by it).

Protocol as upstream main.py: paper settings (lr 3e-4, dropout 0.7, alpha 0.5, alpha_ce 0.1, batch 64, clip 124,
Adam wd 1e-4, cosine to 1%), test every epoch, best-rho and best-MSE epoch per score reported; also last epoch.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY ours/train_fisv.py --name fisv_mt_s0 --seed 0 --multitask
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY ours/train_fisv.py --name fisv_lds_TES_s0 --seed 0 --score TES --lds
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import options  # noqa: E402
from main import setup_seed  # noqa: E402
from models import loss as mlavl_loss  # noqa: E402
from models.model import MLAVL  # noqa: E402

FEAT = '/data/mlavl/features/FISV'
RANGE = {'TES': 45.0, 'PCS': 40.0}


class FisV(Dataset):
    """Fis-V with both scores; feature dicts are loaded once (upstream reloads the whole file per item)."""
    _cache = {}

    def __init__(self, split, clip_num, train, seed):
        if 'v' not in FisV._cache:
            FisV._cache['v'] = np.load(f'{FEAT}/FISV_rgb_VST.npy', allow_pickle=True).item()
            FisV._cache['a'] = np.load(f'{FEAT}/FISV_audio_AST.npy', allow_pickle=True).item()
        self.rows = []
        for i, line in enumerate(open(f'{FEAT}/{split}.txt')):
            p = line.split()
            if i == 0 and not p[0].isdigit():
                continue
            self.rows.append((p[0], float(p[1]) / RANGE['TES'], float(p[2]) / RANGE['PCS']))
        self.n, self.train, self.rng = clip_num, train, np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        vid, tes, pcs = self.rows[i]
        v, a = FisV._cache['v'][vid], FisV._cache['a'][vid]
        if len(v) > self.n:
            st = self.rng.integers(0, len(v) - self.n) if self.train else (len(v) - self.n) // 2
            v, a = v[st:st + self.n], a[st:st + self.n]
        elif len(v) < self.n:
            v = np.concatenate([v, np.zeros((self.n - len(v), v.shape[1]))])
            a = np.concatenate([a, np.zeros((self.n - len(a), a.shape[1]))])
        f = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()
        return f(v), f(a), torch.tensor([tes, pcs], dtype=torch.float32), i


class SecondHead(nn.Module):
    """A second score from the same grade embeddings, mirroring MLAVL's score generation."""

    def __init__(self, base):
        super().__init__()
        h, n = base.regressor_V.in_features, base.n_query
        self.reg_AV, self.reg_V = nn.Linear(h, 2 * n), nn.Linear(h, n)
        self.w = nn.Parameter(torch.ones(1) * 0.5)
        self.base = [base]                      # not a submodule (avoid double registration)

    def forward(self, grade_AV, grade_V):
        b = self.base[0]
        def score(reg, g, weight):
            s = torch.sigmoid(torch.diagonal(reg(g), dim1=-2, dim2=-1))
            s = s / s.sum(1, keepdim=True)
            return (weight.unsqueeze(0) * s).sum(1)
        return self.w * score(self.reg_AV, grade_AV, b.weight_AV) + (1 - self.w) * score(self.reg_V, grade_V, b.weight_V)


def lds_weights(y, sigma=0.05):
    d = np.exp(-((y[:, None] - y[None]) ** 2) / (2 * sigma ** 2)).sum(1)
    w = 1 / np.sqrt(d)
    return w / w.mean()


def evaluate(model, head2, dl, device, scores):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for v, a, y, _ in dl:
            o = model(v.to(device), a.to(device))
            p = [o['output']] + ([head2(o['embed'], o['embed2'])] if head2 is not None else [])
            P.append(torch.stack(p, 1).cpu().numpy())
            Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    out = {}
    for k, s in enumerate(scores):
        col = 0 if s == 'TES' else 1
        p, y = P[:, k] * RANGE[s], Y[:, col] * RANGE[s]
        out[s] = (float(spearmanr(p, y)[0]), float(np.mean((p - y) ** 2)), p, y)
    return out


def main(a):
    setup_seed(a.seed)
    device = torch.device('cuda')
    scores = ['TES', 'PCS'] if a.multitask else [a.score]
    cfg = options.parser.parse_args(['--dataset', 'fisv', '--n_text', '42', '--text_feature', './FS_text_feature.npy',
                                     '--clip-num', '124', '--in_dim', '1024', '--dropout', '0.7'])
    import pandas as pd
    names = pd.read_csv('./action-label_FS.csv')['name'].tolist()
    model = MLAVL(1024, 256, 8, 1, 3, 6, 0.7, cfg, names).to(device)
    head2 = SecondHead(model).to(device) if a.multitask else None
    params = list(model.parameters()) + (list(head2.parameters()) if head2 is not None else [])
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.01)
    lf = mlavl_loss.LossFun(0.5, 1.0, alpha_ce=0.1)
    tr = FisV('train', 124, True, a.seed)
    te = FisV('test', 124, False, a.seed)
    tr_eval = FisV('train', 124, False, a.seed)
    col0 = 0 if scores[0] == 'TES' else 1
    W = {s: torch.tensor(lds_weights(np.array([r[1 if s == 'TES' else 2] for r in tr.rows])) if a.lds
                         else np.ones(len(tr.rows)), dtype=torch.float32) for s in ['TES', 'PCS']}
    dl = DataLoader(tr, batch_size=64, shuffle=True, num_workers=0)
    dte = DataLoader(te, batch_size=64, shuffle=False, num_workers=0)
    dtr = DataLoader(tr_eval, batch_size=64, shuffle=False, num_workers=0)
    best = {s: {'rho': -1, 'rho_ep': -1, 'rho_mse': None, 'mse': 1e9, 'mse_ep': -1, 'mse_rho': None} for s in scores}
    hist = []
    for ep in range(a.epochs):
        model.train()
        for v, au, y, idx in dl:
            v, au, y = v.to(device), au.to(device), y.to(device)
            o = model(v, au)
            p0 = o['output']
            aux = lf(p0, p0.detach(), o['embed'], o['embed2'], o['embed3'], o['embed4'], o['embed5'], o['embed6'],
                     o['logits1'], o['logits2'], None)               # triplet + CE terms (MSE switched off)
            w0 = W[scores[0]][idx].to(device)
            loss = aux + 10 * (w0 * (p0 - y[:, col0]) ** 2).mean()
            if head2 is not None:
                p1 = head2(o['embed'], o['embed2'])
                loss = loss + 10 * (W['PCS'][idx].to(device) * (p1 - y[:, 1]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        r = evaluate(model, head2, dte, device, scores)
        hist.append({s: r[s][:2] for s in scores})
        for s in scores:
            rho, mse = r[s][:2]
            if rho > best[s]['rho']:
                best[s].update(rho=rho, rho_ep=ep, rho_mse=mse)
                torch.save({'model': model.state_dict(), 'head2': head2.state_dict() if head2 else None},
                           f'ckpt/{a.name}_{s}_best.pkl')
            if mse < best[s]['mse']:
                best[s].update(mse=mse, mse_ep=ep, mse_rho=rho)
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f'epoch {ep}: ' + ' | '.join(f'{s} rho {r[s][0]:.3f} mse {r[s][1]:.2f}' for s in scores), flush=True)
    # post-hoc linear calibration of the best-rho checkpoint, fitted on training-set predictions
    final = {}
    for s in scores:
        ck = torch.load(f'ckpt/{a.name}_{s}_best.pkl', map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        if head2 is not None:
            head2.load_state_dict(ck['head2'])
        rt, rv = evaluate(model, head2, dtr, device, scores)[s], evaluate(model, head2, dte, device, scores)[s]
        k, c = np.polyfit(rt[2], rt[3], 1)
        mse_cal = float(np.mean((k * rv[2] + c - rv[3]) ** 2))
        slope = float(np.polyfit(rv[3], rv[2], 1)[0])
        final[s] = {**{k_: v for k_, v in best[s].items()}, 'mse_cal': mse_cal, 'slope_test': slope,
                    'last_rho': hist[-1][s][0], 'last_mse': hist[-1][s][1]}
        print(f"{s}: best rho {best[s]['rho']:.3f} (ep {best[s]['rho_ep']}, mse {best[s]['rho_mse']:.2f}, "
              f"calibrated mse {mse_cal:.2f}, slope {slope:.2f}) | best mse {best[s]['mse']:.2f} (ep {best[s]['mse_ep']}) | "
              f"last rho {hist[-1][s][0]:.3f}", flush=True)
    os.makedirs('logs/fisv', exist_ok=True)
    json.dump({'final': final, 'args': vars(a)}, open(f'logs/fisv/{a.name}.json', 'w'), indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--score', choices=['TES', 'PCS'], default='TES')
    p.add_argument('--multitask', action='store_true')
    p.add_argument('--lds', action='store_true')
    p.add_argument('--epochs', type=int, default=440)
    p.add_argument('--lr', type=float, default=3e-4)
    main(p.parse_args())
