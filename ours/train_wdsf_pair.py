"""Pairwise MLAVL for comparative judging (ours, Phase 5c improvement 1).

WDSF judges score a round with comparative sliders: the label is the margin between the two athletes of that round,
not a score of either performance alone. The paper's formulation (ours/train_wdsf.py: one turn -> its signed score)
lands at 51-53% round accuracy, below the 54.4% of "the second dancer wins". Here the same MLAVL (same modules,
prompts and auxiliary losses) scores both turns of a round with shared weights, and the loss is on the margin:
    d = s(blue turn) - s(red turn) in [-1, 1],  target t = clip(margin / S, -1, 1),  S = 3 std of the margins
    loss = 10 * MSE(d, t) + MLAVL triplet / CE terms on both turns (their MSE term switched off).
Scoring each turn separately keeps the paper's model (a per-performance score) and makes the prediction
antisymmetric by construction (swapping the athletes negates d).
Evaluation as in train_wdsf.py: 5 event-grouped folds, round accuracy (sign of d vs the judges' margin), and the
Spearman rho of d against the margin.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY ours/train_wdsf_pair.py --name pair_total_s0 --target total --seed 0
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import loss as mlavl_loss  # noqa: E402
from ours.battle_dataset import crop_or_pad, windows  # noqa: E402
from ours.train_battle import aux_loss, build_model  # noqa: E402
from ours.train_wdsf import CRIT, build_rows, folds  # noqa: E402


def pairs(rows):
    by = {}
    for r in rows:
        by.setdefault((r['competition'], r['battle_id'], r['round']), {})[r['athlete']] = r
    out = []
    for k, d in by.items():
        if 'blue' in d and 'red' in d:
            out.append(dict(key=k, event=d['blue']['event'], margin=d['blue']['score'], b=d['blue'], r=d['red'],
                            source='hand' if d['blue']['source'] == 'hand' else 'auto'))
    return out


class PairDataset(Dataset):
    def __init__(self, P, n, S, seed):
        self.P, self.n, self.S, self.rng = P, n, S, np.random.default_rng(seed)

    def __len__(self):
        return len(self.P)

    def __getitem__(self, i):
        p = self.P[i]
        vb, ab = crop_or_pad(p['b']['v'], p['b']['a'], self.n, self.rng)
        vr, ar = crop_or_pad(p['r']['v'], p['r']['a'], self.n, self.rng)
        t = np.clip(p['margin'] / self.S, -1, 1)
        f = lambda x: torch.from_numpy(np.ascontiguousarray(x))
        return f(vb), f(ab), f(vr), f(ar), torch.tensor(t, dtype=torch.float32)


@torch.no_grad()
def turn_score(model, r, n, device):
    ws = windows(r['v'], r['a'], n)
    x = torch.from_numpy(np.stack([w[0] for w in ws])).to(device)
    y = torch.from_numpy(np.stack([w[1] for w in ws])).to(device)
    return model(x, y)['output'].mean().item()


def main(a):
    os.chdir(ROOT)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device('cuda')
    rows = build_rows(a.target)
    P = [p for p in pairs(rows) if p['margin'] != 0]
    S = 3 * float(np.std([p['margin'] for p in P]))
    assign = folds(rows, a.folds)
    cfg = SimpleNamespace(text_feature=a.text_feature, n_text=a.n_text, clip_num=a.clip_num, dropout=a.dropout,
                          learn_text=False, action_list=a.action_list, init=a.init, pose=False)
    print(f'{len(P)} rounds, target {a.target}, S = {S:.2f}', flush=True)
    pred = np.full(len(P), np.nan)
    for f in range(a.folds):
        tr = [p for p in P if assign[p['event']] != f]
        te = [i for i, p in enumerate(P) if assign[p['event']] == f]
        model = build_model(cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.01)
        lf = mlavl_loss.LossFun(a.alpha, 1.0, alpha_ce=a.alpha_ce)
        dl = DataLoader(PairDataset(tr, a.clip_num, S, a.seed + f), batch_size=a.batch, shuffle=True, num_workers=0,
                        drop_last=True)
        for ep in range(a.epochs):
            model.train()
            for vb, ab, vr, ar, t in dl:
                B = vb.shape[0]
                o = model(torch.cat([vb, vr]).to(device), torch.cat([ab, ar]).to(device))
                d = o['output'][:B] - o['output'][B:]
                loss = 10 * torch.mean((d - t.to(device)) ** 2) + a.w_aux * aux_loss(lf, o)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        for i in te:
            pred[i] = (turn_score(model, P[i]['b'], a.clip_num, device) - turn_score(model, P[i]['r'], a.clip_num, device)) * S
        m = np.array([P[i]['margin'] for i in te])
        print(f'fold {f}: n_test={len(te)}  rho {spearmanr(m, pred[te])[0]:.3f}  round acc {(np.sign(pred[te]) == np.sign(m)).mean():.3f}',
              flush=True)
    df = pd.DataFrame([dict(competition=p['key'][0], battle_id=p['key'][1], round=p['key'][2], event=p['event'],
                            source=p['source'], margin=p['margin']) for p in P]).assign(pred=pred)
    acc = float((np.sign(df.pred) == np.sign(df.margin)).mean())
    hand = df[df.source == 'hand']
    summary = dict(rho=float(spearmanr(df.margin, df.pred)[0]), round_acc=acc, n_rounds=len(df),
                   second_dancer_rule=float((df.margin > 0).mean()),
                   round_acc_hand=float((np.sign(hand.pred) == np.sign(hand.margin)).mean()) if len(hand) else float('nan'),
                   n_rounds_hand=len(hand), S=S, args=vars(a))
    out = os.path.join('logs/wdsf', a.name)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'predictions.csv'), index=False)
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=float)
    print('ALL:', {k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items() if k != 'args'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--target', default='total', choices=['total'] + CRIT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--init', default=None)
    p.add_argument('--action_list', default='ours/vocab/action-label_BREAKING_draft0.csv')
    p.add_argument('--text_feature', default='ours/vocab/BREAKING_draft0_text_feature.npy')
    p.add_argument('--n_text', type=int, default=36)
    p.add_argument('--clip_num', type=int, default=32)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--alpha_ce', type=float, default=0.1)
    p.add_argument('--w_aux', type=float, default=1.0)
    main(p.parse_args())
