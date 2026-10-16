"""MLAVL as in the paper (one performance -> one score), for breaking: regress a dancer's tournament placement from
their turns in one battle (ours, Phase 6). Target = (placement - 1) / 4 in [0, 1] (5 champion ... 1 top-16 exit),
loss = 10 * MSE + MLAVL's auxiliary triplet / CE terms, leave-one-year-out, fixed epochs, model after the last epoch.
Metrics as in tools/battle/placement_probe.py (rho within round, rho on Top-16, within-battle pair accuracy).

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY ours/train_placement.py --name place_s0 --seed 0 --init ckpt/rg_..._best.pkl
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'battle'))
from models import loss as mlavl_loss  # noqa: E402
from ours.battle_dataset import crop_or_pad, windows  # noqa: E402
from ours.train_battle import build_model  # noqa: E402
from placement_probe import metrics, samples  # noqa: E402


class PerfDataset(Dataset):
    def __init__(self, rows, n, seed):
        self.rows, self.n, self.rng = rows, n, np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        v, a = crop_or_pad(r['v'], r['a'], self.n, self.rng)
        return torch.from_numpy(np.ascontiguousarray(v)).float(), torch.from_numpy(np.ascontiguousarray(a)).float(), \
            torch.tensor((r['placement'] - 1) / 4.0, dtype=torch.float32)


def main(a):
    os.chdir(ROOT)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device('cuda')
    df = samples()
    # samples() keeps pooled features only; reload per-clip arrays for the model
    from ours.battle_dataset import load_battles, load_extra_battles
    items = load_battles('/data/mlavl/features/BRACE', '/data/mlavl/datasets/brace', 'ours/battle')
    items += load_extra_battles('/data/mlavl/features/BCONE_EXTRA', 'ours/battle/extra_battles.csv', min_consistency=0.8)[0]
    arr = {(it['video_id'], s): (it['rgb_' + s], it['audio_' + s]) for it in items for s in 'ab'}
    rows = [dict(r._asdict(), v=arr[(r.video_id, r.side)][0], a=arr[(r.video_id, r.side)][1]) for r in df.itertuples()]
    pred = np.zeros(len(rows))
    for year in sorted(df.year.unique()):
        tr = [r for r in rows if r['year'] != year]
        te = [i for i, r in enumerate(rows) if r['year'] == year]
        model = build_model(a, device)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.01)
        lf = mlavl_loss.LossFun(a.alpha, 1.0, alpha_ce=a.alpha_ce)
        dl = DataLoader(PerfDataset(tr, a.clip_num, a.seed), batch_size=a.batch, shuffle=True, num_workers=0)
        for ep in range(a.epochs):
            model.train()
            for v, au, y in dl:
                v, au, y = v.to(device), au.to(device), y.to(device)
                o = model(v, au)
                loss = lf(o['output'], y, o['embed'], o['embed2'], o['embed3'], o['embed4'], o['embed5'], o['embed6'],
                          o['logits1'], o['logits2'], None)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                ws = windows(rows[i]['v'], rows[i]['a'], a.clip_num)
                x = torch.from_numpy(np.stack([w[0] for w in ws])).float().to(device)
                y_ = torch.from_numpy(np.stack([w[1] for w in ws])).float().to(device)
                pred[i] = model(x, y_)['output'].mean().item()
        print(f'fold {year}: n_test={len(te)}', flush=True)
    m = metrics(df, pred)
    out = os.path.join('logs/battle', a.name)
    os.makedirs(out, exist_ok=True)
    df.drop(columns=['x', 'xv', 'xa']).assign(pred=pred).to_csv(os.path.join(out, 'predictions.csv'), index=False)
    json.dump({**{k: float(v) for k, v in m.items()}, 'args': vars(a)}, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('ALL:', {k: round(float(v), 3) for k, v in m.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init', default=None)
    p.add_argument('--action_list', default='ours/vocab/action-label_BREAKING_draft0.csv')
    p.add_argument('--text_feature', default='ours/vocab/BREAKING_draft0_text_feature.npy')
    p.add_argument('--n_text', type=int, default=36)
    p.add_argument('--pose', action='store_true')
    p.add_argument('--clip_num', type=int, default=64)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--alpha_ce', type=float, default=0.1)
    main(p.parse_args())
