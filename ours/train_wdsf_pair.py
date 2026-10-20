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


def build_model_dim(cfg, device, in_dim):
    """MLAVL as in ours/train_battle.build_model but with the visual input dimension of a non-Swin feature (pose)."""
    import options
    from models.model import MLAVL
    c = options.parser.parse_args([])
    c.dataset, c.text_feature, c.n_text, c.n_query = 'rg', cfg.text_feature, cfg.n_text, 6
    c.clip_num, c.in_dim, c.dropout, c.learn_text = cfg.clip_num, in_dim, cfg.dropout, False
    names = pd.read_csv(cfg.action_list)['name'].tolist()
    return MLAVL(in_dim, 256, 8, 1, 3, 6, cfg.dropout, c, names).to(device)


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
    rows = build_rows(a.target, feat=a.feat)
    P = [p for p in pairs(rows) if p['margin'] != 0]
    S = 3 * float(np.std([p['margin'] for p in P]))
    assign = folds(rows, a.folds)
    cfg = SimpleNamespace(text_feature=a.text_feature, n_text=a.n_text, clip_num=a.clip_num, dropout=a.dropout,
                          learn_text=False, action_list=a.action_list, init=a.init, pose=False)
    print(f'{len(P)} rounds, target {a.target}, S = {S:.2f}', flush=True)
    pred = np.full(len(P), np.nan)
    E = sorted({int(e) for e in a.eval_epochs.split(',')} | {a.epochs})
    pred_e = {e: np.full(len(P), np.nan) for e in E}
    chosen = {}
    for f in range(a.folds):
        tr = [p for p in P if assign[p['event']] != f]
        te = [i for i, p in enumerate(P) if assign[p['event']] == f]
        # inner validation (selects the epoch without looking at the test fold): the largest training event
        tr_events = pd.Series([p['event'] for p in tr]).value_counts()
        val_ev = tr_events.index[0] if a.inner_val else None
        fit = [p for p in tr if p['event'] != val_ev]
        val = [p for p in tr if p['event'] == val_ev]
        model = build_model(cfg, device) if a.feat == 'pamfn' else build_model_dim(cfg, device, P[0]['b']['v'].shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.01)
        lf = mlavl_loss.LossFun(a.alpha, 1.0, alpha_ce=a.alpha_ce)
        dl = DataLoader(PairDataset(fit, a.clip_num, S, a.seed + f), batch_size=a.batch, shuffle=True, num_workers=0,
                        drop_last=True)
        val_acc = {}
        for ep in range(1, a.epochs + 1):
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
            if ep in E:
                model.eval()
                for i in te:
                    pred_e[ep][i] = (turn_score(model, P[i]['b'], a.clip_num, device)
                                     - turn_score(model, P[i]['r'], a.clip_num, device)) * S
                if val:
                    pv = np.array([turn_score(model, p['b'], a.clip_num, device) - turn_score(model, p['r'], a.clip_num, device)
                                   for p in val])
                    val_acc[ep] = float((np.sign(pv) == np.sign([p['margin'] for p in val])).mean())
        best = max(val_acc, key=lambda e: (val_acc[e], -e)) if val_acc else a.epochs
        chosen[f] = dict(val_event=val_ev, val_acc=val_acc, epoch=best)
        pred[te] = pred_e[best][te]
        m = np.array([P[i]['margin'] for i in te])
        print(f'fold {f}: n_test={len(te)}  val {val_ev} {val_acc}  -> epoch {best}  '
              f'rho {spearmanr(m, pred[te])[0]:.3f}  round acc {(np.sign(pred[te]) == np.sign(m)).mean():.3f}', flush=True)
    df = pd.DataFrame([dict(competition=p['key'][0], battle_id=p['key'][1], round=p['key'][2], event=p['event'],
                            source=p['source'], margin=p['margin']) for p in P]).assign(
        pred=pred, **{f'pred_e{e}': pred_e[e] for e in E})
    acc = float((np.sign(df.pred) == np.sign(df.margin)).mean())
    hand = df[df.source == 'hand']
    summary = dict(rho=float(spearmanr(df.margin, df.pred)[0]), round_acc=acc, n_rounds=len(df),
                   second_dancer_rule=float((df.margin > 0).mean()),
                   round_acc_hand=float((np.sign(hand.pred) == np.sign(hand.margin)).mean()) if len(hand) else float('nan'),
                   n_rounds_hand=len(hand), S=S, chosen=chosen,
                   acc_at_epoch={e: float((np.sign(pred_e[e]) == np.sign(df.margin)).mean()) for e in E}, args=vars(a))
    out = os.path.join('logs/wdsf', a.name)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'predictions.csv'), index=False)
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=float)
    print('ALL:', {k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items() if k not in ('args', 'chosen')})


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
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--feat', default='pamfn', choices=['pamfn', 'pose'])
    p.add_argument('--eval_epochs', default='5,10,20,30,40')
    p.add_argument('--inner_val', type=int, default=1, help='select the epoch on a held-out training event')
    main(p.parse_args())
