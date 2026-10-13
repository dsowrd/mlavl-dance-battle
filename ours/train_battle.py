"""Siamese MLAVL for 1-vs-1 battle winner prediction, leave-one-year-out on BRACE (ours, Phase 5).

The same MLAVL network scores both dancers; P(A wins) = sigmoid(k * (s_A - s_B)) with a learnable scale k.
Loss = BCE on the winner + MLAVL's label-free auxiliary terms (hard-triplet separation of the grade/text embeddings,
CE that matches each grade embedding to its own text query), so the grading structure of the original model is kept.
The score is antisymmetric in A/B and training swaps A/B at random, so the first-dancer prior (59%) is not used.

No validation split exists (81 battles), so the number of epochs is fixed in advance and the model after the last
epoch is evaluated; nothing is selected on the test year. At test time each dancer is scored on overlapping
`clip_num` windows covering all of their rounds and the window scores are averaged.

    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY ours/train_battle.py --init ckpt/fisv_PCS_1011_1554_best.pkl --seed 0 --name pcs_s0

Writes logs/battle/{name}/predictions.csv (one row per battle, fold = held-out year) and summary.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import options  # noqa: E402
from models import loss as mlavl_loss  # noqa: E402
from models.model import MLAVL  # noqa: E402
from ours.battle_dataset import BattleDataset, load_battles, windows  # noqa: E402


def build_model(a, device):
    cfg = options.parser.parse_args([])
    cfg.dataset, cfg.text_feature, cfg.n_text, cfg.n_query = 'rg', a.text_feature, a.n_text, 6
    cfg.clip_num, cfg.in_dim, cfg.dropout = a.clip_num, 1024, a.dropout
    names = pd.read_csv(a.action_list)['name'].tolist()
    model = MLAVL(1024, 256, 8, 1, 3, 6, a.dropout, cfg, names).to(device)
    if a.init:
        sd = torch.load(a.init, map_location=device, weights_only=True)
        sd = {k: v for k, v in sd.items() if k != 'text_feature'}       # vocabulary differs; keep ours
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert set(missing) <= {'text_feature'} and not unexpected, (missing, unexpected)
    return model


def aux_loss(lf, out):
    """MLAVL's triplet + CE terms only (MSE switched off by comparing the prediction with itself)."""
    p = out['output']
    return lf(p, p.detach(), out['embed'], out['embed2'], out['embed3'], out['embed4'], out['embed5'],
              out['embed6'], out['logits1'], out['logits2'], None)


@torch.no_grad()
def score(model, v, a, n, device):
    ws = windows(v, a, n)
    x = torch.from_numpy(np.stack([w[0] for w in ws])).float().to(device)
    y = torch.from_numpy(np.stack([w[1] for w in ws])).float().to(device)
    return model(x, y)['output'].mean().item()


def run_fold(a, items, test_year, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = [it for it in items if it['year'] != test_year]
    test = [it for it in items if it['year'] == test_year]
    model = build_model(a, device)
    log_k = torch.nn.Parameter(torch.tensor(float(np.log(a.init_scale)), device=device))
    params = [p for n_, p in model.named_parameters() if p.requires_grad] + [log_k]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.01)
    lf = mlavl_loss.LossFun(a.alpha, 1.0, alpha_ce=a.alpha_ce)
    dl = DataLoader(BattleDataset(train, a.clip_num, swap=0.5, seed=seed), batch_size=a.batch, shuffle=True,
                    num_workers=0, drop_last=False)
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for va, aa, vb, ab, y in dl:
            va, aa, vb, ab, y = (t.to(device) for t in (va, aa, vb, ab, y))
            oa, ob = model(va, aa), model(vb, ab)
            logit = log_k.exp() * (oa['output'] - ob['output'])
            loss = F.binary_cross_entropy_with_logits(logit, y) + a.aux * 0.5 * (aux_loss(lf, oa) + aux_loss(lf, ob))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
    model.eval()
    k = log_k.exp().item()
    rows = []
    for it in test:
        sa = score(model, it['rgb_a'], it['audio_a'], a.clip_num, device)
        sb = score(model, it['rgb_b'], it['audio_b'], a.clip_num, device)
        p = 1 / (1 + np.exp(-k * (sa - sb)))
        rows.append({'video_id': it['video_id'], 'year': it['year'], 'round': it['round'], 'y': it['y'],
                     'score_a': sa, 'score_b': sb, 'p_a': p, 'correct': int((p > 0.5) == bool(it['y']))})
    train_loss = tot / len(train)
    return rows, train_loss, k


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main(a):
    device = torch.device('cuda')
    items = load_battles(a.features, a.brace, a.labels)
    out_dir = os.path.join('logs/battle', a.name)
    os.makedirs(out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for year in sorted({it['year'] for it in items}):
        r, tl, k = run_fold(a, items, year, device, a.seed)
        acc = np.mean([x['correct'] for x in r])
        print(f'fold {year}: n={len(r)} acc={acc:.3f} final train loss {tl:.4f} k={k:.1f} ({time.time() - t0:.0f}s)', flush=True)
        rows += r
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    rng = np.random.default_rng(0)
    c = df.correct.values
    boots = rng.choice(c, size=(10000, len(c))).mean(1)
    summ = {'name': a.name, 'n': len(df), 'accuracy': float(c.mean()), 'ci95': [float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))], 'auc': auc(df.y, df.p_a),
            'log_loss': float(-np.mean(df.y * np.log(df.p_a + 1e-7) + (1 - df.y) * np.log(1 - df.p_a + 1e-7))),
            'per_year': df.groupby('year').correct.mean().round(3).to_dict(), 'args': vars(a)}
    json.dump(summ, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1, default=str)
    print(f"ALL: acc {summ['accuracy']:.3f} (95% CI {summ['ci95'][0]:.3f}-{summ['ci95'][1]:.3f}), AUC {summ['auc']:.3f}, "
          f"log loss {summ['log_loss']:.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--features', default='/data/mlavl/features/BRACE')
    p.add_argument('--brace', default='/data/mlavl/datasets/brace')
    p.add_argument('--labels', default='ours/battle')
    p.add_argument('--action_list', default='ours/vocab/action-label_BREAKING.csv')
    p.add_argument('--text_feature', default='ours/vocab/BREAKING_text_feature.npy')
    p.add_argument('--n_text', type=int, default=36)
    p.add_argument('--init', default=None, help='MLAVL state dict to start from (e.g. a Fis-V or RG best ckpt)')
    p.add_argument('--clip_num', type=int, default=64)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--alpha_ce', type=float, default=0.1)
    p.add_argument('--aux', type=float, default=0.1, help='weight of the MLAVL auxiliary loss relative to BCE')
    p.add_argument('--init_scale', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
