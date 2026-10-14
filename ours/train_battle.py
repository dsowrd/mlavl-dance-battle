"""Siamese MLAVL for 1-vs-1 battle winner prediction, leave-one-year-out on BRACE (ours, Phase 5).

The same MLAVL network scores both dancers; P(A wins) = sigmoid(k * (s_A - s_B)) with a learnable scale k.
Loss = BCE on the winner + MLAVL's label-free auxiliary terms (hard-triplet separation of the grade/text embeddings,
CE that matches each grade embedding to its own text query), so the grading structure of the original model is kept.
The score is antisymmetric in A/B and training swaps A/B at random, so the first-dancer prior (59%) is not used,
unless --order_bias adds a learned logit offset b for "the left-hand dancer danced first" (sign flips on swap).
--freeze PREFIXES freezes those submodules (and keeps their BatchNorm statistics fixed), e.g. everything up to the
fused clip features so only the text projection and the grading decoder/regressors adapt to battles.

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
from ours.pose_branch import add_pose_branch  # noqa: E402
from ours.battle_dataset import (BattleDataset, RoundPairDataset, load_battles, load_brace_auto, load_extra_battles,  # noqa: E402
                                 round_pairs, windows)


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
    if a.pose:
        add_pose_branch(model)
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
    test = [it for it in items if it['year'] == test_year and (a.test_on == 'all' or it.get('source') != 'extra')]
    if not test:
        return [], float('nan'), float('nan'), 0.0, 0, []
    model = build_model(a, device)
    log_k = torch.nn.Parameter(torch.tensor(float(np.log(a.init_scale)), device=device))
    bias = torch.nn.Parameter(torch.zeros((), device=device), requires_grad=a.order_bias)
    frozen = [m for m in a.freeze.split(',') if m]
    for n_, p in model.named_parameters():
        if any(n_.startswith(m + '.') for m in frozen):
            p.requires_grad = False
    params = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.endswith('.gate')]
    scalars = [log_k] + ([bias] if a.order_bias else []) + [p for n_, p in model.named_parameters() if n_.endswith('.gate')]
    # the scale k and the order bias get their own, larger lr: with Adam a scalar moves ~lr per step, so at the
    # model lr (1e-4) they could not leave their initial values within a few hundred steps (seen in rg_e10bias)
    opt = torch.optim.Adam([{'params': params, 'lr': a.lr, 'weight_decay': 1e-4},
                            {'params': scalars, 'lr': a.scalar_lr, 'weight_decay': 0.0}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.01)
    lf = mlavl_loss.LossFun(a.alpha, 1.0, alpha_ce=a.alpha_ce)
    DS = RoundPairDataset if a.unit == 'round' else BattleDataset
    dl = DataLoader(DS(train, a.clip_num, swap=0.5, seed=seed), batch_size=a.batch, shuffle=True,
                    num_workers=0, drop_last=False)
    for ep in range(a.epochs):
        model.train()
        for m in frozen:
            model.get_submodule(m).eval()
        tot = 0.0
        for va, aa, vb, ab, y, first in dl:
            va, aa, vb, ab, y, first = (t.to(device) for t in (va, aa, vb, ab, y, first))
            oa, ob = model(va, aa), model(vb, ab)
            logit = log_k.exp() * (oa['output'] - ob['output']) + bias * first
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
        if a.unit == 'round':   # battle logit = mean over round pairs of the round-level logits
            sc = [(score(model, ra[0], ra[1], a.clip_num, device), score(model, rb[0], rb[1], a.clip_num, device))
                  for ra, rb in round_pairs(it)]
            sa, sb = float(np.mean([x[0] for x in sc])), float(np.mean([x[1] for x in sc]))
        else:
            sa = score(model, it['rgb_a'], it['audio_a'], a.clip_num, device)
            sb = score(model, it['rgb_b'], it['audio_b'], a.clip_num, device)
        p = 1 / (1 + np.exp(-(k * (sa - sb) + bias.item())))           # A always danced first at test
        rows.append({'video_id': it['video_id'], 'year': it['year'], 'round': it['round'], 'y': it['y'],
                     'source': it.get('source', 'brace'),
                     'score_a': sa, 'score_b': sb, 'p_a': p, 'correct': int((p > 0.5) == bool(it['y']))})
    train_loss = tot / len(dl.dataset)
    gate = [p.item() for n_, p in model.named_parameters() if n_.endswith('.gate')]
    return rows, train_loss, k, bias.item(), sum(p.numel() for p in params) + len(scalars), gate


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main(a):
    device = torch.device('cuda')
    if a.brace_auto_turns:   # BRACE with automatic turns (segmentation-noise check); source 'brace_auto' is evaluated
        items = load_brace_auto(a.features, a.labels)
        for it in items:
            it['source'] = 'brace'
    else:
        items = load_battles(a.features, a.brace, a.labels, pose=a.pose)
    if a.extra:
        assert not a.pose, 'extra battles have no BRACE keypoints, so --pose cannot be combined with --extra'
        extra, skipped = load_extra_battles(a.extra_features, a.extra_csv, a.extra_min_cut_p,
                                            min_consistency=a.extra_min_consistency)
        print(f'extra battles: {len(extra)} used, {skipped} skipped (missing features / low-confidence turns)')
        items += extra
    out_dir = os.path.join('logs/battle', a.name)
    os.makedirs(out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for year in sorted({it['year'] for it in items}):
        r, tl, k, b, npar, gate = run_fold(a, items, year, device, a.seed)
        if not r:
            print(f'fold {year}: no test battles (train-only year)', flush=True)
            continue
        acc = np.mean([x['correct'] for x in r])
        print(f'fold {year}: n={len(r)} acc={acc:.3f} final train loss {tl:.4f} k={k:.1f} bias={b:.2f} '
              f'trainable={npar / 1e6:.2f}M' + (f' pose_gate={gate[0]:.3f}' if gate else '') + f' ({time.time() - t0:.0f}s)', flush=True)
        rows += r
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    rng = np.random.default_rng(0)
    c = df.correct.values
    boots = rng.choice(c, size=(10000, len(c))).mean(1)
    summ = {'name': a.name, 'n': len(df), 'accuracy': float(c.mean()), 'ci95': [float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))], 'auc': auc(df.y, df.p_a),
            'log_loss': float(-np.mean(df.y * np.log(df.p_a + 1e-7) + (1 - df.y) * np.log(1 - df.p_a + 1e-7))),
            'per_year': df.groupby('year').correct.mean().round(3).to_dict(),
            'per_source': df.groupby('source').correct.agg(['mean', 'size']).round(3).to_dict('index'), 'args': vars(a)}
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
    p.add_argument('--brace_auto_turns', action='store_true',
                   help='use held-out-year automatic turns for BRACE instead of the human annotation')
    p.add_argument('--extra', action='store_true', help='also train on the extra (non-BRACE) battles')
    p.add_argument('--extra_features', default='/data/mlavl/features/BCONE_EXTRA')
    p.add_argument('--extra_csv', default='ours/battle/extra_battles.csv')
    p.add_argument('--extra_min_cut_p', type=float, default=0.0)
    p.add_argument('--extra_min_consistency', type=float, default=0.8)
    p.add_argument('--test_on', choices=['brace', 'all'], default='brace',
                   help='evaluate on BRACE battles only (human round annotation) or also on the extra battles')
    p.add_argument('--pose', action='store_true', help='add the pose / kick-beat branch (ours/pose_branch.py)')
    p.add_argument('--unit', choices=['battle', 'round'], default='battle',
                   help="training examples: whole battles, or round pairs labelled with the battle winner")
    p.add_argument('--freeze', default='', help='comma-separated submodule prefixes to freeze, e.g. '
                   'proj_visual,proj_audio,transformer.encoder,gcn_V,gcn_A,T_decoder,clip_fusion')
    p.add_argument('--order_bias', action='store_true', help='learn a first-dancer logit offset')
    p.add_argument('--scalar_lr', type=float, default=1e-2, help='lr of the logit scale k and the order bias')
    main(p.parse_args())
