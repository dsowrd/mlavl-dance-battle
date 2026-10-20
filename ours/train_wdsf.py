"""MLAVL as in the paper on real breaking judge scores (ours, Phase 5c): one dancer's turn in one round -> the panel's
score for that round.

Data: turns from tools/battle/wdsf_turn_detector.py (/data/mlavl/datasets/wdsf/wdsf_turns.csv; hand-labelled where
available, automatic elsewhere), scores from tools/battle/wdsf_parse.py (wdsf_judge_scores.csv). WDSF judges move
comparative sliders, so a round's score is signed: the mean over judges of the overall percentage (or one criterion),
positive towards the blue athlete. The target of a turn is that mean from its athlete's side (+ for the athlete the
judges favoured), the same number with opposite signs for the two turns of a round.
Input: PAMFN-style clip features (Video Swin-B K600 1024-d + AST 768-d, clips_pamfn/) of the clips inside the turn.
Model and loss as in the paper (RG settings): MLAVL, 10 * MSE + triplet + 0.1 * CE, Adam lr 1e-4, cosine, dropout 0.3,
grades N = 6; the output lies in [0, 1], so targets are mapped with y = clip((s + S) / 2S, 0, 1), S = 3 std.
Text prompts: the breaking move vocabulary (ours/vocab, draft0, 36 moves).
Evaluation: 5 folds grouped by event (Adult + Youth of one event together), model after the last epoch, test turns
scored over overlapping windows. Metrics: Spearman rho and MSE (in score units) over test turns, and round accuracy
= share of rounds where the turn with the higher prediction belongs to the athlete the judges favoured.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY ours/train_wdsf.py --name wdsf_total_s0 --target total --seed 0
"""
import argparse
import json
import os
import re
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
from ours.train_battle import build_model  # noqa: E402

D = '/data/mlavl/datasets/wdsf'
F = '/data/mlavl/features/WDSF'
CRIT = ['Technique', 'Vocabulary', 'Originality', 'Execution', 'Musicality']


def event_of(comp):
    return re.sub(r'-(Adult|Youth)-Breaking.*', '', comp)


def build_rows(target, min_clips=8, feat='pamfn'):
    """feat: 'pamfn' = Video Swin K600 (1024) + AST (768) per clip; 'pose' = the dancer's pose / kick-beat features
    (114, tools/battle/wdsf_pose.py, z-scored over all turns, clips without pose dropped) + the same AST audio."""
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')
    if target != 'total':
        js = js[(js.system == '5crit') & js.has_criteria]
    rnd = js.groupby(['competition', 'battle_id', 'round'])[target].mean().rename('blue_score').reset_index()
    turns = pd.read_csv(f'{D}/wdsf_turns.csv').merge(rnd, on=['competition', 'battle_id', 'round'], how='inner')
    rows, cache = [], {}
    for t in turns.itertuples():
        if t.video_id not in cache:
            d = np.load(f'{F}/clips_pamfn/{t.video_id}.npz')
            centre = (d['clip_start_frame'] + float(d['clip_len']) / 2) / float(d['fps'])
            ok = np.ones(len(centre), bool)
            v = d['rgb'].astype(np.float32)
            if feat == 'pose':
                p = np.load(f'{F}/pose/{t.video_id}.npz')
                v, ok = p['pose'].astype(np.float32), p['has_pose'] > 0
            cache[t.video_id] = (v, d['audio'].astype(np.float32), centre, ok)
        v, a, centre, ok = cache[t.video_id]
        m = (centre >= t.start_s) & (centre <= t.end_s) & ok
        if m.sum() < min_clips:
            continue
        s = t.blue_score if t.athlete == 'blue' else -t.blue_score
        rows.append(dict(video_id=t.video_id, competition=t.competition, battle_id=t.battle_id, round=t.round,
                         athlete=t.athlete, source=t.source, event=event_of(t.competition), score=float(s),
                         v=v[m], a=a[m]))
    if feat == 'pose' and rows:
        allv = np.concatenate([r['v'] for r in rows])
        mu, sd = allv.mean(0), allv.std(0) + 1e-6
        for r in rows:
            r['v'] = np.clip((r['v'] - mu) / sd, -5, 5).astype(np.float32)
    return rows


FOLDS = os.path.join(ROOT, 'ours', 'wdsf', 'folds.json')


def folds(rows, k=5):
    """Event -> fold. Fixed in ours/wdsf/folds.json (frozen 2026-10-20 from the first runs: events assigned largest
    first to the currently smallest fold) so that every run and feature set uses the same split."""
    if os.path.exists(FOLDS):
        return json.load(open(FOLDS))
    sizes = pd.Series([r['event'] for r in rows]).value_counts()
    load, assign = [0] * k, {}
    for ev, n in sizes.items():
        f = int(np.argmin(load))
        assign[ev], load[f] = f, load[f] + n
    return assign


class TurnDataset(Dataset):
    def __init__(self, rows, n, S, seed):
        self.rows, self.n, self.S, self.rng = rows, n, S, np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        v, a = crop_or_pad(r['v'], r['a'], self.n, self.rng)
        y = np.clip((r['score'] + self.S) / (2 * self.S), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(v)), torch.from_numpy(np.ascontiguousarray(a)), \
            torch.tensor(y, dtype=torch.float32)


def round_accuracy(df):
    ok, n = 0, 0
    for _, g in df.groupby(['competition', 'battle_id', 'round']):
        if len(g) != 2 or g.score.iloc[0] == 0:
            continue
        n += 1
        ok += (g.pred.idxmax() == g.score.idxmax())
    return ok / n if n else float('nan'), n


def main(a):
    os.chdir(ROOT)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device('cuda')
    rows = build_rows(a.target)
    if a.hand_only:
        rows = [r for r in rows if r['source'] == 'hand']
    S = 3 * float(np.std([r['score'] for r in rows]))
    assign = folds(rows, a.folds)
    pred = np.full(len(rows), np.nan)
    cfg = SimpleNamespace(text_feature=a.text_feature, n_text=a.n_text, clip_num=a.clip_num, dropout=a.dropout,
                          learn_text=False, action_list=a.action_list, init=a.init, pose=False)
    print(f'{len(rows)} turns, {len({(r["competition"], r["battle_id"]) for r in rows})} battles, '
          f'{len({r["event"] for r in rows})} events, target {a.target}, S = {S:.2f}', flush=True)
    for f in range(a.folds):
        tr = [r for r in rows if assign[r['event']] != f]
        te = [i for i, r in enumerate(rows) if assign[r['event']] == f]
        model = build_model(cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.01)
        lf = mlavl_loss.LossFun(a.alpha, 1.0, alpha_ce=a.alpha_ce)
        dl = DataLoader(TurnDataset(tr, a.clip_num, S, a.seed + f), batch_size=a.batch, shuffle=True, num_workers=0,
                        drop_last=True)
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
                x = torch.from_numpy(np.stack([w[0] for w in ws])).to(device)
                y_ = torch.from_numpy(np.stack([w[1] for w in ws])).to(device)
                pred[i] = model(x, y_)['output'].mean().item() * 2 * S - S
        sub = pd.DataFrame([{k: rows[i][k] for k in ('competition', 'battle_id', 'round', 'score')} for i in te]).assign(pred=pred[te])
        acc, n = round_accuracy(sub)
        print(f'fold {f}: n_test={len(te)}  rho {spearmanr(sub.score, sub.pred)[0]:.3f}  round acc {acc:.3f} (n={n})', flush=True)
    df = pd.DataFrame([{k: r[k] for k in ('video_id', 'competition', 'battle_id', 'round', 'athlete', 'source', 'event', 'score')}
                       for r in rows]).assign(pred=pred)
    rho = spearmanr(df.score, df.pred)[0]
    mse = float(((df.score - df.pred) ** 2).mean())
    acc, n = round_accuracy(df)
    hand = df[df.source == 'hand']
    acc_h, n_h = round_accuracy(hand) if len(hand) else (float('nan'), 0)
    summary = dict(rho=rho, mse=mse, mse_zero=float((df.score ** 2).mean()), round_acc=acc, n_rounds=n,
                   round_acc_hand=acc_h, n_rounds_hand=n_h, n_turns=len(df), S=S, args=vars(a))
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
    p.add_argument('--hand_only', action='store_true')
    p.add_argument('--action_list', default='ours/vocab/action-label_BREAKING_draft0.csv')
    p.add_argument('--text_feature', default='ours/vocab/BREAKING_draft0_text_feature.npy')
    p.add_argument('--n_text', type=int, default=36)
    p.add_argument('--clip_num', type=int, default=32)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--alpha_ce', type=float, default=0.1)
    main(p.parse_args())
