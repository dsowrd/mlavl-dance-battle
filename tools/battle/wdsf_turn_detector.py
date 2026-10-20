"""WDSF-specific turn detector, trained on hand labels (ours, Phase 5c).

Labels: /data/mlavl/datasets/wdsf/wdsf_turn_labels.json, 22 battles (106 turns) labelled by 정현도 on 2026-10-20
(review page https://claude.ai/artifact/Gh1wPRbMT7mmESY54etB3C): start / end in seconds and the athlete of every
turn. Finding: RED dances first in every round (48 / 48 rounds), so once the turns are found the athletes follow:
red, blue, red, blue, ...

Model: the BiGRU of segment_turns.py (per 32-frame clip: p(dancing), p(turn starts here)) on the K400 + AST clip
features (clips/). Decoding: the number of turns is known (2 x rounds from the judges' scores), so the turn starts
are the DP-optimal 2n cuts (wdsf_turns.decode_fixed); each turn ends where its first dancing run ends.

    $MLAVL_PY -I tools/battle/wdsf_turn_detector.py eval      # leave-one-event-out over the 22 labelled battles
    $MLAVL_PY -I tools/battle/wdsf_turn_detector.py apply     # train on all labels, segment all 313 videos
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_turns import Detector, brace_truth, smooth_binary  # noqa: E402
from wdsf_turns import decode_fixed  # noqa: E402

F = '/data/mlavl/features/WDSF'
D = '/data/mlavl/datasets/wdsf'
BRACE_F, BRACE = '/data/mlavl/features/BRACE', '/data/mlavl/datasets/brace'


def load_clips(vid):
    d = np.load(f'{F}/clips/{vid}.npz')
    X = np.concatenate([d['rgb'], d['audio']], 1).astype(np.float32)
    return X, float(d['clip_len']) / float(d['fps'])


def wdsf_data():
    L = json.load(open(f'{D}/wdsf_turn_labels.json'))
    bv = pd.read_csv(f'{D}/wdsf_battle_videos.csv').drop_duplicates('video_id').set_index('video_id')
    out = {}
    for vid, lab in L.items():
        turns = [t for t in lab['turns'] if t['start'] is not None and t['end'] is not None]
        X, sec = load_clips(vid)
        T = len(X)
        centre = (np.arange(T) + 0.5) * sec
        y = np.zeros((T, 2), np.float32)
        for t in turns:
            y[(centre >= t['start']) & (centre <= t['end']), 0] = 1.0
            i = int(t['start'] // sec)
            y[max(0, i - 1):i + 2, 1] = 1.0                     # start clip, +-1 (as on BRACE)
        event = re.sub(r'-(Adult|Youth)-Breaking.*', '', bv.loc[vid, 'competition'])
        out[vid] = dict(X=X, y=y, sec=sec, turns=turns, event=event)
    return out


def train(samples, device, epochs=60, seed=0, init=None, lr=1e-3):
    torch.manual_seed(seed)
    m = Detector().to(device)
    if init:
        m.load_state_dict(torch.load(init, map_location=device))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    pw = torch.tensor([1.0, 5.0], device=device)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        for k in rng.permutation(len(samples)):
            X, y = samples[k]
            out = m(torch.from_numpy(X)[None].to(device))[0]
            loss = nn.functional.binary_cross_entropy_with_logits(out, torch.from_numpy(y).to(device), pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def predict(m, X, device):
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(X)[None].to(device))[0]).cpu().numpy()


def segment(p, n_turns, sec, gap=2, min_turn_s=15.0):
    """Turns as [(start_s, end_s)]: DP starts, each turn ends where its first dancing run ends (gaps <= `gap`
    clips bridged) or at the next start."""
    bounds, on, cut_p = decode_fixed(p[:, 0], p[:, 1], n_turns, min_turn=max(6, int(round(min_turn_s / sec))))
    if bounds is None:
        bounds, on, cut_p = decode_fixed(p[:, 0], p[:, 1], n_turns)     # too short for the prior: relax
    if bounds is None:
        return None, []
    on = smooth_binary(p[:, 0], min_on=2, min_off=gap + 1)
    turns = []
    for s, e in bounds:
        t, miss = s, 0
        last = s
        while t < e:
            if on[t]:
                last, miss = t, 0
            else:
                miss += 1
                if miss > gap and last > s:
                    break
            t += 1
        turns.append((s * sec, (last + 1) * sec))
    return turns, cut_p


def score(pred, truth):
    """Per turn: start error, end error (s), IoU with the same-index true turn."""
    rows = []
    for i, t in enumerate(truth):
        if pred is None or i >= len(pred):
            rows.append(dict(start_err=np.nan, end_err=np.nan, iou=0.0))
            continue
        ps, pe = pred[i]
        inter = max(0.0, min(pe, t['end']) - max(ps, t['start']))
        union = max(pe, t['end']) - min(ps, t['start'])
        rows.append(dict(start_err=abs(ps - t['start']), end_err=abs(pe - t['end']), iou=inter / union if union else 0.0))
    return rows


def brace_samples():
    data = brace_truth(BRACE_F, BRACE)
    return [(X, y) for X, y, _, _ in data.values()]


def evaluate(a):
    device = torch.device('cuda')
    W = wdsf_data()
    events = sorted({d['event'] for d in W.values()})
    B = brace_samples() if a.variant in ('joint',) else []
    init = '/data/mlavl/features/BCONE_EXTRA/turn_detector.pt' if a.variant == 'finetune' else None
    rows = []
    for ev in events:
        tr = [(d['X'], d['y']) for d in W.values() if d['event'] != ev]
        if a.variant == 'joint':
            tr = tr * 3 + B                                     # WDSF videos weighted 3x against 81 BRACE videos
        ms = [train(tr, device, epochs=a.epochs, seed=a.seed + k, init=init, lr=3e-4 if init else 1e-3)
              for k in range(a.n_seeds)]
        for vid, d in W.items():
            if d['event'] != ev:
                continue
            p = np.mean([predict(m, d['X'], device) for m in ms], 0)
            pred, cut_p = segment(p, len(d['turns']), d['sec'], min_turn_s=a.min_turn_s)
            for i, r in enumerate(score(pred, d['turns'])):
                rows.append(dict(video_id=vid, event=ev, turn=i, **r))
    df = pd.DataFrame(rows)
    tag = f'{a.variant}_s{a.n_seeds}_min{a.min_turn_s:.0f}'
    df.to_csv(f'{D}/turn_detector_eval_{tag}.csv', index=False)
    per_vid = df.groupby('video_id').iou.mean()
    print(f'[{tag}] {len(df)} turns, {df.video_id.nunique()} videos (leave-one-event-out)')
    print(f'  start err median {df.start_err.median():.1f}s, <=3s {(df.start_err <= 3).mean():.2f}, <=5s {(df.start_err <= 5).mean():.2f}')
    print(f'  end err median {df.end_err.median():.1f}s, <=5s {(df.end_err <= 5).mean():.2f}')
    print(f'  IoU mean {df.iou.mean():.3f}, >=0.7 {(df.iou >= 0.7).mean():.2f}, >=0.5 {(df.iou >= 0.5).mean():.2f}')
    print(f'  videos with all turns IoU>=0.5: {(df.groupby("video_id").iou.min() >= 0.5).mean():.2f}; '
          f'worst videos: {per_vid.sort_values().head(4).round(2).to_dict()}')


def apply(a):
    device = torch.device('cuda')
    W = wdsf_data()
    tr = [(d['X'], d['y']) for d in W.values()]
    B = brace_samples() if a.variant == 'joint' else []
    if a.variant == 'joint':
        tr = tr * 3 + B
    init = '/data/mlavl/features/BCONE_EXTRA/turn_detector.pt' if a.variant == 'finetune' else None
    ms = [train(tr, device, epochs=a.epochs, seed=a.seed + k, init=init, lr=3e-4 if init else 1e-3)
          for k in range(a.n_seeds)]
    for k, m in enumerate(ms):
        torch.save(m.state_dict(), f'{F}/turn_detector_wdsf_{a.variant}_s{k}.pt')
    bv = pd.read_csv(f'{D}/wdsf_battle_videos.csv')
    bv = bv[bv.video_id.notna()].drop_duplicates('video_id')
    L = json.load(open(f'{D}/wdsf_turn_labels.json'))
    rows = []
    for r in bv.itertuples():
        if not os.path.exists(f'{F}/clips/{r.video_id}.npz'):
            continue
        X, sec = load_clips(r.video_id)
        n = 2 * int(r.n_rounds)
        if r.video_id in L:                                        # hand labels win where they exist
            turns = [(t['start'], t['end']) for t in L[r.video_id]['turns'] if t['start'] is not None]
            src, cut_p = 'hand', [1.0]
        else:
            turns, cut_p = segment(np.mean([predict(m, X, device) for m in ms], 0), n, sec, min_turn_s=a.min_turn_s)
            src = 'auto'
        if not turns:
            continue
        for i, (s, e) in enumerate(turns):
            athlete = 'red' if i % 2 == 0 else 'blue'                # red dances first in every round (48 / 48)
            rows.append({'video_id': r.video_id, 'competition': r.competition, 'battle_id': r.battle_id,
                         'round': i // 2 + 1, 'order': i % 2 + 1, 'athlete': athlete,
                         'name': r.red if athlete == 'red' else r.blue, 'start_s': round(s, 1), 'end_s': round(e, 1),
                         'source': src, 'min_cut_p': round(float(min(cut_p)) if cut_p else 0.0, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(f'{D}/wdsf_turns.csv', index=False)
    print(f'{len(out)} turns, {out.video_id.nunique()} videos ({(out.source == "hand").sum()} hand-labelled turns)')
    print('turn length s:', out.assign(d=out.end_s - out.start_s).d.describe().round(1).to_dict())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['eval', 'apply'])
    p.add_argument('--variant', choices=['wdsf', 'finetune', 'joint'], default='wdsf')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_seeds', type=int, default=1)
    p.add_argument('--min_turn_s', type=float, default=15.0)
    main = evaluate if p.parse_args().mode == 'eval' else apply
    main(p.parse_args())
