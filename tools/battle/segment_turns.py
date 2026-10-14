"""Who is dancing when: turn segmentation for battle videos without BRACE annotations (ours, Phase 4 extension).

Step 1, dancing detector: a small bidirectional GRU over the clip features (Video Swin 1024 + AST 768) predicts per
clip whether a round is being danced (1) or not (MC, crowd, walking in/out, replays: 0). Trained on BRACE, where a
clip is 1 if it lies inside an annotated sequence.
Step 2, dancer assignment: dancing clips are split into the two dancers by clustering their video features
(2-means on L2-normalised Video Swin features, then the label sequence is smoothed so that a turn lasts at least
`min_run` clips); the cluster that dances first is dancer A (the first name in the YouTube title, verified on BRACE).
Turns alternate A, B, A, B in all 81 BRACE battles; the clustering does not need a pause between turns.

    $MLAVL_PY -I tools/battle/segment_turns.py eval            # leave-one-year-out evaluation on BRACE
    $MLAVL_PY -I tools/battle/segment_turns.py apply --features /data/mlavl/features/BCONE_EXTRA --ids ...
Writes {features}/turns/{video_id}.npz with dancer (T,) int8: 0 none, 1 A, 2 B (apply mode), and for eval mode
prints clip-level dancing accuracy, dancer-assignment accuracy and round-count agreement.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Detector(nn.Module):
    def __init__(self, d_in=1792, h=128):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(0.3))
        self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, 2)          # [dancing, turn starts here]

    def forward(self, x):                      # (B, T, d) -> (B, T, 2) logits
        return self.out(self.gru(self.inp(x))[0])


def brace_truth(features, brace):
    """video_id -> (X (T, 1792), dancing (T,), dancer (T,) 0/1/2) from BRACE annotations."""
    seqs = pd.read_csv(os.path.join(brace, 'repo/annotations/sequences.csv'))
    out = {}
    for vid, g in seqs.groupby('video_id'):
        d = np.load(os.path.join(features, 'clips', vid + '.npz'))
        st, L = d['clip_start_frame'], int(d['clip_len'])
        g = g.sort_values('start_frame')
        first = g.dancer.iloc[0]
        lab = np.zeros(len(st), np.int8)
        start = np.zeros(len(st), np.float32)
        for r in g.itertuples():
            m = (st + L / 2 >= r.start_frame) & (st + L / 2 <= r.end_frame)   # clip centre inside the round
            lab[m] = 1 if r.dancer == first else 2
            if m.any():
                i = np.argmax(m)
                start[max(0, i - 1):i + 2] = 1.0                                # first clip of the round, +-1
        y = np.stack([(lab > 0).astype(np.float32), start], 1)
        out[vid] = (np.concatenate([d['rgb'], d['audio']], 1).astype(np.float32), y, lab, int(g.year.iloc[0]))
    return out


def train_detector(data, ids, device, epochs=60, seed=0):
    torch.manual_seed(seed)
    m = Detector().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        m.train()
        for vid in np.random.default_rng(seed + _).permutation(ids):
            X, y = data[vid][0], data[vid][1]
            x = torch.from_numpy(X)[None].to(device)
            yt = torch.from_numpy(y).to(device)
            pw = torch.tensor([1.0, 5.0], device=device)                    # starts are rare
            loss = nn.functional.binary_cross_entropy_with_logits(m(x)[0], yt, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def smooth_binary(p, thr=0.5, min_on=4, min_off=2):
    on = p > thr
    # fill short gaps, then drop short islands
    runs = lambda b: [(i, j) for i, j in _runs(b)]
    for i, j in runs(~on):
        if j - i < min_off and i > 0 and j < len(on):
            on[i:j] = True
    for i, j in runs(on):
        if j - i < min_on:
            on[i:j] = False
    return on


def _runs(b):
    i, n = 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j < n and b[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def assign_dancers(X_video, on, min_run=4, iters=20):
    """2-means on the dancing clips' L2-normalised video features + run-length smoothing. Returns 0/1/2 per clip."""
    idx = np.where(on)[0]
    lab = np.zeros(len(on), np.int8)
    if len(idx) < 2:
        return lab
    f = X_video[idx] / np.linalg.norm(X_video[idx], axis=1, keepdims=True)
    # init: first and the most dissimilar dancing clip
    c = np.stack([f[:8].mean(0), f[np.argmin(f @ f[:8].mean(0))]])
    for _ in range(iters):
        a = np.argmax(f @ c.T, 1)
        c = np.stack([f[a == k].mean(0) if (a == k).any() else c[k] for k in range(2)])
    a = np.argmax(f @ c.T, 1)
    # smooth: majority vote in a sliding window over the dancing clips, then merge short runs into neighbours
    w = 2
    a = np.array([np.bincount(a[max(0, i - w):i + w + 1], minlength=2).argmax() for i in range(len(a))])
    changed = True
    while changed:
        changed = False
        rs = list(_runs_labels(a))
        for k, (i, j, v) in enumerate(rs):
            if j - i < min_run and len(rs) > 1:
                a[i:j] = rs[k - 1][2] if k > 0 else rs[k + 1][2]
                changed = True
                break
    first = a[0]
    lab[idx] = np.where(a == first, 1, 2)
    return lab


def _runs_labels(a):
    i = 0
    while i < len(a):
        j = i
        while j < len(a) and a[j] == a[i]:
            j += 1
        yield i, j, a[i]
        i = j


TURN_PRIOR = {4: 24 / 162, 6: 135 / 162, 8: 3 / 162}   # BRACE: rounds per dancer 2 / 3 / 4


def decode_turns(p_dance, p_start, min_turn=6):
    """Pick the turn starts jointly: the first dancing clip starts turn 1; choose N-1 further starts (N in 4/6/8 total
    turns, prior from BRACE) among dancing clips, at least `min_turn` apart, maximising
    sum log p_start(cut) + sum log(1 - p_start) elsewhere-equivalent (i.e. sum of log-odds at cuts) + log prior(N).
    Turns alternate A (1), B (2). Returns (labels, dancing mask, score of the chosen N)."""
    on = smooth_binary(p_dance)
    lab = np.zeros(len(on), np.int8)
    idx = np.where(on)[0]
    if len(idx) == 0:
        return lab, on, -np.inf
    pe = np.clip(p_start, 1e-4, 1 - 1e-4)
    lo = np.log(pe) - np.log(1 - pe)
    cand = [t for t in idx if t - idx[0] >= min_turn and idx[-1] - t >= min_turn - 1]
    best = (-np.inf, None, None)
    for N, prior in TURN_PRIOR.items():
        k = N - 1                                       # cuts to place
        if len(cand) < k:
            continue
        # DP over candidates in time order: f[j][c] = best score with c cuts, last cut at cand[c-index j]
        n = len(cand)
        f = np.full((k + 1, n), -np.inf)
        bp = np.full((k + 1, n), -1, int)
        for j in range(n):
            f[1, j] = lo[cand[j]]
        for c in range(2, k + 1):
            for j in range(n):
                prev = [i for i in range(j) if cand[j] - cand[i] >= min_turn and f[c - 1, i] > -np.inf]
                if prev:
                    i = max(prev, key=lambda i: f[c - 1, i])
                    f[c, j], bp[c, j] = f[c - 1, i] + lo[cand[j]], i
        j = int(np.argmax(f[k]))
        score = f[k, j] + np.log(prior)
        if score > best[0]:
            cuts, c = [], k
            while c >= 1 and j >= 0:
                cuts.append(cand[j]); j = bp[c, j]; c -= 1
            best = (score, N, sorted(cuts))
    if best[1] is None:
        lab[on] = 1
        return lab, on, -np.inf
    bounds = [idx[0]] + best[2] + [len(on)]
    for n_ in range(len(bounds) - 1):
        m = np.zeros(len(on), bool)
        m[bounds[n_]:bounds[n_ + 1]] = True
        lab[m & on] = 1 + (n_ % 2)
    return lab, on, best[0]


def n_turns(lab):
    return len([1 for i, j, v in _runs_labels(lab) if v > 0])


def evaluate(a):
    device = torch.device('cuda')
    data = brace_truth(a.features, a.brace)
    years = sorted({v[3] for v in data.values()})
    rows = []
    for y in years:
        tr = [k for k, v in data.items() if v[3] != y]
        te = [k for k, v in data.items() if v[3] == y]
        m = train_detector(data, tr, device)
        for vid in te:
            X, dancing, truth, _ = data[vid]
            with torch.no_grad():
                p = torch.sigmoid(m(torch.from_numpy(X)[None].to(device))[0]).cpu().numpy()
            lab, on, sc = decode_turns(p[:, 0], p[:, 1])
            lab_cluster = assign_dancers(X[:, :1024], on)              # appearance clustering (kept for reference)
            dancing_clips = truth > 0
            rows.append({'video_id': vid, 'year': y,
                         'dancing_acc': float((on == dancing_clips).mean()),
                         'dancer_acc': float((lab[dancing_clips] == truth[dancing_clips]).mean()),
                         'dancer_acc_cluster': float((lab_cluster[dancing_clips] == truth[dancing_clips]).mean()),
                         'turns_true': n_turns(truth), 'turns_pred': n_turns(lab), 'score': sc,
                         'min_cut_p': float(min([p[t, 1] for t in np.where(np.diff(lab.astype(int)) != 0)[0] + 1 if lab[t] > 0] or [0]))})
        print(f'fold {y}: done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.features, 'turn_segmentation_eval.csv'), index=False)
    print(df[['dancing_acc', 'dancer_acc', 'dancer_acc_cluster']].describe().round(3).to_string())
    print('turn count exact:', (df.turns_true == df.turns_pred).mean().round(3),
          '| videos with dancer_acc >= 0.9:', (df.dancer_acc >= 0.9).mean().round(3))


def apply(a):
    device = torch.device('cuda')
    data = brace_truth(a.brace_features, a.brace)
    m = train_detector(data, list(data), device)
    torch.save(m.state_dict(), os.path.join(a.features, 'turn_detector.pt'))
    os.makedirs(os.path.join(a.features, 'turns'), exist_ok=True)
    ids = a.ids or sorted(f[:-4] for f in os.listdir(os.path.join(a.features, 'clips')) if f.endswith('.npz'))
    for vid in ids:
        d = np.load(os.path.join(a.features, 'clips', vid + '.npz'))
        X = np.concatenate([d['rgb'], d['audio']], 1).astype(np.float32)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.from_numpy(X)[None].to(device))[0]).cpu().numpy()
        lab, on, sc = decode_turns(p[:, 0], p[:, 1])
        np.savez(os.path.join(a.features, 'turns', vid + '.npz'), dancer=lab, p_dancing=p[:, 0], p_start=p[:, 1], score=sc)
        print(f'{vid}: T={len(lab)} dancing {on.mean():.2f} turns {n_turns(lab)}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['eval', 'apply'])
    p.add_argument('--features', default='/data/mlavl/features/BRACE')
    p.add_argument('--brace_features', default='/data/mlavl/features/BRACE')
    p.add_argument('--brace', default='/data/mlavl/datasets/brace')
    p.add_argument('--ids', nargs='*')
    a = p.parse_args()
    evaluate(a) if a.mode == 'eval' else apply(a)
