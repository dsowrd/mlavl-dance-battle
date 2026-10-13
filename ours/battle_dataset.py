"""Battle dataset for siamese MLAVL training on BRACE (ours, Phase 5).

One item = one battle: clip features of dancer A and dancer B (A = the dancer who danced first) and the label
y = 1 if A won. A dancer's sequence is the concatenation, in time order, of every 32-frame clip that lies fully
inside one of that dancer's rounds (BRACE sequences), so the crowd/MC/transition time between rounds is excluded.
Features come from tools/battle/extract_features.py ({features}/clips/{video_id}.npz).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_battles(features, brace, labels):
    seqs = pd.read_csv(os.path.join(brace, 'repo/annotations/sequences.csv'))
    win = pd.read_csv(os.path.join(labels, 'winners.csv'))
    items = []
    for r in win.itertuples():
        d = np.load(os.path.join(features, 'clips', r.video_id + '.npz'))
        starts, L = d['clip_start_frame'], int(d['clip_len'])
        g = seqs[seqs.video_id == r.video_id]
        out = {'video_id': r.video_id, 'year': int(r.year), 'round': r.round, 'y': int(r.winner_is_a),
               'dancer_a': r.dancer_a, 'dancer_b': r.dancer_b}
        for side, name in (('a', r.dancer_a), ('b', r.dancer_b)):
            rounds = g[g.dancer == name][['start_frame', 'end_frame']].values
            keep = np.zeros(len(starts), bool)
            for s, e in rounds:
                keep |= (starts >= s) & (starts + L <= e + 1)
            out['rgb_' + side] = d['rgb'][keep]
            out['audio_' + side] = d['audio'][keep]
        items.append(out)
    return items


def crop_or_pad(v, a, n, rng=None):
    t = len(v)
    if t > n:
        st = rng.integers(0, t - n + 1) if rng is not None else (t - n) // 2
        return v[st:st + n], a[st:st + n]
    if t < n:
        v = np.concatenate([v, np.zeros((n - t, v.shape[1]), v.dtype)])
        a = np.concatenate([a, np.zeros((n - t, a.shape[1]), a.dtype)])
    return v, a


def windows(v, a, n):
    """Overlapping length-n windows (stride n//2) covering the whole sequence; padded if shorter."""
    t = len(v)
    if t <= n:
        return [crop_or_pad(v, a, n)]
    st = list(range(0, t - n + 1, n // 2))
    if st[-1] != t - n:
        st.append(t - n)
    return [(v[s:s + n], a[s:s + n]) for s in st]


class BattleDataset(Dataset):
    """Training view: random crop of each dancer to clip_num and, with probability `swap`, A/B swapped (label
    flipped), so the model cannot learn the first-dancer prior and has to compare the performances."""

    def __init__(self, items, clip_num=64, swap=0.5, seed=0):
        self.items, self.n, self.swap = items, clip_num, swap
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        va, aa = crop_or_pad(it['rgb_a'], it['audio_a'], self.n, self.rng)
        vb, ab = crop_or_pad(it['rgb_b'], it['audio_b'], self.n, self.rng)
        y = it['y']
        if self.rng.random() < self.swap:
            va, aa, vb, ab, y = vb, ab, va, aa, 1 - y
        f = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()
        return f(va), f(aa), f(vb), f(ab), torch.tensor(float(y))
