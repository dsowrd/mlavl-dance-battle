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
            rounds = g[g.dancer == name].sort_values('start_frame')[['start_frame', 'end_frame']].values
            keep = np.zeros(len(starts), bool)
            out['rounds_' + side] = []
            for s, e in rounds:
                m = (starts >= s) & (starts + L <= e + 1)
                keep |= m
                if m.any():
                    out['rounds_' + side].append((d['rgb'][m], d['audio'][m]))
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
        y, first = it['y'], 1.0              # first = 1 if the left-hand dancer danced first
        if self.rng.random() < self.swap:
            va, aa, vb, ab, y, first = vb, ab, va, aa, 1 - y, -1.0
        f = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()
        return f(va), f(aa), f(vb), f(ab), torch.tensor(float(y)), torch.tensor(first)


def round_pairs(item):
    """The i-th round of A paired with the i-th round of B (battles alternate A, B, A, B, ...)."""
    n = min(len(item['rounds_a']), len(item['rounds_b']))
    return [(item['rounds_a'][i], item['rounds_b'][i]) for i in range(n)]


class RoundPairDataset(BattleDataset):
    """Round-level training pairs: every (round i of A, round i of B) of a training battle is one example labelled
    with the battle winner (a noisy label for a single round: judges vote on the whole battle). Gives ~2.9 pairs per
    battle instead of 1."""

    def __init__(self, items, clip_num=32, swap=0.5, seed=0):
        super().__init__(items, clip_num, swap, seed)
        self.pairs = [(it, ra, rb) for it in items for ra, rb in round_pairs(it)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        it, (va, aa), (vb, ab) = self.pairs[i]
        va, aa = crop_or_pad(va, aa, self.n, self.rng)
        vb, ab = crop_or_pad(vb, ab, self.n, self.rng)
        y, first = it['y'], 1.0
        if self.rng.random() < self.swap:
            va, aa, vb, ab, y, first = vb, ab, va, aa, 1 - y, -1.0
        f = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()
        return f(va), f(aa), f(vb), f(ab), torch.tensor(float(y)), torch.tensor(first)

