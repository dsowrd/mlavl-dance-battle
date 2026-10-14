"""Is there a winner signal in the clip features at all? Linear probe with leave-one-year-out (ours, Phase 6).

Per dancer: mean and std over the dancer's clips of the Video Swin (1024) and AST (768) features (optionally
only one modality). Per battle: d = f(A) - f(B). Logistic regression on d without intercept (antisymmetric: swapping
A/B flips the prediction), trained on both d (label y) and -d (label 1-y), standardised, PCA to k dims inside each
training fold, L2 strength C. Evaluated per held-out year, pooled; test sets are BRACE (human turns) and the extra
battles (automatic turns, consistency filter) separately.

    $MLAVL_PY -I tools/battle/linear_probe.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ours.battle_dataset import load_battles, load_extra_battles  # noqa: E402


def feat(v, a, which):
    parts = []
    if which in ('video', 'both'):
        parts += [v.mean(0), v.std(0)]
    if which in ('audio', 'both'):
        parts += [a.mean(0), a.std(0)]
    return np.concatenate(parts)


def margin(it, w):
    r = w.get(it['video_id'])
    if r is None:
        return np.nan
    nums = [int(x) for x in str(r).replace('/', ' ').split() if x.isdigit()]
    return nums[0] if nums else np.nan


def run(items, which, k, C, rng):
    D = np.stack([feat(i['rgb_a'], i['audio_a'], which) - feat(i['rgb_b'], i['audio_b'], which) for i in items])
    y = np.array([i['y'] for i in items])
    yr = np.array([i['year'] for i in items])
    src = np.array([i.get('source', 'brace') for i in items])
    p = np.zeros(len(items))
    for year in np.unique(yr):
        tr, te = yr != year, yr == year
        X = np.concatenate([D[tr], -D[tr]])
        Y = np.concatenate([y[tr], 1 - y[tr]])
        sc = StandardScaler(with_mean=False).fit(X)            # no centring: keep d -> -d antisymmetry
        Z = sc.transform(X)
        pca = PCA(min(k, Z.shape[1]), random_state=0).fit(np.concatenate([Z, -Z]))   # symmetric data -> mean 0
        clf = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(pca.transform(Z), Y)
        p[te] = clf.predict_proba(pca.transform(sc.transform(D[te])))[:, 1]
    out = {}
    for s in ['brace', 'extra', 'all']:
        m = (src == s) if s != 'all' else np.ones(len(y), bool)
        c = ((p[m] > 0.5) == y[m]).astype(float)
        b = rng.choice(c, (5000, len(c))).mean(1)
        out[s] = (c.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(c))
    return out, p, y, src


def main():
    rng = np.random.default_rng(0)
    brace = load_battles('/data/mlavl/features/BRACE', '/data/mlavl/datasets/brace', 'ours/battle')
    for it in brace:
        it['source'] = 'brace'
    extra, _ = load_extra_battles('/data/mlavl/features/BCONE_EXTRA', 'ours/battle/extra_battles.csv',
                                  min_consistency=0.8)
    items = brace + extra
    print(f'{len(brace)} BRACE + {len(extra)} extra battles\n')
    print('| features | PCA k | C | BRACE acc (95% CI) | extra acc (95% CI) | all acc (95% CI) |')
    print('| --- | --- | --- | --- | --- | --- |')
    best = None
    for which in ['video', 'audio', 'both']:
        for k in [8, 32]:
            for C in [0.01, 1.0]:
                o, p, y, src = run(items, which, k, C, rng)
                f = lambda t: f'{t[0]:.1%} ({t[1]:.1%}–{t[2]:.1%}, n={t[3]})'
                print(f"| {which} | {k} | {C} | {f(o['brace'])} | {f(o['extra'])} | {f(o['all'])} |")
    # vote margin: are clear (5-0 / 4-1) battles easier? (bracket score of the winner)
    w = pd.read_csv('ours/battle/winners.csv').set_index('video_id')
    e = pd.read_csv('ours/battle/extra_battles.csv').set_index('video_id')
    print('\nwinner vote count (non-2014 brackets), share of battles: ',
          pd.concat([w.bracket_score_a.where(w.winner_is_a == 1, w.bracket_score_b),
                     e.bracket_score_a.where(e.winner_is_a == 1, e.bracket_score_b)]).value_counts().head(8).to_dict())


if __name__ == '__main__':
    main()
