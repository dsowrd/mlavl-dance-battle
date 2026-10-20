"""Linear reference for the comparative WDSF scores (ours, Phase 5c): antisymmetric ridge on the difference of the two
turns' pooled features (mean and std over the turn's clips of video and audio), PCA in-fold, the same event folds as
the MLAVL runs (ours/wdsf/folds.json). Writes logs/wdsf/<name>/predictions.csv for ours/eval_pair.py.

    $MLAVL_PY ours/probe_diff.py --feat pose --target Technique --k 8
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ours.train_wdsf import build_rows, folds  # noqa: E402
from ours.train_wdsf_pair import pairs  # noqa: E402


def main(a):
    os.chdir(ROOT)
    rows = build_rows(a.target, feat=a.feat)
    P = [p for p in pairs(rows) if p['margin'] != 0]
    assign = folds(rows)
    feat = lambda r: np.concatenate([r['v'].mean(0), r['v'].std(0)] + ([] if a.video_only else [r['a'].mean(0), r['a'].std(0)]))
    X = np.stack([feat(p['b']) - feat(p['r']) for p in P])
    y = np.array([p['margin'] for p in P])
    ev = np.array([p['event'] for p in P])
    pred = np.zeros(len(P))
    for f in sorted(set(assign.values())):
        tr = np.array([assign[e] != f for e in ev])
        m = make_pipeline(StandardScaler(), PCA(a.k, random_state=0), Ridge(alpha=10.0))
        m.fit(np.concatenate([X[tr], -X[tr]]), np.concatenate([y[tr], -y[tr]]))
        pred[~tr] = m.predict(X[~tr])
    out = os.path.join('logs/wdsf', a.name or f'probe_{a.feat}{"_vonly" if a.video_only else ""}_k{a.k}_{a.target}')
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([dict(competition=p['key'][0], battle_id=p['key'][1], round=p['key'][2], event=p['event'],
                       source=p['source'], margin=p['margin']) for p in P]).assign(pred=pred).to_csv(
        os.path.join(out, 'predictions.csv'), index=False)
    print(out, f'round acc {(np.sign(pred) == np.sign(y)).mean():.3f} (n={len(y)})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--feat', default='pamfn', choices=['pamfn', 'pose'])
    p.add_argument('--target', default='total')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--video_only', action='store_true', help='leave the audio statistics out')
    p.add_argument('--name', default='')
    main(p.parse_args())
