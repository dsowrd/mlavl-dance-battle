"""Training-free baselines for battle winner prediction on the 81 BRACE battles (ours, Phase 4/6).

    $MLAVL_PY -I tools/battle/baselines.py --brace /data/mlavl/datasets/brace --labels ours/battle

Per dancer and battle it aggregates BRACE keypoints (COCO-17, 1920x1080 px) over all of the dancer's segments:
    dance_s         total annotated dancing time (s), per-video fps
    energy          mean joint speed per frame, normalised by the dancer's bbox height (scale-free)
    energy_p90      90th percentile of that per-frame speed (peaks: power moves, freezes into fast transitions)
    powermove_frac  share of the dancer's annotated frames labelled powermove
Each baseline predicts "A wins" if its score for A is larger (ties -> first dancer). Reported: pairwise accuracy over
all 81 battles with a 95% bootstrap CI (10,000 resamples of battles) and accuracy per year.
Pose statistics are cached to {cache}/pose_stats.csv (derived from BRACE, S-Lab License 1.0; not committed).
"""
import argparse
import json
import os
import re
import zipfile

import numpy as np
import pandas as pd

SEG_RE = re.compile(r'dataset/(\d{4})/([^/]+)/(.+)_(\d+)-(\d+)_(toprock|footwork|powermove)\.json')


def pose_stats(brace, fps, seqs):
    """One row per (video_id, dancer)."""
    rows = []
    z = zipfile.ZipFile(os.path.join(brace, 'dataset.zip'))
    seq_of = {}
    for r in seqs.itertuples():
        seq_of.setdefault(r.video_id, []).append((r.start_frame, r.end_frame, r.dancer))
    for name in z.namelist():
        m = SEG_RE.match(name)
        if not m:
            continue
        _, vid, _, start, end, move = m.groups()
        start, end = int(start), int(end)
        dancer = next((d for s, e, d in seq_of.get(vid, []) if s <= start and end <= e + 1), None)
        if dancer is None:
            continue
        d = json.loads(z.read(name))
        keys = sorted(d, key=lambda k: int(k.split('img-')[1].split('.')[0]))
        kp = np.array([[p[:2] for p in d[k]['keypoints']] for k in keys], np.float32)      # F x 17 x 2
        h = np.array([max(d[k]['box'][3], 1.0) for k in keys], np.float32)
        if len(kp) < 2:
            continue
        sp = np.linalg.norm(np.diff(kp, axis=0), axis=2).mean(1) / h[1:]                   # per-frame speed
        rows.append({'video_id': vid, 'dancer': dancer, 'move': move, 'frames': len(kp),
                     'speed_sum': float(sp.sum()), 'speed_n': len(sp), 'speeds': sp})
    seg = pd.DataFrame(rows)
    out = []
    for (vid, dancer), g in seg.groupby(['video_id', 'dancer']):
        sp = np.concatenate(g.speeds.values)
        frames = g.frames.sum()
        out.append({'video_id': vid, 'dancer': dancer, 'segments': len(g), 'frames': int(frames),
                    'dance_s': frames / fps[vid], 'energy': float(sp.mean()), 'energy_p90': float(np.percentile(sp, 90)),
                    'powermove_frac': float(g[g.move == 'powermove'].frames.sum() / frames)})
    return pd.DataFrame(out)


def acc_ci(correct, rng, n=10000):
    correct = np.asarray(correct, float)
    boots = rng.choice(correct, size=(n, len(correct)), replace=True).mean(1)
    return correct.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main(a):
    seqs = pd.read_csv(os.path.join(a.brace, 'repo/annotations/sequences.csv'))
    fps = pd.read_csv(os.path.join(a.brace, 'repo/videos_info.csv')).set_index('video_id').fps.to_dict()
    cache = os.path.join(a.cache, 'pose_stats.csv')
    if os.path.exists(cache):
        ps = pd.read_csv(cache)
    else:
        ps = pose_stats(a.brace, fps, seqs)
        os.makedirs(a.cache, exist_ok=True)
        ps.to_csv(cache, index=False)
    w = pd.read_csv(os.path.join(a.labels, 'winners.csv'))
    st = ps.set_index(['video_id', 'dancer'])
    feats = ['dance_s', 'energy', 'energy_p90', 'powermove_frac']
    rows = []
    for r in w.itertuples():
        A, B = st.loc[(r.video_id, r.dancer_a)], st.loc[(r.video_id, r.dancer_b)]
        row = {'video_id': r.video_id, 'year': r.year, 'y': int(r.winner_is_a)}
        for f in feats:
            row[f] = int(A[f] >= B[f])
        rows.append(row)
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    preds = {'random (expected)': None, 'first dancer wins': np.ones(len(df), int)}
    preds.update({f'higher {f}': df[f].values for f in feats})
    years = sorted(df.year.unique())
    print(f'n = {len(df)} battles; A = dancer who danced first; A won {df.y.mean():.1%}\n')
    print('| baseline | accuracy | 95% CI | ' + ' | '.join(str(y) for y in years) + ' |')
    print('| --- | --- | --- | ' + ' | '.join('---' for _ in years) + ' |')
    for name, p in preds.items():
        if p is None:
            print(f'| {name} | 50.0% | | ' + ' | '.join('' for _ in years) + ' |')
            continue
        c = (p == df.y.values)
        m, lo, hi = acc_ci(c, rng)
        per = [f'{c[df.year.values == y].mean():.0%}' for y in years]
        print(f'| {name} | {m:.1%} | {lo:.1%}–{hi:.1%} | ' + ' | '.join(per) + ' |')
    print('\nwinner vs loser, mean (paired):')
    for f in feats:
        win = [st.loc[(r.video_id, r.winner)][f] for r in w.itertuples()]
        lose = [st.loc[(r.video_id, r.dancer_b if r.winner == r.dancer_a else r.dancer_a)][f] for r in w.itertuples()]
        print(f'  {f}: winner {np.mean(win):.4f} vs loser {np.mean(lose):.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--brace', default='/data/mlavl/datasets/brace')
    p.add_argument('--labels', default='ours/battle')
    p.add_argument('--cache', default='/data/mlavl/features/BRACE')
    main(p.parse_args())
