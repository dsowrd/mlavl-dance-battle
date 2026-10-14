"""Compare battle runs against each other and against the training-free rules (ours, Phase 6).

    $MLAVL_PY -I tools/battle/compare_runs.py initRG v1_battle v1_round ...   # tags = logs/battle/{tag}_s{seed}

Per tag: accuracy mean ± std over seeds, AUC, log loss, and for the seed ensemble (mean p over seeds) the accuracy with
a 95% bootstrap CI, the one-sided bootstrap p of accuracy <= 50%, and the paired bootstrap difference to the
"higher joint-speed energy wins" rule (a content-based rule; same battles resampled for both).

The task is judging who danced better, so the reference is chance (50%) and content-based rules. The dancing order
(first dancer wins 59% of BRACE battles) is not a reference: the model is order-invariant by construction
(antisymmetric score, random A/B swaps), and a result counts as valid if its CI lower bound is above 50%.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

ENERGY = '/data/mlavl/features/BRACE/pose_stats.csv'


def main(tags, n=10000):
    rng = np.random.default_rng(0)
    st = pd.read_csv(ENERGY).set_index(['video_id', 'dancer'])
    w = pd.read_csv('ours/battle/winners.csv').set_index('video_id')
    print('| run | seeds | acc mean ± std | AUC | log loss | ensemble acc (95% CI) | p(acc≤50%) | Δ vs energy rule (95% CI) |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for t in tags:
        files = sorted(glob.glob(f'logs/battle/{t}_s*/summary.json'))
        if not files:
            print(f'| {t} | 0 | | | | | | |')
            continue
        S = [json.load(open(f)) for f in files]
        P = [pd.read_csv(f.replace('summary.json', 'predictions.csv')).set_index('video_id').sort_index() for f in files]
        y = P[0].y.values
        p = np.mean([d.p_a.values for d in P], 0)
        c = ((p > 0.5) == y).astype(float)
        ids = P[0].index
        ea = np.array([st.loc[(v, w.loc[v, 'dancer_a'])].energy for v in ids])
        eb = np.array([st.loc[(v, w.loc[v, 'dancer_b'])].energy for v in ids])
        rule = ((ea >= eb) == (y == 1)).astype(float)          # higher joint-speed energy wins
        idx = rng.integers(0, len(y), (n, len(y)))
        acc_b, diff_b = c[idx].mean(1), (c[idx] - rule[idx]).mean(1)
        accs = [s['accuracy'] for s in S]
        print(f"| {t} | {len(S)} | {np.mean(accs):.1%} ± {np.std(accs):.1%} | {np.mean([s['auc'] for s in S]):.3f} | "
              f"{np.mean([s['log_loss'] for s in S]):.3f} | {c.mean():.1%} ({np.percentile(acc_b, 2.5):.1%}–{np.percentile(acc_b, 97.5):.1%}) | "
              f"{(acc_b <= 0.5).mean():.3f} | {(c - rule).mean():+.1%} ({np.percentile(diff_b, 2.5):+.1%}–{np.percentile(diff_b, 97.5):+.1%}) |")


if __name__ == '__main__':
    main(sys.argv[1:])
