"""Compare battle runs against each other and against the training-free rules (ours, Phase 6).

    $MLAVL_PY -I tools/battle/compare_runs.py initRG v1_battle v1_round ...   # tags = logs/battle/{tag}_s{seed}

Per tag: accuracy mean ± std over seeds, AUC, log loss, and for the seed ensemble (mean p over seeds) the accuracy with
a 95% bootstrap CI and the paired bootstrap difference to the first-dancer rule (same battles resampled for both):
mean difference, 95% CI, and the share of resamples where the model is not better (one-sided p).
A result counts as a real improvement over the rule only if the CI of the paired difference is above 0.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd


def main(tags, n=10000):
    rng = np.random.default_rng(0)
    print('| run | seeds | acc mean ± std | AUC | log loss | ensemble acc (95% CI) | Δ vs first-dancer rule (95% CI) | p(Δ≤0) |')
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
        rule = (y == 1).astype(float)                          # first dancer (A) wins
        idx = rng.integers(0, len(y), (n, len(y)))
        acc_b, diff_b = c[idx].mean(1), (c[idx] - rule[idx]).mean(1)
        accs = [s['accuracy'] for s in S]
        print(f"| {t} | {len(S)} | {np.mean(accs):.1%} ± {np.std(accs):.1%} | {np.mean([s['auc'] for s in S]):.3f} | "
              f"{np.mean([s['log_loss'] for s in S]):.3f} | {c.mean():.1%} ({np.percentile(acc_b, 2.5):.1%}–{np.percentile(acc_b, 97.5):.1%}) | "
              f"{(c - rule).mean():+.1%} ({np.percentile(diff_b, 2.5):+.1%}–{np.percentile(diff_b, 97.5):+.1%}) | {(diff_b <= 0).mean():.2f} |")


if __name__ == '__main__':
    main(sys.argv[1:])
