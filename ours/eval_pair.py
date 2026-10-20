"""Strict evaluation of round-level predictions on the WDSF battles (ours, Phase 5c).

Input: one or more prediction files (logs/wdsf/<run>/predictions.csv with competition, battle_id, round, margin, pred;
pairwise runs) whose predictions are averaged (seed ensemble). For every claim we report what would make it wrong:
  - round accuracy with a 95% bootstrap CI, resampling whole battles (rounds of a battle are not independent);
  - one-sided p-values against 50% and against the second-dancer rule (blue wins: 54.4% of rounds), by a
    battle-level bootstrap of the accuracy difference against that rule on the same rounds;
  - subsets: hand-labelled turns (no segmentation error), athletes never seen in training folds (no identity
    shortcut), per event;
  - the athlete-strength reference: mean margin of each athlete at OTHER events (63.6% overall), which uses
    information outside the video and is reported for context, not as a bar the video model must clear.

    $MLAVL_PY ours/eval_pair.py logs/wdsf/pair_total_s0 logs/wdsf/pair_total_s1 logs/wdsf/pair_total_s2
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

D = '/data/mlavl/datasets/wdsf'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ours.train_wdsf import folds, build_rows  # noqa: E402


def load(runs):
    dfs = [pd.read_csv(os.path.join(r, 'predictions.csv')) for r in runs]
    key = ['competition', 'battle_id', 'round']
    df = dfs[0][key + ['event', 'source', 'margin']].copy()
    df['pred'] = np.mean([d.set_index(key).loc[df.set_index(key).index, 'pred'].values for d in dfs], 0)
    return df


def boot(df, stat, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g.index.values for _, g in df.groupby(['competition', 'battle_id'])]
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(stat(df.loc[idx]))
    return np.array(vals)


def report(name, df):
    ok = (np.sign(df.pred) == np.sign(df.margin)).astype(float)
    rule = (df.margin > 0).astype(float)
    d = df.assign(ok=ok, rule=rule)
    acc = d.ok.mean()
    b_acc = boot(d, lambda x: x.ok.mean())
    b_diff = boot(d, lambda x: x.ok.mean() - x.rule.mean())
    lo, hi = np.percentile(b_acc, [2.5, 97.5])
    p50 = (b_acc <= 0.5).mean()
    prule = (b_diff <= 0).mean()
    print(f'{name:38s} n={len(d):4d} battles={d.groupby(["competition", "battle_id"]).ngroups:3d}  '
          f'acc {acc:.3f} [{lo:.3f}, {hi:.3f}]  rule {d.rule.mean():.3f}  '
          f'p(acc<=50%) {p50:.4f}  p(acc<=rule) {prule:.4f}')
    return dict(subset=name, n=len(d), acc=acc, ci_lo=lo, ci_hi=hi, rule=d.rule.mean(), p50=p50, p_rule=prule)


def unseen_athletes(df):
    """Rounds whose two athletes never appear in the rounds of the other folds (the training data)."""
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')[['competition', 'battle_id', 'round', 'blue_id', 'red_id']]
    js = js.drop_duplicates(['competition', 'battle_id', 'round'])
    d = df.merge(js, on=['competition', 'battle_id', 'round'], how='left')
    assign = folds(build_rows('total'))
    d['fold'] = d.event.map(assign)
    keep = []
    for i, r in d.iterrows():
        other = d[d.fold != r.fold]
        seen = set(other.blue_id) | set(other.red_id)
        keep.append(r.blue_id not in seen and r.red_id not in seen)
    return df[np.array(keep)]


def main(runs):
    df = load(runs)
    df = df[df.margin != 0].reset_index(drop=True)
    print('runs:', ', '.join(os.path.basename(r) for r in runs))
    rows = [report('all rounds', df),
            report('hand-labelled turns', df[df.source == 'hand'].reset_index(drop=True)),
            report('auto-segmented turns', df[df.source != 'hand'].reset_index(drop=True))]
    un = unseen_athletes(df).reset_index(drop=True)
    if len(un):
        rows.append(report('both athletes unseen in training', un))
    for ev, g in df.groupby('event'):
        rows.append(report('  ' + ev[:36], g.reset_index(drop=True)))
    out = os.path.join(runs[0], 'strict_eval' + ('_ens%d' % len(runs) if len(runs) > 1 else '') + '.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1:])
