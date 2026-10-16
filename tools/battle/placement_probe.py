"""MLAVL-style score regression for breaking: predict a dancer's tournament placement from one battle performance
(ours, Phase 6). Linear probe before training MLAVL.

Score target (per dancer, year, division; from the Wikipedia brackets): 5 champion, 4 runner-up, 3 semifinal,
2 quarterfinal, 1 top-16 exit. One sample = one dancer's own turns in one battle video (BRACE human turns; extra
battles: automatic turns with the appearance-consistency filter).

Confound: the battle's round alone predicts placement (a finalist is >= 4). Reported:
  rho_all          Spearman over all samples (pooled out-of-fold predictions)
  rho_round_only   the same for a predictor that only knows the battle's round
  rho_within_round mean Spearman inside each (year, division, round) group (round held fixed), size-weighted
  rho_top16        Spearman on Top-16 battles only (every dancer once, same round)
  pair_acc         within-battle: higher predicted score = battle winner
CV: leave-one-year-out, and grouped by dancer (no dancer in both train and test).

    $MLAVL_PY -I tools/battle/placement_probe.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'battle'))
import build_battle_labels as B  # noqa: E402
from ours.battle_dataset import load_battles, load_extra_battles  # noqa: E402

B.YEARS = set(range(2004, 2027))
RANK = {'Top 16': 1, 'Quarterfinal': 2, 'Semifinal': 3, 'Final': 4}


def placements():
    recs = B.parse_wiki(open('/data/mlavl/datasets/brace/meta/redbull_bc_one.wiki').read())
    best = {}
    for r in recs:
        k = RANK.get(r['wiki_round'])
        if k is None:
            continue
        for e, won in ((r['a'], r['a']['bold']), (r['b'], r['b']['bold'])):
            key = (r['year'], r['division'], B.key(e['name']))
            best[key] = max(best.get(key, 0), k + (1 if (won and k == 4) else 0))   # champion = 5
    return best


def samples(clips='clips'):
    brace = load_battles('/data/mlavl/features/BRACE', '/data/mlavl/datasets/brace', 'ours/battle', clips=clips)
    for it in brace:
        it['source'] = 'brace'
    extra, _ = load_extra_battles('/data/mlavl/features/BCONE_EXTRA', 'ours/battle/extra_battles.csv',
                                  min_consistency=0.8, clips=clips)
    div = pd.concat([pd.read_csv('ours/battle/battles.csv')[['video_id', 'division']],
                     pd.read_csv('ours/battle/extra_battles.csv')[['video_id', 'division']]]).drop_duplicates('video_id')
    div = dict(zip(div.video_id, div.division))
    best = placements()
    rows = []
    for it in brace + extra:
        for side, name in (('a', it['dancer_a']), ('b', it['dancer_b'])):
            p = best.get((it['year'], div.get(it['video_id'], 'bboy'), B.key(name)))
            if p is None:
                continue
            v, a = it['rgb_' + side], it['audio_' + side]
            rows.append({'video_id': it['video_id'], 'year': it['year'], 'round': it['round'], 'side': side,
                         'dancer': B.key(name), 'won': int(it['y'] == (side == 'a')), 'placement': p,
                         'source': it['source'], 'x': np.concatenate([v.mean(0), v.std(0), a.mean(0), a.std(0)]),
                         'xv': np.concatenate([v.mean(0), v.std(0)]), 'xa': np.concatenate([a.mean(0), a.std(0)])})
    return pd.DataFrame(rows)


def fit_predict(X, y, tr, te, k, alpha):
    sc = StandardScaler().fit(X[tr])
    pca = PCA(min(k, tr.sum() - 1), random_state=0).fit(sc.transform(X[tr]))
    m = Ridge(alpha=alpha).fit(pca.transform(sc.transform(X[tr])), y[tr])
    return m.predict(pca.transform(sc.transform(X[te])))


def metrics(df, pred):
    d = df.assign(pred=pred)
    out = {'rho_all': spearmanr(d.pred, d.placement)[0],
           'rho_round_only': spearmanr(d['round'].map(RANK), d.placement)[0]}
    ws, rs = [], []
    for _, g in d.groupby(['year', 'round']):
        if len(g) >= 4 and g.placement.nunique() > 1:
            rs.append(spearmanr(g.pred, g.placement)[0])
            ws.append(len(g))
    out['rho_within_round'] = float(np.average(rs, weights=ws)) if rs else np.nan
    t = d[d['round'] == 'Top 16']
    out['rho_top16'] = spearmanr(t.pred, t.placement)[0] if len(t) > 5 else np.nan
    pa = []
    for _, g in d.groupby('video_id'):
        if len(g) == 2:
            pa.append(float(g.sort_values('pred').won.iloc[-1] == 1))
    out['pair_acc'] = float(np.mean(pa))
    out['n'] = len(d)
    return out


def main():
    os.chdir(ROOT)
    df = samples()
    print(f"{len(df)} dancer-performances from {df.video_id.nunique()} battles; placement counts "
          f"{df.placement.value_counts().sort_index().to_dict()}; Top-16 samples {(df['round'] == 'Top 16').sum()}\n")
    y = df.placement.values.astype(float)
    rng = np.random.default_rng(0)
    print('| features | CV | PCA k | rho_all | rho_round_only | rho_within_round | rho_top16 | pair_acc |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for fname in ['xv', 'xa', 'x']:
        X = np.stack(df[fname].values)
        for cv in ['year', 'dancer']:
            for k in [16, 64]:
                pred = np.zeros(len(df))
                if cv == 'year':
                    splits = [(df.year.values != t, df.year.values == t) for t in df.year.unique()]
                else:
                    splits = [(np.isin(np.arange(len(df)), tr), np.isin(np.arange(len(df)), te))
                              for tr, te in GroupKFold(5).split(X, y, df.dancer.values)]
                for tr, te in splits:
                    pred[te] = fit_predict(X, y, tr, te, k, alpha=10.0)
                m = metrics(df, pred)
                print(f"| {fname} | {cv} | {k} | {m['rho_all']:.3f} | {m['rho_round_only']:.3f} | {m['rho_within_round']:.3f} | "
                      f"{m['rho_top16']:.3f} | {m['pair_acc']:.1%} |")
    # permutation reference for rho_top16 (null distribution under shuffled placements within Top-16)
    t = df[df['round'] == 'Top 16']
    null = [spearmanr(rng.permutation(t.placement.values), t.placement.values)[0] for _ in range(2000)]
    print(f"\nnull rho_top16 (shuffled), 95% interval: {np.percentile(null, 2.5):.3f} .. {np.percentile(null, 97.5):.3f} (n={len(t)})")


if __name__ == '__main__':
    main()
