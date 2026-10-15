"""Is a single battle's outcome predictable at all? Dancer-history (Elo) baseline (ours, Phase 6 diagnostic).

Uses only who the dancers are and their earlier Red Bull BC One World Final results (Wikipedia brackets 2004-2025,
chronological: year, then Top 16 -> Quarterfinal -> Semifinal -> Final), never the videos. Each match is predicted
with ratings computed from strictly earlier matches. This is NOT a judging model; it tells whether outcomes are
predictable from information about the dancers, i.e. whether a content model has anything to find.

    $MLAVL_PY -I tools/battle/elo_baseline.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_battle_labels as B  # noqa: E402

B.YEARS = set(range(2004, 2027))
ORDER = {'Top 16': 0, 'Quarterfinal': 1, 'Semifinal': 2, 'Final': 3}


def main():
    recs = B.parse_wiki(open('/data/mlavl/datasets/brace/meta/redbull_bc_one.wiki').read())
    recs = sorted(recs, key=lambda r: (r['year'], ORDER.get(r['wiki_round'], 9)))
    rng = np.random.default_rng(0)
    rows = []
    for K in [16, 32, 64]:
        elo, played, wins = {}, {}, {}
        res = []
        for r in recs:
            a, b = B.key(r['a']['name']), B.key(r['b']['name'])
            ya = 1 if r['a']['bold'] else 0
            ra, rb = elo.get(a, 1500.0), elo.get(b, 1500.0)
            pa = 1 / (1 + 10 ** ((rb - ra) / 400))
            known = played.get(a, 0) > 0 and played.get(b, 0) > 0
            res.append({'year': r['year'], 'pa': pa, 'y': ya, 'known': known, 'tie': ra == rb,
                        'wins_diff': wins.get(a, 0) - wins.get(b, 0), 'a': a, 'b': b, 'round': r['wiki_round']})
            elo[a], elo[b] = ra + K * (ya - pa), rb + K * ((1 - ya) - (1 - pa))
            for x, won in ((a, ya), (b, 1 - ya)):
                played[x] = played.get(x, 0) + 1
                wins[x] = wins.get(x, 0) + won
        df = pd.DataFrame(res)
        for name, m in [('all matches', np.ones(len(df), bool)), ('both dancers seen before', df.known.values)]:
            d = df[m & ~df.tie.values]
            c = ((d.pa > 0.5) == (d.y == 1)).astype(float).values
            bt = rng.choice(c, (5000, len(c))).mean(1)
            rows.append((K, name, len(d), c.mean(), np.percentile(bt, 2.5), np.percentile(bt, 97.5)))
        if K == 32:
            d = df[df.known & (df.wins_diff != 0)]
            c = ((d.wins_diff > 0) == (d.y == 1)).astype(float).values
            print(f'more previous World Final wins -> wins: {c.mean():.1%} (n={len(c)})')
            # our video battles (BRACE + extra) only
            vid = pd.concat([pd.read_csv('ours/battle/winners.csv')[['year', 'dancer_a', 'dancer_b']],
                             pd.read_csv('ours/battle/extra_battles.csv').query('in_brace == 0')[['year', 'dancer_a', 'dancer_b']]])
            keys = {(r.year, frozenset([B.key(r.dancer_a), B.key(r.dancer_b)])) for r in vid.itertuples()}
            m = df.apply(lambda r: (r.year, frozenset([r.a, r.b])) in keys, axis=1) & df.known & ~df.tie
            d = df[m]
            c = ((d.pa > 0.5) == (d.y == 1)).astype(float).values
            bt = rng.choice(c, (5000, len(c))).mean(1)
            print(f'Elo K=32 on our video battles with both dancers seen before: {c.mean():.1%} '
                  f'({np.percentile(bt, 2.5):.1%}-{np.percentile(bt, 97.5):.1%}, n={len(c)})')
    print('\n| K | matches | n | Elo accuracy (95% CI) |\n| --- | --- | --- | --- |')
    for K, name, n, acc, lo, hi in rows:
        print(f'| {K} | {name} | {n} | {acc:.1%} ({lo:.1%}-{hi:.1%}) |')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
