"""Match Red Bull BC One World Final battle videos on YouTube to the Wikipedia brackets (ours, Phase 4 extension).

BRACE covers 81 battles from 6 years. The official Red Bull BC One channel has the World Final battles of most years
2004-2025; the English Wikipedia article has every bracket (winner, judge votes, and who danced first via
<sup>1</sup>/<sup>2</sup>). This script

Note: the Wikipedia <sup> order marks are NOT reliable (agree with BRACE's real order for 41/81 battles).

1. reads the channel listing (`yt-dlp --flat-playlist --print '%(id)s\t%(title)s\t%(duration)s'`),
2. keeps battle titles ("A vs B ... Red Bull BC One ... <year>", not cyphers/qualifiers/compilations),
3. matches the two names + year to a bracket match (name keys as in build_battle_labels.key, plus aliases),
4. writes ours/battle/extra_battles.csv: video_id, year, division, round, dancer_a (first name in the title = danced
   first, verified on all 81 BRACE battles),
   dancer_b, winner, winner_is_a, bracket scores, title, source, and whether the video is already in BRACE.

    $MLAVL_PY -I tools/battle/collect_extra_battles.py --listing /data/mlavl/datasets/bcone_extra/channel_videos.tsv
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_battle_labels as B  # noqa: E402

B.YEARS = set(range(2004, 2027))
EXTRA_ALIASES = {'hong10': 'hong10', 'lilg': 'lilg', 'roxrite': 'roxrite', 'neguin': 'neguin'}
SKIP = re.compile(r'cypher|qualif|last chance|camp|top \d+ moves|best of|compilation|highlight|recap|trailer|'
                  r'preview|interview|documentary|reaction|tutorial|judge|showcase|exhibition|all battles|full',
                  re.I)


def names_from_title(title):
    """Return (name1, name2) from a battle title, or None. Prefers the 'vs' segment that names B-Boys/B-Girls
    (titles like 'Cold Style vs Raw Power l B-Boy Issin vs. B-Boy Icey Ives | ...')."""
    parts = re.split(r'\s+[|lI]\s+|\s+-\s+|\s*\|\s*', title)
    segs = [p for p in parts if re.search(r'\s+vs\.?\s+', p, re.I)]
    if not segs:
        return None
    seg = next((p for p in segs if re.search(r'b-?(boy|girl)', p, re.I)), segs[-1])
    a, b = re.split(r'\s+vs\.?\s+', seg, maxsplit=1, flags=re.I)
    a = re.sub(r'^.*[!?:]\s+', '', a)                         # 'Insane Moves! B-Boy X', 'Top 8: Sunny'
    a = re.sub(r'^red bull bc one\s+\d{4}\s+', '', a, flags=re.I)   # 'Red Bull BC One 2004 Ronnie'
    b = re.sub(r':.*$', '', b)                                  # 'Kolobok: Red Bull Bc One 2008 Semifinalist'
    clean = lambda n: re.sub(r'\(.*?\)|^(b-?boys?|b-?girls?)\s+', '', n.strip(), flags=re.I).strip(' .')
    return clean(a), clean(b)


def same(k1, k2):
    """Name keys match exactly, or one contains the other (>= 3 chars), e.g. 'allefthedeep' / 'allef'."""
    return k1 == k2 or (min(len(k1), len(k2)) >= 3 and (k1 in k2 or k2 in k1))


def main(a):
    lst = pd.read_csv(a.listing, sep='\t', header=None, names=['video_id', 'title', 'duration'], dtype=str,
                      quoting=3).dropna(subset=['title'])
    recs = B.parse_wiki(open(a.wiki).read())
    rev = json.load(open(a.revision))
    rev = next(iter(rev['query']['pages'].values()))['revisions'][0]
    brace = set(pd.read_csv(os.path.join(a.brace, 'repo/videos_info.csv')).video_id)
    rows, unmatched = [], []
    for r in lst.itertuples():
        t = r.title
        if 'bc one' not in t.lower() or SKIP.search(t):
            continue
        ym = re.findall(r'\b(20[0-2]\d)\b', t)
        nm = names_from_title(t)
        if not ym or not nm:
            continue
        year = int(ym[-1])
        k1, k2 = B.key(nm[0]), B.key(nm[1])
        cand = [x for x in recs if x['year'] == year and (
            (same(k1, B.key(x['a']['name'])) and same(k2, B.key(x['b']['name']))) or
            (same(k1, B.key(x['b']['name'])) and same(k2, B.key(x['a']['name']))))]
        if not cand:
            unmatched.append((r.video_id, year, nm, t))
            continue
        t_round = B.title_round(t)
        rec = ([x for x in cand if x['wiki_round'] == t_round] or cand)[0]
        # who danced first: the first name in the official YouTube title (true for 81/81 BRACE battles); the
        # Wikipedia <sup>1</sup>/<sup>2</sup> marks agree with BRACE for only 41/81 and are not used
        first, second = (rec['a'], rec['b']) if same(k1, B.key(rec['a']['name'])) else (rec['b'], rec['a'])
        winner = first['name'] if first['bold'] else second['name']
        rows.append({'video_id': r.video_id, 'year': year, 'division': rec['division'],
                     'round': rec['wiki_round'], 'title_round': t_round,
                     'dancer_a': first['name'], 'dancer_b': second['name'], 'winner': winner,
                     'winner_is_a': int(winner == first['name']),
                     'bracket_score_a': first['score'], 'bracket_score_b': second['score'],
                     'duration_s': float(r.duration) if r.duration not in (None, 'NA', '') else float('nan'),
                     'in_brace': int(r.video_id in brace), 'youtube_title': t,
                     'source': B.WIKI_URL.format(rev=rev['revid'])})
    df = pd.DataFrame(rows).drop_duplicates('video_id')
    # one video per bracket match: prefer the full battle upload (longest), drop re-uploads / shorts
    df['match_key'] = df.apply(lambda x: f"{x.year}|{x.division}|{x['round']}|{'|'.join(sorted([B.key(x.dancer_a), B.key(x.dancer_b)]))}", axis=1)
    df = df.sort_values(['in_brace', 'duration_s'], ascending=[False, False]).drop_duplicates('match_key')
    df = df.sort_values(['year', 'division', 'round', 'dancer_a'])
    df.to_csv(a.out, index=False)
    print(f'{len(df)} matched battles ({df.in_brace.sum()} already in BRACE, {(df.in_brace == 0).sum()} new); '
          f'{len(unmatched)} battle-like titles unmatched')
    print(df[df.in_brace == 0].groupby(['year', 'division']).size().to_string())
    with open(a.out.replace('.csv', '_unmatched.tsv'), 'w') as f:
        for u in unmatched:
            f.write('\t'.join(map(str, u)) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--listing', default='/data/mlavl/datasets/bcone_extra/channel_videos.tsv')
    p.add_argument('--wiki', default='/data/mlavl/datasets/brace/meta/redbull_bc_one.wiki')
    p.add_argument('--revision', default='/data/mlavl/datasets/brace/meta/wiki_revision.json')
    p.add_argument('--brace', default='/data/mlavl/datasets/brace')
    p.add_argument('--out', default='ours/battle/extra_battles.csv')
    main(p.parse_args())
