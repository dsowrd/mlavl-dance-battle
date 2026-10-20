"""WDSF breaking pre-selection scores (ours, Phase 5c): absolute per-judge scores (0-100) for every athlete's solo
pre-selection performance, from /Competitions/PreSeed/<competition>. Caches the HTML next to the other pages.

    $MLAVL_PY -I tools/battle/wdsf_preseed.py   -> /data/mlavl/datasets/wdsf/wdsf_preselection_scores.csv
"""
import html
import os
import re
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

R = '/data/mlavl/datasets/wdsf'
BASE = 'https://www.worlddancesport.org/Competitions/PreSeed/'


def get(comp):
    path = f'{R}/html/comps/{comp}__PreSeed.html'
    if os.path.exists(path):
        return open(path, encoding='utf-8').read()
    try:
        req = urllib.request.Request(BASE + comp, headers={'User-Agent': 'Mozilla/5.0 (academic research)'})
        t = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        t = f'<!-- {e.code} -->'
    open(path, 'w', encoding='utf-8').write(t)
    time.sleep(1)
    return t


def parse(t):
    t = re.sub(r'<script.*?</script>|<style.*?</style>', '', t, flags=re.S)
    txt = html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', t)))
    i = txt.find('Rank Athlete Country Scores')
    if i < 0:
        return []
    body = txt[i + len('Rank Athlete Country Scores'):]
    # "<rank> <name> <CCC> <avg> <judge> : <score> <judge> : <score> ..." ; judge names may contain spaces
    rows = []
    for m in re.finditer(r'(\d+) (.+?) ([A-Z]{3}) (\d+(?:\.\d+)?) ((?:[^:]+? : \d+(?:\.\d+)? ?)+?)(?=\d+ \D|$)', body):
        judges = re.findall(r'([^:]+?) : (\d+(?:\.\d+)?)', m.group(5))
        rows.append(dict(rank=int(m.group(1)), athlete=m.group(2).strip(), country=m.group(3), score=float(m.group(4)),
                         judges={j.strip(): float(s) for j, s in judges}))
    return rows


def main():
    comps = pd.read_csv(f'{R}/competitions.tsv', sep='\t').drop_duplicates('competition')
    out = []
    for c in comps.itertuples():
        if 'Team' in c.competition:
            continue
        rows = parse(get(c.competition))
        for r in rows:
            for j, s in r['judges'].items():
                out.append(dict(competition=c.competition, month=c.month, rank=r['rank'], athlete=r['athlete'],
                                country=r['country'], score=r['score'], judge=j, judge_score=s))
        print(f'{c.competition}: {len(rows)} athletes', flush=True)
    df = pd.DataFrame(out)
    df.to_csv(f'{R}/wdsf_preselection_scores.csv', index=False)
    a = df.drop_duplicates(['competition', 'athlete'])
    print(f'{len(a)} scored performances in {a.competition.nunique()} competitions, '
          f'{df.groupby(["competition", "athlete"]).judge.nunique().median():.0f} judges median; '
          f'score mean {a.score.mean():.1f} sd {a.score.std():.1f}')


if __name__ == '__main__':
    main()
