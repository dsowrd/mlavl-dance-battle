"""Collect WDSF breaking competitions with per-round, per-judge criterion scores (ours, Phase 5c: score regression).

WDSF publishes, for every battle of its breaking events (Breaking for Gold series, continental / world championships,
Olympic Qualifier Series), each judge's overall percentage and the five criteria (Technique, Vocabulary, Originality,
Execution, Musicality) on the competition pages (/Competitions/FinalBracket/..., /Competitions/RoundRobin/...).
This script only downloads and caches the HTML (1 request / s); parsing is in wdsf_parse.py.

    $MLAVL_PY -I tools/battle/wdsf_scrape.py --out /data/mlavl/datasets/wdsf --from 2021-01 --to 2026-10
"""
import argparse
import os
import re
import time
import urllib.error
import urllib.request

BASE = 'https://www.worlddancesport.org'
UA = {'User-Agent': 'Mozilla/5.0 (academic research)'}


def get(url, path, delay=1.0):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, encoding='utf-8').read()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                html = r.read().decode('utf-8', 'replace')
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:                        # page type not used by this competition; remember it
                open(path, 'w').write('<!-- 404 -->')
                return ''
            print('retry', url, e, flush=True)
            time.sleep(5 * (attempt + 1))
        except Exception as e:                       # noqa: BLE001 - retry any other network error
            print('retry', url, e, flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        return ''
    open(path, 'w', encoding='utf-8').write(html)
    time.sleep(delay)
    return html


def main(a):
    for d in ('months', 'events', 'comps'):
        os.makedirs(f'{a.out}/html/{d}', exist_ok=True)
    y, m = map(int, a.date_from.split('-'))
    y1, m1 = map(int, a.date_to.split('-'))
    events = {}
    while (y, m) <= (y1, m1):
        h = get(f'{BASE}/Calendar/Results?DisciplineIds%5B0%5D=104&Month={m}&Year={y}',
                f'{a.out}/html/months/{y}-{m:02d}.html')
        for slug in re.findall(r'href="/Events/([^"#]+)#tab=competitions"', h):
            events[slug] = f'{y}-{m:02d}'
        m += 1
        if m == 13:
            y, m = y + 1, 1
    print(len(events), 'events', flush=True)
    comps = []
    for slug, ym in sorted(events.items(), key=lambda kv: kv[1]):
        h = get(f'{BASE}/Events/{slug}', f'{a.out}/html/events/{slug}.html')
        for c in sorted(set(re.findall(r'href="/Competitions/\w+/([^"]*Breaking[^"]*-\d+)"', h))):
            comps.append((ym, slug, c))
    print(len(comps), 'breaking competitions', flush=True)
    with open(f'{a.out}/competitions.tsv', 'w') as f:
        f.write('month\tevent\tcompetition\n')
        for ym, slug, c in comps:
            f.write(f'{ym}\t{slug}\t{c}\n')
    for i, (ym, slug, c) in enumerate(comps):
        for page in ('FinalBracket', 'RoundRobin'):
            get(f'{BASE}/Competitions/{page}/{c}', f'{a.out}/html/comps/{c}__{page}.html')
        if i % 20 == 0:
            print(i, ym, c, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='/data/mlavl/datasets/wdsf')
    p.add_argument('--from', dest='date_from', default='2021-01')
    p.add_argument('--to', dest='date_to', default='2026-10')
    main(p.parse_args())
