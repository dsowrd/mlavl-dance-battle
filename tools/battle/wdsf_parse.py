"""Parse the WDSF breaking pages cached by wdsf_scrape.py into one row per (battle, round, judge) (ours, Phase 5c).

Page structure (checked on European Championship Almeria 2023, B-Girls, 59552): every battle is a
`battleresult` block with the blue athlete first, then red; every round has a title "Round k - (8:1) - 1:0" and one
block per judge with the overall percentage, a class `winner_1` / `winner_2` and five faders. A fader is
`score-bar-1` or `score-bar-2` with its value; criteria at 0 are left out of the page. Side 2 = blue, side 1 = red
(the 8:1 winner Nicka is blue and gets `winner_2` from 8 judges); checked on every round below (votes vs title).

Signs in the output: positive = towards the blue athlete.

    $MLAVL_PY -I tools/battle/wdsf_parse.py --root /data/mlavl/datasets/wdsf
"""
import argparse
import glob
import html as htmlmod
import os
import re

import pandas as pd

CRIT = ['Technique', 'Vocabulary', 'Originality', 'Execution', 'Musicality']               # 2023-
CRIT6 = ['Technique', 'Variety', 'Performativity', 'Musicality', 'Creativity', 'Personality']  # 2022 (older system)


def pct(s):
    s = s.strip().replace('%', '').replace('+', '')
    return float(s) if s else 0.0


def parse_page(text, comp, page):
    rows, stages = [], {}
    # stage name of each battle id from the battle list ("Top 8", "Final", "Group A", ...)
    for blk in re.split(r'<h4 class="group-title">', text)[1:]:
        stage = htmlmod.unescape(blk[:blk.find('<')]).strip()
        for bid in re.findall(r'data-id="([^"]+)" class="js-toggle-battle"', blk[:blk.find('</ul>')]):
            stages.setdefault(bid, stage)
    for bi, blk in enumerate(re.split(r'<div class="battleresult d-none js-group-result"', text)[1:]):
        bid = re.search(r'data-id="([^"]+)"', blk).group(1)
        ath = {c: (u, n) for c, u, n in re.findall(
            r'athlete-(blue|red)">.*?class="athlete-name" href="/Athletes/([^"]+)">([^<]*)<', blk, re.S)}
        if 'blue' not in ath or 'red' not in ath:
            continue
        for rblk in re.split(r'<h5 class="round-title">', blk)[1:]:
            title = re.sub(r'<[^>]+>', '', rblk[:rblk.find('</h5>')])
            m = re.search(r'Round\s+(\d+)\s*-\s*\((\d+):(\d+)\)', title)
            if not m:
                continue
            rnd, v_a, v_b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            for j in re.split(r'<h6 class="judge-name">', rblk)[1:]:
                name = re.search(r'>([^<]+)</a>|^([^<]+)<', j)
                name = htmlmod.unescape((name.group(1) or name.group(2)).strip()) if name else ''
                tot = re.search(r'<p>([^<]*)</p>', j)
                win = re.search(r'class="scores winner_(\d)"', j)
                side = int(win.group(1)) if win else 0
                sign = {2: 1.0, 1: -1.0}.get(side, 0.0)
                row = {'competition': comp, 'page': page, 'battle_id': bid, 'battle_idx': bi,
                       'stage': stages.get(bid, ''), 'blue': htmlmod.unescape(ath['blue'][1]).strip(),
                       'red': htmlmod.unescape(ath['red'][1]).strip(), 'blue_id': ath['blue'][0],
                       'red_id': ath['red'][0], 'round': rnd, 'title_votes_1': v_a, 'title_votes_2': v_b,
                       'judge': name, 'judge_side': side,
                       'total': sign * pct(tot.group(1)) if tot else float('nan')}
                faders = re.findall(r'fader-name">([^<]*)</h6>\s*<div class="score-bar">\s*'
                                    r'(?:<div class="score-bar-(\d)"[^>]*>\s*([^<]*)<)?', j)
                row['system'] = '5crit' if {f[0] for f in faders} <= set(CRIT) else '6crit'
                for c, side_c, v in faders:
                    row[c] = (1.0 if side_c == '2' else -1.0) * pct(v) if side_c else 0.0
                extras = re.findall(r'<div class="(athlete[12])">([^<]+)</div>', j)
                row['extras'] = ';'.join(f'{a}:{t.strip()}' for a, t in extras if t.strip())
                rows.append(row)
    return rows


def main(a):
    rows = []
    for path in sorted(glob.glob(f'{a.root}/html/comps/*.html')):
        comp, page = os.path.basename(path)[:-5].split('__')
        rows += parse_page(open(path, encoding='utf-8').read(), comp, page)
    df = pd.DataFrame(rows)
    if df.empty:
        print('no rows')
        return
    df = df.drop_duplicates(['competition', 'battle_id', 'round', 'judge'])
    crit_cols = [c for c in df.columns if c in CRIT or c in CRIT6]
    df[crit_cols] = df[crit_cols].fillna(0.0)
    df['has_criteria'] = df[crit_cols].abs().sum(1) > 0     # Challenge Series etc. publish only the total
    df.to_csv(f'{a.root}/wdsf_judge_scores.csv', index=False)
    crit_sum = df[crit_cols].sum(1)
    print(f'{len(df)} judge rows, {df.groupby(["competition", "battle_id", "round"]).ngroups} rounds, '
          f'{df.groupby(["competition", "battle_id"]).ngroups} battles, {df.competition.nunique()} competitions')
    for (sys_, has), d in df.groupby(['system', 'has_criteria']):
        print(f'  {sys_} criteria={has}: {d.competition.nunique()} competitions, '
              f'{d.groupby(["competition", "battle_id", "round"]).ngroups} rounds, '
              f'criteria sum == total {(abs(crit_sum[d.index] - d.total) < 0.11).mean():.3f}')
    # side check: judges voting side 2 vs the title's vote count
    g = df.groupby(['competition', 'battle_id', 'round']).agg(n2=('judge_side', lambda s: (s == 2).sum()),
                                                              n1=('judge_side', lambda s: (s == 1).sum()),
                                                              t1=('title_votes_1', 'first'),
                                                              t2=('title_votes_2', 'first'))
    print('title (a:b) == (side2:side1):', f'{((g.n2 == g.t1) & (g.n1 == g.t2)).mean():.3f}',
          ' == (side1:side2):', f'{((g.n1 == g.t1) & (g.n2 == g.t2)).mean():.3f}')
    print('extras seen:', df.extras[df.extras != ''].str.split(';').explode().str.split(':').str[1]
          .value_counts().head(10).to_dict())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/data/mlavl/datasets/wdsf')
    main(p.parse_args())
