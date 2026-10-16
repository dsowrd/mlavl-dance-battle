"""What do real breaking judges decide on? Per-criterion judge votes of the Paris 2024 Olympic breaking events
(ours, Phase 6: expert evidence from the official Results Book, no video).

The WDSF results book (OG2024_BKG_B99_BKG.pdf, https://cdnb.worlddancesport.org/legacy-results/58801/OG2024_BKG_B99_BKG.pdf)
lists, for every round of every battle, each of the 9 judges' overall vote ("Red +3.8%") and the five criteria
(Technique, Vocabulary, Originality, Execution, Musicality) as signed percentages towards red or blue.

    pdftotext -layout OG2024_BKG_B99_BKG.pdf og.txt
    $MLAVL_PY -I tools/battle/olympics_judging.py og.txt

Reports: rounds and judge-votes parsed; how often each criterion favours the athlete the judge voted for overall;
the mean share of the overall margin each criterion carries; inter-judge agreement per criterion (share of judges on
the majority side); the distribution of round vote splits (9-0 ... 5-4). Writes ours/battle/olympics_criteria.csv.
"""
import re
import sys

import numpy as np
import pandas as pd

CRIT = ['Technique', 'Vocabulary', 'Originality', 'Execution', 'Musicality']
CELL = re.compile(r'(Red|Blue)\s*\+(\d+(?:\.\d+)?)%|(?<!\S)-(?!\S)')


def signed(m):
    if m.group(0) == '-':
        return 0.0
    v = float(m.group(2))
    return v if m.group(1) == 'Red' else -v


def parse(path):
    rows, rnd, battle, uid = [], None, None, 0     # battle numbers repeat across B-Boys / B-Girls -> running uid
    for line in open(path, encoding='utf-8', errors='replace'):
        b = re.search(r'Battle\s+(\d+)', line)
        if b and 'Round' not in line:
            battle = int(b.group(1))
        r = re.search(r'^\s*Round\s+(\d+)\s+(Red|Blue)\s+\d+-\d+\s+\((\d)-(\d)\)', line)
        if r:
            if int(r.group(1)) == 1:
                uid += 1
            rnd = (uid, int(r.group(1)), r.group(2), int(r.group(3)), int(r.group(4)))
            continue
        if rnd is None or 'Judge' in line:
            continue
        cells = list(CELL.finditer(line))
        if len(cells) == 6:
            name = line[:line.find(cells[0].group(0))].strip()
            if not name or not name.replace('-', '').replace(' ', '').isalpha():
                continue
            vals = [signed(c) for c in cells]
            rows.append({'battle': rnd[0], 'round': rnd[1], 'round_winner': rnd[2], 'split': f'{rnd[3]}-{rnd[4]}',
                         'judge': name, 'overall': vals[0], **dict(zip(CRIT, vals[1:]))})
    return pd.DataFrame(rows)


def main(path):
    df = parse(path)
    df.to_csv('ours/battle/olympics_criteria.csv', index=False)
    n_rounds = df.groupby(['battle', 'round']).ngroups
    print(f'{n_rounds} rounds, {len(df)} judge votes, {df.judge.nunique()} judges\n')
    side = np.sign(df.overall)
    print('| criterion | favours the judge\'s overall pick | against | neutral | mean share of |overall| | judges on the majority side |')
    print('| --- | --- | --- | --- | --- | --- |')
    for c in CRIT:
        s = np.sign(df[c])
        agree = ((s == side) & (s != 0)).mean()
        against = ((s == -side) & (s != 0)).mean()
        share = (df[c] * side).sum() / df.overall.abs().sum()
        maj = df.groupby(['battle', 'round'])[c].apply(lambda x: max((x > 0).sum(), (x < 0).sum()) / max(1, (x != 0).sum())).mean()
        print(f'| {c} | {agree:.1%} | {against:.1%} | {1 - agree - against:.1%} | {share:.1%} | {maj:.1%} |')
    maj = df.groupby(['battle', 'round']).overall.apply(lambda x: max((x > 0).sum(), (x < 0).sum()) / len(x)).mean()
    print(f'| Overall | | | | 100% | {maj:.1%} |')
    sp = df.groupby(['battle', 'round']).split.first().value_counts(normalize=True).sort_index(ascending=False)
    print('\nround vote splits:', {k: f'{v:.1%}' for k, v in sp.items()})
    # how well does each single criterion, summed over judges, predict the round winner?
    g = df.groupby(['battle', 'round'])
    win = g.round_winner.first().map({'Red': 1, 'Blue': -1})
    for c in CRIT:
        pred = np.sign(g[c].sum())
        print(f'round winner = side favoured by the summed {c} margins: {(pred == win).mean():.1%}')


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main(sys.argv[1])
