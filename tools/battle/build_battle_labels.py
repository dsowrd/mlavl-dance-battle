"""Build battle-level metadata and winner labels for the 81 BRACE videos (ours, Phase 4).

Each BRACE video is one 1-vs-1 Red Bull BC One battle between two dancers. Winners are not part of BRACE, so they
are parsed from the English Wikipedia "Red Bull BC One" article (brackets for 2011, 2013, 2014, 2017, 2018, 2020) and
cross-checked against the round named in the official Red Bull BC One YouTube title.

    $MLAVL_PY -I tools/battle/build_battle_labels.py \
        --brace /data/mlavl/datasets/brace --meta /data/mlavl/datasets/brace/meta --out ours/battle

Inputs (downloaded once, see ours/battle/README.md):
    {brace}/repo/annotations/sequences.csv, {brace}/repo/videos_info.csv
    {meta}/redbull_bc_one.wiki      wikitext via index.php?title=Red_Bull_BC_One&action=raw
    {meta}/wiki_revision.json       revision id of that wikitext
    {meta}/titles.tsv               video_id, YouTube title, upload date, channel (yt-dlp --print)
Outputs:
    {out}/battles.csv   one row per video: year, division, round, both dancers, sequences, title
    {out}/winners.csv   one row per video: winner, bracket score, parse status, source URL, verified_by (human)
"""
import argparse
import json
import os
import re
import unicodedata

import pandas as pd

WIKI_URL = 'https://en.wikipedia.org/w/index.php?title=Red_Bull_BC_One&oldid={rev}'
YEARS = {2011, 2013, 2014, 2017, 2018, 2020}
ALIASES = {'alcolil': 'alkolil', 'drhill': 'hill'}  # Wikipedia/YouTube spelling -> BRACE dancer name


def key(name):
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'^(b-?boy|b-?girl)\s+', '', s.strip())
    s = re.sub(r'[^a-z0-9]', '', s)
    return ALIASES.get(s, s)


def clean(cell):
    cell = re.sub(r'<sup>.*?</sup>', '', cell)
    cell = re.sub(r'<[^>]+>', '', cell)
    cell = cell.replace('§FLAG', '').replace("'''", '').replace("''", '').replace('*', '')
    return cell.strip()


def normalise(text):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = text.replace('{{!}}', '/')
    text = re.sub(r'\{\{#invoke:flag\|icon\|[^}]*\}\}', '§FLAG ', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    return text


def template_end(text, start):
    depth, i = 0, start
    while i < len(text):
        if text.startswith('{{', i):
            depth += 1
            i += 2
        elif text.startswith('}}', i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return len(text)


def entrant(cell):
    return {'name': clean(cell), 'bold': "'''" in cell}


def matches_round_template(body, kind):
    """Round16 / Round8: entrant cells (with a flag) each followed by a score cell; consecutive entrants pair up."""
    cells = normalise(body).split('|')
    ents = []
    for i, c in enumerate(cells):
        if '§FLAG' in c:
            score = cells[i + 1] if i + 1 < len(cells) else ''
            e = entrant(c)
            e['score'] = clean(score.replace('color=no}}', ''))
            ents.append(e)
    rounds = (['Top 16'] * 8 + ['Quarterfinal'] * 4 if kind == 'Round16' else ['Quarterfinal'] * 4) + ['Semifinal'] * 2 + ['Final']
    return [(rounds[k // 2] if k // 2 < len(rounds) else '?', ents[k], ents[k + 1]) for k in range(0, len(ents) - 1, 2)]


def matches_bracket_module(body):
    body = normalise(body)
    teams, scores = {}, {}
    for m in re.finditer(r'\|\s*RD(\d)-team(\d+)\s*=([^\n]*)', body):
        teams[(int(m.group(1)), int(m.group(2)))] = m.group(3)
    for m in re.finditer(r'\|\s*RD(\d)-score(\d+)\s*=([^\n]*)', body):
        scores[(int(m.group(1)), int(m.group(2)))] = m.group(3)
    names = {1: 'Top 16', 2: 'Quarterfinal', 3: 'Semifinal', 4: 'Final'}
    out = []
    for (rd, idx) in sorted(teams):
        if idx % 2 == 1 and (rd, idx + 1) in teams:
            a, b = entrant(teams[(rd, idx)]), entrant(teams[(rd, idx + 1)])
            a['score'] = clean(scores.get((rd, idx), ''))
            b['score'] = clean(scores.get((rd, idx + 1), ''))
            out.append((names[rd], a, b))
    return out


def parse_wiki(text):
    year_heads = [(m.start(), int(m.group(1))) for m in re.finditer(r'^===\s*(\d{4})\s*===\s*$', text, flags=re.M)]
    div_heads = [(m.start(), m.group(1)) for m in re.finditer(r'^====\s*(.*?)\s*====\s*$', text, flags=re.M)]
    records = []
    for m in re.finditer(r'(?<!<nowiki>)\{\{(Round16|Round8|#invoke:bracket\|16TeamBracket)', text):
        start = m.start()
        years = [y for p, y in year_heads if p < start]
        if not years or years[-1] not in YEARS:
            continue
        divs = [d for p, d in div_heads if p < start and p > max(p2 for p2, y in year_heads if p2 < start)]
        division = 'bgirl' if divs and 'girl' in divs[-1].lower() else 'bboy'
        body = text[start:template_end(text, start)]
        kind = m.group(1)
        ms = matches_bracket_module(body) if kind.startswith('#invoke') else matches_round_template(body, kind)
        for rnd, a, b in ms:
            records.append({'year': years[-1], 'division': division, 'wiki_round': rnd, 'a': a, 'b': b})
    return records


def title_round(title):
    t = title.lower()
    if 'final battle' in t or re.search(r'\|\s*finals?\s*\|?', t) or re.search(r'-\s*final battle', t) or re.search(r'\|\s*final\b', t):
        return 'Final'
    if 'semi' in t:
        return 'Semifinal'
    if 'quarter' in t:
        return 'Quarterfinal'
    if 'top 8' in t:              # Top 8 = quarterfinal of a 16-dancer bracket (2020: first round of an 8-bracket)
        return 'Quarterfinal'
    m = re.search(r'battle (\d+)', t)  # 2011-2014 titles number battles 1-8 (top 16) and 9-12 (quarterfinals)
    if m:
        return 'Top 16' if int(m.group(1)) <= 8 else 'Quarterfinal'
    if 'top 16' in t:
        return 'Top 16'
    return ''


def score_value(s):
    nums = [int(x) for x in re.findall(r'\d+', s)]
    return sum(nums) if nums else None


def main(a):
    seq = pd.read_csv(os.path.join(a.brace, 'repo/annotations/sequences.csv'))
    vinfo = pd.read_csv(os.path.join(a.brace, 'repo/videos_info.csv'))
    titles = pd.read_csv(os.path.join(a.meta, 'titles.tsv'), sep='\t', header=None,
                         names=['video_id', 'title', 'upload_date', 'channel'], dtype=str)
    rev = json.load(open(os.path.join(a.meta, 'wiki_revision.json')))
    rev = next(iter(rev['query']['pages'].values()))['revisions'][0]
    source = WIKI_URL.format(rev=rev['revid'])
    records = parse_wiki(open(os.path.join(a.meta, 'redbull_bc_one.wiki')).read())

    battles, winners = [], []
    for vid, g in seq.sort_values(['video_id', 'seq_idx']).groupby('video_id', sort=False):
        year = int(g.year.iloc[0])
        order = list(dict.fromkeys(g.dancer))           # dancer who danced first = dancer_a
        ids = {d: int(g[g.dancer == d].dancer_id.iloc[0]) for d in order}
        title = titles.set_index('video_id').title.get(vid, '')
        t_round = title_round(title)
        cand = [r for r in records if r['year'] == year and {key(r['a']['name']), key(r['b']['name'])} == {key(d) for d in order}]
        status, note = 'ok', ''
        if not cand:
            status, rec = 'no_bracket_match', None
        else:
            # same pair can meet twice in one year only in theory; prefer the one whose round matches the title
            same = [r for r in cand if r['wiki_round'] == t_round]
            rec = (same or cand)[0]
            if len(cand) > 1:
                note = f'{len(cand)} bracket matches for this pair'
        winner = score_a = score_b = wiki_round = division = ''
        if rec:
            wiki_round, division = rec['wiki_round'], rec['division']
            ea, eb = (rec['a'], rec['b']) if key(rec['a']['name']) == key(order[0]) else (rec['b'], rec['a'])
            score_a, score_b = ea['score'], eb['score']
            if ea['bold'] != eb['bold']:
                winner = order[0] if ea['bold'] else order[1]
            else:
                va, vb = score_value(score_a), score_value(score_b)
                if va is not None and vb is not None and va != vb:
                    winner = order[0] if va > vb else order[1]
                    note = (note + '; ' if note else '') + 'winner from score (bold marks ambiguous)'
                else:
                    status = 'ambiguous'
            if t_round and wiki_round != t_round:
                note = (note + '; ' if note else '') + f'round mismatch: wiki {wiki_round} vs title {t_round}'
        battles.append({
            'video_id': vid, 'year': year, 'division': division or ('bgirl' if 'b-girl' in title.lower() else 'bboy'),
            'round': t_round or wiki_round, 'dancer_a': order[0], 'dancer_a_id': ids[order[0]],
            'dancer_b': order[1] if len(order) > 1 else '', 'dancer_b_id': ids[order[1]] if len(order) > 1 else -1,
            'n_sequences': len(g), 'sequence_uids': ' '.join(g.uid),
            'start_frame': int(g.start_frame.min()), 'end_frame': int(g.end_frame.max()),
            'video_duration_s': float(vinfo.set_index('video_id').duration.get(vid, float('nan'))),
            'youtube_title': title,
        })
        winners.append({
            'video_id': vid, 'year': year, 'round': t_round or wiki_round, 'dancer_a': order[0],
            'dancer_b': order[1] if len(order) > 1 else '', 'winner': winner,
            'winner_is_a': '' if not winner else int(winner == order[0]),
            'bracket_score_a': score_a, 'bracket_score_b': score_b, 'status': status, 'notes': note,
            'source': source, 'source_revision_time': rev['timestamp'], 'verified_by': '',
        })
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(battles).to_csv(os.path.join(a.out, 'battles.csv'), index=False)
    w = pd.DataFrame(winners)
    w.to_csv(os.path.join(a.out, 'winners.csv'), index=False)
    print(w.status.value_counts().to_string())
    print('labelled:', (w.winner != '').sum(), '/', len(w))
    print(w[(w.status != 'ok') | (w.notes != '')][['video_id', 'year', 'round', 'dancer_a', 'dancer_b', 'status', 'notes']].to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--brace', default='/data/mlavl/datasets/brace')
    p.add_argument('--meta', default='/data/mlavl/datasets/brace/meta')
    p.add_argument('--out', default='ours/battle')
    main(p.parse_args())
