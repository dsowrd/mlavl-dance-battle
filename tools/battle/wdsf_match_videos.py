"""Match WDSF scored battles (wdsf_parse.py) to single-battle YouTube videos (ours, Phase 5c).

Video list: `yt-dlp --flat-playlist --print "%(id)s\t%(duration)s\t%(title)s" <channel>/videos` (one TSV per
channel in <root>/yt/). A video matches a battle when both athletes' names appear in the title (normalised: lower
case, letters and digits only, "b-boy"/"b-girl" prefixes dropped) and the title names the event city and the year
of the competition. Writes <root>/wdsf_battle_videos.csv (one row per matched battle) and prints coverage.

    $MLAVL_PY -I tools/battle/wdsf_match_videos.py --root /data/mlavl/datasets/wdsf
"""
import argparse
import glob
import re

import pandas as pd


def norm(s):
    s = s.lower()
    s = re.sub(r'\b(b-?boy|b-?girl|bboy|bgirl)\b', ' ', s)
    return re.sub(r'[^a-z0-9]', '', s)


def name_tokens(name):
    """'Nicka (LTU)' -> 'nicka'; multi-word names kept joined ('mini ali' -> 'miniali')."""
    return norm(re.sub(r'\([A-Z]{3}\)', '', name))


def comp_meta(comp):
    m = re.match(r'(.*?)-(Adult|Youth)-Breaking-1-vs-1-(B-Boys|B-Girls)-(\d+)$', comp)
    head = m.group(1) if m else comp
    city = head.split('-')[-1].lower() if m else ''
    # multi-word cities ("Santiago-de-Chile", "Rio-de-Janerio", "Hong-Kong", "Villa-de-Leyva", "Fukuoka-Kurume-City")
    for multi in ('santiago-de-chile', 'rio-de-janerio', 'hong-kong', 'villa-de-leyva', 'fukuoka-kurume-city'):
        if head.lower().endswith(multi):
            city = multi
    return city, (m.group(2) if m else ''), (m.group(3) if m else '')


CITY_ALIASES = {'fukuoka-kurume-city': ['kurume', 'fukuoka'], 'rio-de-janerio': ['rio', 'brazil', 'brasil'],
                'hong-kong': ['hongkong'], 'santiago-de-chile': ['santiago', 'chile'],
                'villa-de-leyva': ['villadeleyva', 'leyva'], 'kitakyushu': ['kitakyushu', 'japan'],
                'chengdu': ['chengdu'], 'nagoya': ['nagoya', 'aichi']}
EVENT_KIND = {'European-Championship': ['european'], 'Asian-Championship': ['asianbreakingchampionship'],
              'Asian-Games': ['asiangames'], 'Pan-American': ['panam'], 'African-Championship': ['african'],
              'Oceania-Championship': ['oceania'], 'IWGA-The-World-Games': ['worldgames'],
              'World-Championships': ['worldchampionship']}


def main(a):
    js = pd.read_csv(f'{a.root}/wdsf_judge_scores.csv')
    months = pd.read_csv(f'{a.root}/competitions.tsv', sep='\t').drop_duplicates('competition')
    js = js.merge(months[['competition', 'month']], on='competition', how='left')
    battles = js.groupby(['competition', 'battle_id']).agg(
        stage=('stage', 'first'), blue=('blue', 'first'), red=('red', 'first'), month=('month', 'first'),
        system=('system', 'first'), has_criteria=('has_criteria', 'max'), n_rounds=('round', 'nunique')).reset_index()
    vids = pd.concat([pd.read_csv(p, sep='\t', names=['video_id', 'duration', 'title'], quoting=3)
                      .assign(channel=p.split('/')[-1][:-4]) for p in glob.glob(f'{a.root}/yt/*.tsv')])
    vids = vids[vids.title.str.contains(r'\bvs\.?\b', case=False, regex=True)].drop_duplicates('video_id')
    vids['nt'] = vids.title.map(norm)
    rows = []
    for _, b in battles.iterrows():
        city, age, div = comp_meta(b.competition)
        year = str(b.month)[:4]
        nb, nr = name_tokens(b.blue), name_tokens(b.red)
        if not nb or not nr:
            continue
        keys = [norm(c) for c in CITY_ALIASES.get(city, [city])]
        for kind, alias in EVENT_KIND.items():           # titles often name the event, not the city
            if b.competition.startswith(kind):
                keys += alias
        cand = vids[vids.nt.str.contains(nb, regex=False) & vids.nt.str.contains(nr, regex=False)]
        cand = cand[cand.nt.map(lambda t: any(k in t for k in keys)) & cand.nt.str.contains(year[2:])]
        if age == 'Youth':
            cand = cand[cand.nt.str.contains('youth')]
        rows.append({**b.to_dict(), 'n_candidates': len(cand),
                     'video_id': cand.video_id.iloc[0] if len(cand) == 1 else '',
                     'video_title': cand.title.iloc[0] if len(cand) == 1 else '',
                     'duration': cand.duration.iloc[0] if len(cand) == 1 else None})
    out = pd.DataFrame(rows)
    out.to_csv(f'{a.root}/wdsf_battle_videos.csv', index=False)
    m = out[out.video_id != '']
    print(f'battles {len(out)}, matched to exactly one video {len(m)}, ambiguous {(out.n_candidates > 1).sum()}')
    print(f'videos used {m.video_id.nunique()} of {len(vids)} battle-titled videos')
    print(f'matched with 5 criteria: {((m.system == "5crit") & m.has_criteria).sum()} battles, '
          f'{m[(m.system == "5crit") & m.has_criteria].n_rounds.sum()} rounds')
    print(m.groupby('competition').size().to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/data/mlavl/datasets/wdsf')
    main(p.parse_args())
