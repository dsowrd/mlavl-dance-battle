"""Zero-shot: score each dancer with the reproduced AQA models as they are, compare scores (ours, Phase 6 baseline).

No training on battles. Every reproduction checkpoint (RG Ball/Clubs/Hoop/Ribbon and Fis-V TES/PCS, seeds 0-2,
`ckpt/*_best.pkl`) is loaded with its own configuration (vocabulary, n_text, dropout, clip number) and scores each
dancer's own turns in each battle exactly as it scores a routine: same 1024-d Video Swin + 768-d AST clip features,
output = the model's score. A dancer's sequence up to 136 clips (MLAVL's positional-encoding limit) is scored in one
pass at its real length; longer ones are scored on windows of the model's training clip number and averaged.
Prediction: the dancer with the higher score wins.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY tools/battle/zero_shot_scores.py
Writes /data/mlavl/features/BRACE/zero_shot_scores_<clips>.csv (one row per battle and checkpoint).
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import options  # noqa: E402
from models.model import MLAVL  # noqa: E402
from ours.battle_dataset import load_battles, load_extra_battles, windows  # noqa: E402

CFG = {'rg': dict(n_text=32, text='./RG_text_feature.npy', csv='./action-label_RG.csv', dropout=0.3, clip=68),
       'fisv': dict(n_text=42, text='./FS_text_feature.npy', csv='./action-label_FS.csv', dropout=0.7, clip=124)}


def load(ckpt, device):
    fam = 'rg' if os.path.basename(ckpt).startswith('rg_') else 'fisv'
    c = CFG[fam]
    a = options.parser.parse_args(['--dataset', 'rg' if fam == 'rg' else 'fisv', '--n_text', str(c['n_text']),
                                   '--text_feature', c['text'], '--clip-num', str(c['clip']), '--in_dim', '1024'])
    names = pd.read_csv(c['csv'])['name'].tolist()
    m = MLAVL(1024, 256, 8, 1, 3, 6, c['dropout'], a, names).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m.eval(), c['clip']


@torch.no_grad()
def score(m, v, a, clip, device):
    if len(v) <= 136:
        x, y = torch.from_numpy(v)[None].float().to(device), torch.from_numpy(a)[None].float().to(device)
        return m(x, y)['output'].item()
    ws = windows(v, a, clip)
    x = torch.from_numpy(np.stack([w[0] for w in ws])).float().to(device)
    y = torch.from_numpy(np.stack([w[1] for w in ws])).float().to(device)
    return m(x, y)['output'].mean().item()


def main(clips='clips'):
    device = torch.device('cuda')
    brace = load_battles('/data/mlavl/features/BRACE', '/data/mlavl/datasets/brace', 'ours/battle', clips=clips)
    for it in brace:
        it['source'] = 'brace'
    extra_all, _ = load_extra_battles('/data/mlavl/features/BCONE_EXTRA', 'ours/battle/extra_battles.csv', min_consistency=0,
                                      clips=clips)
    cons = {i['video_id'] for i in load_extra_battles('/data/mlavl/features/BCONE_EXTRA', 'ours/battle/extra_battles.csv',
                                                      min_consistency=0.8)[0]}
    items = brace + extra_all
    ckpts = sorted(glob.glob('ckpt/rg_*_best.pkl') + glob.glob('ckpt/fisv_*_best.pkl'))
    ckpts = [c for c in ckpts if re.match(r'ckpt/(rg_(Ball|Clubs|Hoop|Ribbon)|fisv_(TES|PCS))_(s[12]_)?1011_\d+_best\.pkl$', c)]
    print(len(items), 'battles;', len(ckpts), 'checkpoints:', [os.path.basename(c) for c in ckpts])
    rows = []
    for ck in ckpts:
        m, clip = load(ck, device)
        tag = re.match(r'ckpt/(rg_\w+?|fisv_\w+?)_(?:s\d_)?1011', ck).group(1)
        seed = int(re.search(r'_s(\d)_', ck).group(1)) if re.search(r'_s(\d)_', ck) else 0
        for it in items:
            sa = score(m, it['rgb_a'], it['audio_a'], clip, device)
            sb = score(m, it['rgb_b'], it['audio_b'], clip, device)
            rows.append({'video_id': it['video_id'], 'source': it.get('source', 'brace'), 'year': it['year'],
                         'y': it['y'], 'votes': it.get('votes', -1), 'consistent': it.get('source', 'brace') == 'brace' or it['video_id'] in cons,
                         'ckpt': tag, 'seed': seed, 's_a': sa, 's_b': sb})
        print('done', os.path.basename(ck), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'/data/mlavl/features/BRACE/zero_shot_scores_{clips}.csv', index=False)
    report(df)


def report(df):
    rng = np.random.default_rng(0)

    def acc(d):
        c = ((d.s_a - d.s_b > 0) == (d.y == 1)).astype(float).values
        b = rng.choice(c, (5000, len(c))).mean(1)
        return f'{c.mean():.1%} ({np.percentile(b, 2.5):.1%}-{np.percentile(b, 97.5):.1%})'

    def ens(d, keys):
        # average of per-checkpoint standardised score differences
        d = d[d.ckpt.isin(keys)].copy()
        d['diff'] = d.s_a - d.s_b
        d['z'] = d.groupby(['ckpt', 'seed'])['diff'].transform(lambda x: x / (x.std() + 1e-9))
        e = d.groupby('video_id').agg(z=('z', 'mean'), y=('y', 'first'), source=('source', 'first'),
                                      consistent=('consistent', 'first'), votes=('votes', 'first')).reset_index()
        return e.assign(s_a=e.z, s_b=0.0)

    subsets = {'all 280': lambda d: d, '170 (BRACE + consistent extra)': lambda d: d[d.consistent],
               'BRACE 81': lambda d: d[d.source == 'brace'], 'unanimous (170)': lambda d: d[d.consistent & (d.votes == 5)]}
    print('\n| scorer | ' + ' | '.join(subsets) + ' |')
    print('| --- | ' + ' | '.join('---' for _ in subsets) + ' |')
    for (ck, seed), d in df.groupby(['ckpt', 'seed']):
        print(f'| {ck} s{seed} | ' + ' | '.join(acc(f(d)) for f in subsets.values()) + ' |')
    fams = {'RG ensemble (12)': [c for c in df.ckpt.unique() if c.startswith('rg_')],
            'Fis-V TES ensemble (3)': ['fisv_TES'], 'Fis-V PCS ensemble (3)': ['fisv_PCS'],
            'all 18 ensemble': list(df.ckpt.unique())}
    for name, keys in fams.items():
        e = ens(df, keys)
        print(f'| **{name}** | ' + ' | '.join(acc(f(e)) for f in subsets.values()) + ' |')


if __name__ == '__main__':
    # usage: zero_shot_scores.py [clips_subdir]   |   zero_shot_scores.py report [clips_subdir]
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report(pd.read_csv(f"/data/mlavl/features/BRACE/zero_shot_scores_{sys.argv[2] if len(sys.argv) > 2 else 'clips'}.csv"))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else 'clips')
