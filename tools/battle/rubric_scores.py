"""Expert-rubric scoring of battles without training on battle outcomes (ours, Phase 6).

Criteria and weights come from real judges: the Paris 2024 Olympic results book (tools/battle/olympics_judging.py):
share of the judges' overall margin carried by each criterion = Execution 25.2%, Vocabulary 22.5%, Technique 19.4%,
Musicality 17.9%, Originality 15.0%. Each criterion is measured per dancer (over the dancer's own turns) by observable
cues, standardised over all dancers, and averaged within the criterion:

  Execution   CLIP: clean / controlled vs crash / falling prompts on the dancer crops; pose: -jerk
  Vocabulary  CLIP: number and entropy of distinct breaking moves (36 move prompts, top-1 per clip); pose: pose-cluster entropy
  Technique   CLIP: difficult power moves / freezes vs basic steps; pose: spin p90, inverted share, inverted freezes
  Musicality  pose + music: kick-beat BAS, freeze onsets on the beat
  Originality CLIP: distance of the dancer's clip embeddings from the population mean (rarity)

Prediction: dancer with the larger weighted rubric score wins. Nothing is fitted on our winners (weights and prompts
are fixed beforehand); a leave-one-year-out logistic combination of the 5 criterion differences is reported separately.

    HF_HOME=$MLAVL_DATA/cache/hf $MLAVL_PY -I tools/battle/rubric_scores.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WEIGHTS = {'Execution': 0.252, 'Vocabulary': 0.225, 'Technique': 0.194, 'Musicality': 0.179, 'Originality': 0.150}
PROMPTS = {
    'exec_pos': ['a breakdancer holding a clean, controlled freeze', 'a b-boy landing a power move cleanly',
                 'a breakdancer moving with precise control'],
    'exec_neg': ['a breakdancer losing balance and falling on the floor', 'a b-boy crashing out of a power move',
                 'a breakdancer stumbling clumsily'],
    'tech_pos': ['a b-boy doing a difficult airflare power move', 'a breakdancer spinning on his head',
                 'a b-boy in a one-handed freeze', 'a breakdancer doing complex fast footwork on the floor'],
    'tech_neg': ['a person doing simple dance steps while standing', 'a person walking on a stage',
                 'a person standing still'],
}
SRC = {'brace': '/data/mlavl/features/BRACE', 'extra': '/data/mlavl/features/BCONE_EXTRA'}


@torch.no_grad()
def text_emb(sentences):
    from transformers import CLIPModel, CLIPTokenizer
    name = 'openai/clip-vit-base-patch16'
    tok, m = CLIPTokenizer.from_pretrained(name), CLIPModel.from_pretrained(name).eval()
    e = m.get_text_features(**tok(sentences, padding=True, return_tensors='pt'))
    return torch.nn.functional.normalize(e, dim=1).numpy()


def clip_masks(src, vid, T, w, e):
    if src == 'brace':
        seqs = pd.read_csv('/data/mlavl/datasets/brace/repo/annotations/sequences.csv')
        g = seqs[seqs.video_id == vid]
        c = np.arange(T) * 32 + 16
        out = {}
        for side, name in (('a', w.loc[vid, 'dancer_a']), ('b', w.loc[vid, 'dancer_b'])):
            m = np.zeros(T, bool)
            for s_, e_ in g[g.dancer == name][['start_frame', 'end_frame']].values:
                m |= (c >= s_) & (c <= e_)
            out[side] = m
        return out
    lab = np.load(os.path.join(SRC['extra'], 'turns', vid + '.npz'))['dancer']
    return {'a': lab == 1, 'b': lab == 2}


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    w = pd.read_csv('ours/battle/winners.csv').set_index('video_id')
    e = pd.read_csv('ours/battle/extra_battles.csv').query('in_brace == 0').set_index('video_id')
    moves = pd.read_csv('ours/vocab/action-label_BREAKING.csv').name.tolist()[:36]
    P = {k: text_emb(v) for k, v in PROMPTS.items()}
    M = text_emb(moves)
    jf = pd.read_csv('/data/mlavl/features/BRACE/judge_features.csv').set_index(['video_id', 'side'])
    from ours.battle_dataset import load_extra_battles
    consistent = {i['video_id'] for i in load_extra_battles(SRC['extra'], 'ours/battle/extra_battles.csv', min_consistency=0.8)[0]}
    rows, allemb = [], []
    for src, tab in (('brace', w), ('extra', e)):
        for vid in tab.index:
            af = os.path.join(SRC[src], 'appearance', vid + '.npz')
            if not os.path.exists(af) or (vid, 'a') not in jf.index or (vid, 'b') not in jf.index:
                continue
            ap = np.load(af)
            masks = clip_masks(src, vid, len(ap['has']), w, e)
            for side in 'ab':
                k = masks[side] & (ap['has'] > 0)
                if k.sum() < 4:
                    break
                E = ap['emb'][k].astype(np.float32)
                E /= np.linalg.norm(E, axis=1, keepdims=True)
                sim = lambda key: (E @ P[key].T).max(1)
                top = (E @ M.T).argmax(1)
                hist = np.bincount(top, minlength=36) / len(top)
                j = jf.loc[(vid, side)]
                rows.append({'video_id': vid, 'side': side, 'source': src, 'year': int(tab.loc[vid, 'year']),
                             'winner_is_a': int(tab.loc[vid, 'winner_is_a']), 'consistent': src == 'brace' or vid in consistent,
                             'votes': str(tab.loc[vid, 'bracket_score_a' if tab.loc[vid, 'winner_is_a'] else 'bracket_score_b']),
                             'clip_exec': float((sim('exec_pos') - sim('exec_neg')).mean()),
                             'clip_tech': float((sim('tech_pos') - sim('tech_neg')).mean()),
                             'clip_moves_used': int((hist > 0.02).sum()),
                             'clip_moves_entropy': float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum()),
                             'emb_mean': E.mean(0), **{c: float(j[c]) for c in ['jerk', 'pose_entropy', 'spin_p90',
                                                                                  'inverted', 'freeze_inv', 'bas', 'freeze_on_beat']}})
    df = pd.DataFrame(rows)
    df = df[df.groupby('video_id').side.transform('count') == 2]
    mu = np.mean(np.stack(df.emb_mean.values), 0)
    df['clip_rarity'] = [float(1 - (x @ mu) / (np.linalg.norm(x) * np.linalg.norm(mu))) for x in df.emb_mean.values]
    z = lambda c: (df[c] - df[c].mean()) / (df[c].std() + 1e-9)
    crit = pd.DataFrame({
        'Execution': (z('clip_exec') - z('jerk')) / 2,
        'Vocabulary': (z('clip_moves_used') + z('clip_moves_entropy') + z('pose_entropy')) / 3,
        'Technique': (z('clip_tech') + z('spin_p90') + z('inverted') + z('freeze_inv')) / 4,
        'Musicality': (z('bas') + z('freeze_on_beat')) / 2,
        'Originality': z('clip_rarity')})
    df = pd.concat([df.drop(columns='emb_mean').reset_index(drop=True), crit.reset_index(drop=True)], axis=1)
    df['Rubric'] = sum(WEIGHTS[c] * df[c] for c in WEIGHTS)
    df.to_csv('/data/mlavl/features/BRACE/rubric_scores.csv', index=False)
    A, B = df[df.side == 'a'].set_index('video_id'), df[df.side == 'b'].set_index('video_id')
    y = A.winner_is_a.values
    rng = np.random.default_rng(0)

    def acc(diff, m):
        c = ((diff[m] > 0) == (y[m] == 1)).astype(float)
        b = rng.choice(c, (5000, len(c))).mean(1)
        return f'{c.mean():.1%} ({np.percentile(b, 2.5):.1%}-{np.percentile(b, 97.5):.1%}, n={len(c)})'

    subsets = {'170 (BRACE + consistent extra)': A.consistent.values, 'BRACE 81': (A.source == 'brace').values,
               'all 280': np.ones(len(A), bool), 'unanimous (170 set)': A.consistent.values & (A.votes == '5').values}
    print('Fixed expert rubric (weights from Olympic judges, nothing fitted on our winners): dancer with the higher score wins\n')
    print('| score | ' + ' | '.join(subsets) + ' |')
    print('| --- | ' + ' | '.join('---' for _ in subsets) + ' |')
    for c in list(WEIGHTS) + ['Rubric']:
        d = A[c].values - B[c].values
        print(f'| {c} | ' + ' | '.join(acc(d, m) for m in subsets.values()) + ' |')
    from sklearn.linear_model import LogisticRegression
    D = A[list(WEIGHTS)].values - B[list(WEIGHTS)].values
    yr = A.year.values
    p = np.zeros(len(y))
    for t in np.unique(yr):
        tr, te = (yr != t) & A.consistent.values, yr == t
        X, Y = np.concatenate([D[tr], -D[tr]]), np.concatenate([y[tr], 1 - y[tr]])
        p[te] = LogisticRegression(C=0.1, fit_intercept=False).fit(X, Y).predict_proba(D[te])[:, 1] - 0.5
    print('| 5 criteria, logistic LOYO | ' + ' | '.join(acc(p, m) for m in subsets.values()) + ' |')


if __name__ == '__main__':
    main()
