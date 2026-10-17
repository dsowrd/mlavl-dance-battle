"""Linear probes per information source (dancer-crop CLIP, BRACE pose, Swin video, AST audio), split by the
winner's vote count, and training on clear decisions only (ours, Phase 6). Run: $MLAVL_PY -I tools/battle/probe_sources.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/irl/Workspace_Hyundo/mlavl-dance-battle'); sys.path.insert(0, '/home/irl/Workspace_Hyundo/mlavl-dance-battle/tools/battle')
os.chdir('/home/irl/Workspace_Hyundo/mlavl-dance-battle')
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ours.battle_dataset import turns_to_rounds
rng = np.random.default_rng(0)
seqs = pd.read_csv('/data/mlavl/datasets/brace/repo/annotations/sequences.csv')
w = pd.read_csv('ours/battle/winners.csv'); e = pd.read_csv('ours/battle/extra_battles.csv')
from segment_turns import turn_consistency
def win_votes(r):
    s = str(r.bracket_score_a if r.winner_is_a else r.bracket_score_b)
    return int(s) if s.isdigit() else -1
rows = []
# BRACE: human turns -> per-dancer clip masks
for r in w.itertuples():
    d = np.load(f'/data/mlavl/features/BRACE/clips/{r.video_id}.npz'); ap = np.load(f'/data/mlavl/features/BRACE/appearance/{r.video_id}.npz')
    po = np.load(f'/data/mlavl/features/BRACE/pose/{r.video_id}.npz')
    st, L = d['clip_start_frame'], 32; g = seqs[seqs.video_id == r.video_id]
    m = {}
    for side, name in (('a', r.dancer_a), ('b', r.dancer_b)):
        k = np.zeros(len(st), bool)
        for s_, e_ in g[g.dancer == name][['start_frame', 'end_frame']].values: k |= (st >= s_) & (st + L <= e_ + 1)
        m[side] = k
    rows.append(dict(src='brace', year=r.year, y=r.winner_is_a, votes=win_votes(r), d=d, ap=ap, po=po, m=m,
                     dc=np.load(f'/data/mlavl/features/BRACE/clips_crop/{r.video_id}.npz')))
for r in e[e.in_brace == 0].itertuples():
    cf = f'/data/mlavl/features/BCONE_EXTRA/clips/{r.video_id}.npz'; tf = cf.replace('clips', 'turns'); af = cf.replace('clips', 'appearance')
    if not (os.path.exists(cf) and os.path.exists(tf) and os.path.exists(af)): continue
    t = np.load(tf); ap = np.load(af)
    if turn_consistency(t['dancer'], ap['emb'], ap['has'])[0] < 0.8: continue
    m = {'a': t['dancer'] == 1, 'b': t['dancer'] == 2}
    if m['a'].sum() < 8 or m['b'].sum() < 8: continue
    rows.append(dict(src='extra', year=r.year, y=r.winner_is_a, votes=win_votes(r), d=np.load(cf), ap=ap, po=None, m=m,
                     dc=np.load(cf.replace('/clips/', '/clips_crop/'))))
print(len(rows), 'battles; winner votes', pd.Series([x['votes'] for x in rows]).value_counts().to_dict())
def F(x, side, kind):
    k = x['m'][side]
    if kind == 'crop':
        k2 = k & (x['ap']['has'] > 0); v = x['ap']['emb'][k2].astype(np.float32)
    elif kind == 'pose':
        v = x['po']['pose'][k & (x['po']['has_pose'] > 0)]
    elif kind == 'video': v = x['d']['rgb'][k]
    elif kind == 'cropvid': v = x['dc']['rgb'][k]
    else: v = x['d']['audio'][k]
    return np.concatenate([v.mean(0), v.std(0)])
def probe(rs, kind, k=16, C=1.0):
    D = np.stack([F(x, 'a', kind) - F(x, 'b', kind) for x in rs]); y = np.array([x['y'] for x in rs]); yr = np.array([x['year'] for x in rs])
    p = np.zeros(len(rs))
    for yy in np.unique(yr):
        tr, te = yr != yy, yr == yy
        X = np.concatenate([D[tr], -D[tr]]); Y = np.concatenate([y[tr], 1 - y[tr]])
        sc = StandardScaler(with_mean=False).fit(X); Z = sc.transform(X)
        pca = PCA(min(k, Z.shape[1]), random_state=0).fit(np.concatenate([Z, -Z]))
        clf = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(pca.transform(Z), Y)
        p[te] = clf.predict_proba(pca.transform(sc.transform(D[te])))[:, 1]
    return ((p > 0.5) == y).astype(float), np.array([x['votes'] for x in rs]), np.array([x['src'] for x in rs])
def ci(c):
    b = rng.choice(c, (5000, len(c))).mean(1); return f'{c.mean():.1%} ({np.percentile(b,2.5):.1%}-{np.percentile(b,97.5):.1%}, n={len(c)})'
for kind, rs in [('cropvid', rows), ('video', rows), ('crop', rows), ('pose', [x for x in rows if x['src'] == 'brace']), ('audio', rows)]:
    for k in [8, 16, 32]:
        c, v, s = probe(rs, kind, k)
        print(f'{kind:5s} k={k:2d}: all {ci(c)} | unanimous(5) {ci(c[v==5])} | split(3) {ci(c[v==3])}')

print('\n--- train only on clear decisions (winner votes >= 4), evaluate on all ---')
def probe_clean(rs, kinds, k=16, C=1.0, min_votes=4):
    D = np.stack([np.concatenate([F(x, 'a', kd) - F(x, 'b', kd) for kd in kinds]) for x in rs])
    y = np.array([x['y'] for x in rs]); yr = np.array([x['year'] for x in rs]); v = np.array([x['votes'] for x in rs])
    p = np.zeros(len(rs))
    for yy in np.unique(yr):
        tr = (yr != yy) & (v >= min_votes); te = yr == yy
        X = np.concatenate([D[tr], -D[tr]]); Y = np.concatenate([y[tr], 1 - y[tr]])
        sc = StandardScaler(with_mean=False).fit(X); Z = sc.transform(X)
        pca = PCA(min(k, Z.shape[1]), random_state=0).fit(np.concatenate([Z, -Z]))
        clf = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(pca.transform(Z), Y)
        p[te] = clf.predict_proba(pca.transform(sc.transform(D[te])))[:, 1]
    return ((p > 0.5) == y).astype(float), v
for kinds in [['cropvid'], ['video'], ['cropvid', 'audio']]:
    for k in [8, 16, 32]:
        c, v = probe_clean(rows, kinds, k)
        print(f"{'+'.join(kinds):10s} k={k:2d}: all {ci(c)} | unanimous {ci(c[v==5])} | 4-1 {ci(c[v==4])} | split {ci(c[v==3])}")
