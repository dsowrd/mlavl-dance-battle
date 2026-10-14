"""Judging-criteria features per dancer from estimated 2D pose, and winner prediction from them (ours, Phase 6).

Breaking judges (WDSF / Red Bull BC One) score technique/execution, vocabulary, originality, musicality and
difficulty. From the dancer's estimated COCO-17 keypoints (tools/battle/estimate_pose.py, every 3rd frame) during the
dancer's own turns we compute interpretable per-dancer statistics, all normalised by the dancer's bbox height and
expressed per second:

  time_s            total dancing time
  speed, speed_p90  mean / 90th-percentile joint speed                                   (energy)
  inverted          share of frames with the nose below the hips                          (power moves, freezes)
  floor             share of frames whose box is wider than tall                         (floor work)
  spin, spin_p90    angular speed of the shoulder line (rad/s)                           (spins, power)
  freezes, freeze_s, freeze_inv   holds >= 0.5 s with low joint speed; time; share of hold time inverted
  jerk              median joint acceleration                                            (control / execution, lower = cleaner)
  bas               kick-beat alignment (Bailando BAS, sigma 0.1 s) vs librosa music beats (musicality)
  freeze_on_beat    share of freeze onsets within 0.15 s of a music beat                 (musicality, hitting breaks)
  pose_entropy, pose_used   entropy / count of 24 pose clusters visited (k-means on normalised poses)  (vocabulary)
  travel            spatial range of the hip centre (bbox heights)
  extension         mean distance of wrists/ankles from the hip centre                  (shapes)

    $MLAVL_PY -I tools/battle/judge_features.py build      # -> {features}/judge_features.csv (BRACE + extra)
    $MLAVL_PY -I tools/battle/judge_features.py eval       # paired winner/loser tests + leave-one-year-out models
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_pose import learned_select  # noqa: E402

FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
SRC = {'brace': ('/data/mlavl/videos/brace', '/data/mlavl/features/BRACE'),
       'extra': ('/data/mlavl/videos/bcone_extra', '/data/mlavl/features/BCONE_EXTRA')}
NOSE, LSH, RSH, LHIP, RHIP, LWR, RWR, LANK, RANK = 0, 5, 6, 11, 12, 9, 10, 15, 16
FEATS = ['time_s', 'speed', 'speed_p90', 'inverted', 'floor', 'spin', 'spin_p90', 'freezes', 'freeze_s',
         'freeze_inv', 'jerk', 'bas', 'freeze_on_beat', 'pose_entropy', 'pose_used', 'travel', 'extension']


def music_beats(features, videos, vid):
    f = os.path.join(features, 'beats', vid + '.npy')
    if os.path.exists(f):
        return np.load(f)
    import librosa
    wav = np.frombuffer(subprocess.run([FFMPEG, '-v', 'error', '-i', os.path.join(videos, vid + '.mp4'), '-vn', '-ac', '1',
                                        '-ar', '22050', '-f', 'f32le', '-'], capture_output=True, check=True).stdout, np.float32)
    _, b = librosa.beat.beat_track(y=wav, sr=22050, units='time')
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.save(f, np.asarray(b, float))
    return np.asarray(b, float)


def dancer_of_clip(src, features, vid, T, battles):
    """Per clip 0/1/2 (none / A / B): BRACE human rounds, extra automatic turns."""
    if src == 'brace':
        seqs = pd.read_csv('/data/mlavl/datasets/brace/repo/annotations/sequences.csv')
        g = seqs[seqs.video_id == vid].sort_values('start_frame')
        a = battles.loc[vid, 'dancer_a']
        c = np.arange(T) * 32 + 16
        lab = np.zeros(T, np.int8)
        for r in g.itertuples():
            lab[(c >= r.start_frame) & (c <= r.end_frame)] = 1 if r.dancer == a else 2
        return lab
    return np.load(os.path.join(features, 'turns', vid + '.npz'))['dancer']


def norm_pose(kp, box):
    hip = (kp[:, LHIP, :2] + kp[:, RHIP, :2]) / 2
    return (kp[:, :, :2] - hip[:, None]) / np.maximum(box[:, 3], 1)[:, None, None]


def dancer_stats(fr, kp, box, fps, beats, km):
    """Statistics for one dancer from frames fr (frame numbers), kp (F,17,3), box (F,4)."""
    if len(fr) < 10:
        return None
    t = fr / fps
    h = np.maximum(box[:, 3], 1)
    dt = np.diff(t)
    ok = (dt > 0) & (dt < 0.25)                    # consecutive samples only (stride 3; gaps = turn breaks)
    v = np.linalg.norm(np.diff(kp[:, :, :2], axis=0), axis=2).mean(1) / h[1:] / np.where(dt > 0, dt, 1)
    v = v[ok]
    sh = kp[:, RSH, :2] - kp[:, LSH, :2]
    ang = np.unwrap(np.arctan2(sh[:, 1], sh[:, 0]))
    w = np.abs(np.diff(ang))[ok] / dt[ok]
    hip = (kp[:, LHIP, :2] + kp[:, RHIP, :2]) / 2
    inverted = kp[:, NOSE, 1] > hip[:, 1]
    acc = np.abs(np.diff(v)) if len(v) > 2 else np.zeros(1)
    # freezes: runs of low speed lasting >= 0.5 s
    low = np.zeros(len(t), bool)
    low[1:][ok] = v < np.percentile(v, 20) if len(v) else False
    freezes, freeze_s, freeze_inv, onsets = 0, 0.0, 0.0, []
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            dur = t[j - 1] - t[i]
            if dur >= 0.5:
                freezes += 1
                freeze_s += dur
                freeze_inv += inverted[i:j].mean() * dur
                onsets.append(t[i])
            i = j
        else:
            i += 1
    # kick-beat BAS: kinematic beats = local minima of smoothed speed
    vs = np.convolve(v, np.ones(3) / 3, mode='same') if len(v) > 3 else v
    tk = t[1:][ok][[k for k in range(1, len(vs) - 1) if vs[k] < vs[k - 1] and vs[k] <= vs[k + 1]]] if len(vs) > 3 else []
    tk = np.asarray(tk)
    bas = float(np.exp(-np.abs(tk[:, None] - beats[None]).min(1) ** 2 / (2 * 0.1 ** 2)).mean()) if len(tk) and len(beats) else 0.0
    fob = float(np.mean([np.abs(beats - o).min() < 0.15 for o in onsets])) if onsets and len(beats) else 0.0
    P = norm_pose(kp, box).reshape(len(kp), -1)
    cl = km.predict(P)
    hist = np.bincount(cl, minlength=km.n_clusters) / len(cl)
    ext = np.linalg.norm(kp[:, [LWR, RWR, LANK, RANK], :2] - hip[:, None], axis=2).mean(1) / h
    total = t[-1] - t[0]
    return {'time_s': float(ok.sum() * np.median(dt[ok]) if ok.any() else 0), 'speed': float(v.mean()) if len(v) else 0,
            'speed_p90': float(np.percentile(v, 90)) if len(v) else 0, 'inverted': float(inverted.mean()),
            'floor': float((box[:, 2] > box[:, 3]).mean()), 'spin': float(w.mean()) if len(w) else 0,
            'spin_p90': float(np.percentile(w, 90)) if len(w) else 0, 'freezes': freezes / max(total / 60, 1e-6),
            'freeze_s': freeze_s / max(total, 1e-6), 'freeze_inv': freeze_inv / max(freeze_s, 1e-6),
            'jerk': float(np.median(acc)), 'bas': bas, 'freeze_on_beat': fob,
            'pose_entropy': float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum()), 'pose_used': int((hist > 0.01).sum()),
            'travel': float(np.ptp(hip[:, 0]) / np.median(h)), 'extension': float(ext.mean())}


def build(a):
    from sklearn.cluster import KMeans
    w = pd.read_csv('ours/battle/winners.csv').set_index('video_id')
    e = pd.read_csv('ours/battle/extra_battles.csv')
    e = e[e.in_brace == 0].set_index('video_id')
    fps_b = pd.read_csv('/data/mlavl/datasets/brace/repo/videos_info.csv').set_index('video_id').fps
    jobs = [('brace', v) for v in w.index] + [('extra', v) for v in e.index]
    import pickle
    # dancer selector learned on BRACE (estimate_pose.py --train_selector; 12.0% wrong person held-out-year vs 20.6%
    # for the hand-made rule). It sees BRACE keypoints, never winner labels.
    selector = pickle.load(open('/data/mlavl/features/BRACE/dancer_selector.pkl', 'rb'))
    loaded = []
    for src, vid in jobs:
        videos, feats = SRC[src]
        pf, af = os.path.join(feats, 'pose_est', vid + '.npz'), os.path.join(feats, 'appearance', vid + '.npz')
        if not (os.path.exists(pf) and os.path.exists(af)):
            continue
        d, ap = dict(np.load(pf)), np.load(af)
        sel = learned_select(d, ap['box'], ap['has'], selector)
        keep = sel >= 0
        fr = d['frame'][keep]
        kp = d['kp'][keep, sel[keep]]
        box = d['box'][keep, sel[keep]]
        good = kp[:, :, 2].mean(1) > 0                                  # keypoint-score mean (logit-like)
        loaded.append((src, vid, fr[good], kp[good], box[good], len(ap['has'])))
    allp = np.concatenate([norm_pose(x[3], x[4]).reshape(len(x[3]), -1) for x in loaded])
    km = KMeans(24, n_init=4, random_state=0).fit(allp[np.random.default_rng(0).choice(len(allp), min(60000, len(allp)), replace=False)])
    rows = []
    for src, vid, fr, kp, box, T in loaded:
        videos, feats = SRC[src]
        tab = w if src == 'brace' else e
        lab = dancer_of_clip(src, feats, vid, T, tab)
        fps = float(fps_b[vid]) if src == 'brace' else float(np.load(os.path.join(feats, 'clips', vid + '.npz'))['fps'])
        beats = music_beats(feats, videos, vid)
        cl = np.minimum(fr // 32, T - 1)
        for side, code in (('a', 1), ('b', 2)):
            m = lab[cl] == code
            st = dancer_stats(fr[m], kp[m], box[m], fps, beats, km)
            if st is None:
                continue
            rows.append({'video_id': vid, 'source': src, 'year': int(tab.loc[vid, 'year']), 'side': side,
                         'winner_is_a': int(tab.loc[vid, 'winner_is_a']), **st})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SRC['brace'][1], 'judge_features.csv'), index=False)
    print(df.groupby('source').video_id.nunique().to_dict(), 'battles;', len(df), 'dancer rows')


def evaluate(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(os.path.join(SRC['brace'][1], 'judge_features.csv'))
    if a.extra_filter:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from ours.battle_dataset import load_extra_battles
        ok, _ = load_extra_battles(SRC['extra'][1], 'ours/battle/extra_battles.csv', min_consistency=a.extra_filter)
        ok = {i['video_id'] for i in ok}
        df = df[(df.source == 'brace') | df.video_id.isin(ok)]
    A, B = df[df.side == 'a'].set_index('video_id'), df[df.side == 'b'].set_index('video_id')
    ids = A.index.intersection(B.index)
    A, B = A.loc[ids], B.loc[ids]
    y = A.winner_is_a.values
    src, yr = A.source.values, A.year.values
    D = (A[FEATS].values - B[FEATS].values)
    print(f'{len(ids)} battles ({(src == "brace").sum()} BRACE, {(src == "extra").sum()} extra)\n')
    print('| feature | winner > loser (share) | sign-test p | winner mean | loser mean |')
    print('| --- | --- | --- | --- | --- |')
    from scipy.stats import binomtest
    for k, f in enumerate(FEATS):
        dw = np.where(y == 1, D[:, k], -D[:, k])          # winner minus loser
        nz = dw[dw != 0]
        share = (nz > 0).mean() if len(nz) else 0.5
        p = binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1
        wm = np.where(y == 1, A[f].values, B[f].values).mean()
        lm = np.where(y == 1, B[f].values, A[f].values).mean()
        print(f'| {f} | {share:.1%} | {p:.3f} | {wm:.3f} | {lm:.3f} |')
    rng = np.random.default_rng(0)
    print('\nleave-one-year-out, antisymmetric logistic regression on standardised f(A) - f(B):')
    for C in [0.01, 0.1, 1.0]:
        p = np.zeros(len(y))
        for t in np.unique(yr):
            tr, te = yr != t, yr == t
            X = np.concatenate([D[tr], -D[tr]])
            Y = np.concatenate([y[tr], 1 - y[tr]])
            sc = StandardScaler(with_mean=False).fit(X)
            clf = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(sc.transform(X), Y)
            p[te] = clf.predict_proba(sc.transform(D[te]))[:, 1]
        c = ((p > 0.5) == y).astype(float)
        out = []
        for name, m in [('all', np.ones(len(y), bool)), ('BRACE', src == 'brace'), ('extra', src == 'extra')]:
            b = rng.choice(c[m], (5000, m.sum())).mean(1)
            out.append(f'{name} {c[m].mean():.1%} ({np.percentile(b, 2.5):.1%}-{np.percentile(b, 97.5):.1%}, n={m.sum()})')
        print(f'  C={C}: ' + ' | '.join(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build', 'eval'])
    p.add_argument('--extra_filter', type=float, default=0.8, help='turn-consistency threshold for extra battles (0 = all)')
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    build(a) if a.mode == 'build' else evaluate(a)
