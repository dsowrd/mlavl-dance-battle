"""Turn segmentation of the WDSF battle videos with the round count known from the judges' scores (ours, Phase 5c).

Each WDSF battle video has n_rounds rounds (wdsf_judge_scores.csv), and every round is one set by each athlete, so
the video holds exactly 2 * n_rounds dancing turns. Per video:

1. dancing / turn-start probabilities per 32-frame clip from the BRACE-trained BiGRU detector
   (tools/battle/segment_turns.py, saved in BCONE_EXTRA/turn_detector.pt) on the K400 + AST clip features (clips/);
2. the 2 * n_rounds - 1 turn cuts are chosen by the same DP as segment_turns.decode_turns, but with N fixed;
3. who is who: the two turns of a round are different athletes; the orientation of each round is chosen so that the
   mean dancer-crop CLIP embeddings (appearance/) of each athlete's turns agree best across rounds (all 2^(R-1)
   assignments are scored); X = the athlete of the first turn;
4. X is mapped to blue: WDSF rules let the blue athlete start the battle (BRRM v4, 3.5.1.8). Checked by hand on a
   sample (see the review page); the rule is only an assumption until then.

    $MLAVL_PY -I tools/battle/wdsf_turns.py [--ids ...]
Writes /data/mlavl/features/WDSF/turns/{video_id}.npz (dancer (T,) int8: 0 none, 1 blue, 2 red; p_dancing, p_start)
and /data/mlavl/datasets/wdsf/wdsf_turns.csv: one row per turn with round, order in round, start/end in seconds,
athlete (blue/red), and confidences (weakest cut probability, appearance margin of the chosen orientation).
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_turns import Detector, smooth_binary  # noqa: E402

F = '/data/mlavl/features/WDSF'
D = '/data/mlavl/datasets/wdsf'


def decode_fixed(p_dance, p_start, n_turns, min_turn=6):
    """Cuts for exactly n_turns turns (DP of segment_turns.decode_turns with N fixed). Returns the turn bounds
    [(start_clip, end_clip)] over the dancing span, the dancing mask and the cut probabilities."""
    on = smooth_binary(p_dance)
    idx = np.where(on)[0]
    if len(idx) == 0:
        return None, on, []
    pe = np.clip(p_start, 1e-4, 1 - 1e-4)
    lo = np.log(pe) - np.log(1 - pe)
    k = n_turns - 1
    cand = [t for t in idx if t - idx[0] >= min_turn and idx[-1] - t >= min_turn - 1]
    n = len(cand)
    if n < k:
        return None, on, []
    f = np.full((k + 1, n), -np.inf)
    bp = np.full((k + 1, n), -1, int)
    f[1] = lo[cand]
    for c in range(2, k + 1):
        best_i, best_v = -1, -np.inf
        i = 0
        for j in range(n):                        # running max over i with cand[j] - cand[i] >= min_turn
            while i < j and cand[j] - cand[i] >= min_turn:
                if f[c - 1, i] > best_v:
                    best_v, best_i = f[c - 1, i], i
                i += 1
            if best_i >= 0:
                f[c, j], bp[c, j] = best_v + lo[cand[j]], best_i
    j = int(np.argmax(f[k]))
    if not np.isfinite(f[k, j]):
        return None, on, []
    cuts, c = [], k
    while c >= 1 and j >= 0:
        cuts.append(cand[j])
        j, c = bp[c, j], c - 1
    cuts = sorted(cuts)
    starts = [idx[0]] + cuts
    ends = cuts + [idx[-1] + 1]
    return list(zip(starts, ends)), on, [float(p_start[t]) for t in cuts]


def orient(turn_means):
    """turn_means: list of mean embeddings (or None) in time order, two per round. Returns (x_first per round,
    margin): x_first[r] = True if X dances first in round r, X := athlete of the first turn."""
    R = len(turn_means) // 2
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))
    scores = []
    for flips in itertools.product([True, False], repeat=R - 1):
        x_first = (True,) + flips
        X = [turn_means[2 * r + (0 if xf else 1)] for r, xf in enumerate(x_first)]
        Y = [turn_means[2 * r + (1 if xf else 0)] for r, xf in enumerate(x_first)]
        s = 0.0
        for grp in (X, Y):
            g = [v for v in grp if v is not None]
            s += sum(cos(a, b) for a, b in itertools.combinations(g, 2))
        for a in X:                                # and X turns should differ from Y turns
            for b in Y:
                if a is not None and b is not None:
                    s -= cos(a, b) / max(1, R)
        scores.append((s, x_first))
    scores.sort(reverse=True)
    margin = scores[0][0] - scores[1][0] if len(scores) > 1 else float('nan')
    return list(scores[0][1]), margin


def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det = Detector().to(device)
    det.load_state_dict(torch.load('/data/mlavl/features/BCONE_EXTRA/turn_detector.pt', map_location=device))
    det.eval()
    bv = pd.read_csv(f'{D}/wdsf_battle_videos.csv')
    bv = bv[bv.video_id.notna()].drop_duplicates('video_id')
    if a.ids:
        bv = bv[bv.video_id.isin(a.ids)]
    os.makedirs(f'{F}/turns', exist_ok=True)
    rows = []
    for r in bv.itertuples():
        vid = r.video_id
        cp, ap = f'{F}/clips/{vid}.npz', f'{F}/appearance/{vid}.npz'
        if not (os.path.exists(cp) and os.path.exists(ap)):
            continue
        d, app = np.load(cp), np.load(ap)
        X = np.concatenate([d['rgb'], d['audio']], 1).astype(np.float32)
        with torch.no_grad():
            p = torch.sigmoid(det(torch.from_numpy(X)[None].to(device))[0]).cpu().numpy()
        nt = 2 * int(r.n_rounds)
        bounds, on, cut_p = decode_fixed(p[:, 0], p[:, 1], nt)
        if bounds is None:
            print(f'{vid}: no decode', flush=True)
            continue
        means = []
        for s, e in bounds:
            k = [t for t in range(s, e) if on[t] and app['has'][t] > 0]
            means.append(app['emb'][k].astype(np.float32).mean(0) if len(k) >= 3 else None)
        x_first, margin = orient(means)
        lab = np.zeros(len(on), np.int8)
        sec = float(d['clip_len']) / float(d['fps'])
        for i, (s, e) in enumerate(bounds):
            rd, order = i // 2, i % 2
            is_x = (order == 0) == x_first[rd]
            athlete = 'blue' if is_x else 'red'                 # assumption: X (first turn) = blue
            lab[s:e][on[s:e]] = 1 if athlete == 'blue' else 2
            rows.append({'video_id': vid, 'competition': r.competition, 'battle_id': r.battle_id,
                         'round': rd + 1, 'order': order + 1, 'athlete': athlete,
                         'name': r.blue if athlete == 'blue' else r.red,
                         'start_s': round(s * sec, 1), 'end_s': round(e * sec, 1),
                         'dancing_share': round(float(on[s:e].mean()), 2),
                         'min_cut_p': round(min(cut_p) if cut_p else 0.0, 3), 'orient_margin': round(margin, 3)})
        np.savez(f'{F}/turns/{vid}.npz', dancer=lab, p_dancing=p[:, 0], p_start=p[:, 1])
        print(f'{vid}: {nt} turns, starts ' + ' '.join(f'{s * sec:.0f}' for s, _ in bounds)
              + f'  X first per round {x_first}  margin {margin:.2f}', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f'{D}/wdsf_turns{"_" + a.tag if a.tag else ""}.csv', index=False)
    print(len(out), 'turns from', out.video_id.nunique() if len(out) else 0, 'videos')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ids', nargs='*')
    p.add_argument('--tag', default='')
    main(p.parse_args())
