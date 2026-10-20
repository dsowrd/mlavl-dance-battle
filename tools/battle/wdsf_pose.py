"""Pose-based clip features for the WDSF battle turns (ours, Phase 5c: appearance-free input for MLAVL).

Why: on the WDSF judge scores, MLAVL on whole-frame Video Swin features learns who the dancer is (clothing, body) rather
than how they dance (37-41% round accuracy on athletes unseen in training). Joint positions normalised by the dancer's
box carry no clothing or identity cues.

    $MLAVL_PY -I tools/battle/wdsf_pose.py turns       # features/WDSF/turns/{id}.npz dancing mask from wdsf_turns.csv
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/estimate_pose.py --set wdsf --stride 3 [--shard k --nshards n]
    $MLAVL_PY -I tools/battle/wdsf_pose.py features    # features/WDSF/pose/{id}.npz: pose (T, 114), has_pose (T,)

Per clip (32 frames, the clip grid of clips/ and clips_pamfn/): the dancer in each sampled frame is chosen by the
BRACE-trained selector (estimate_pose.learned_select: tracking + motion + the clip's dancer-crop box), then
extract_pose_features.clip_feature: pose mean / std (joints relative to the box centre / box height), per-joint speed
mean / max, global motion statistics and the kick-beat alignment (Bailando's beat-align idea) against librosa beats.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_pose import learned_select  # noqa: E402
from extract_pose_features import DIM, clip_feature, music_beats  # noqa: E402

F = '/data/mlavl/features/WDSF'
V = '/data/mlavl/videos/wdsf'


def turns():
    t = pd.read_csv('/data/mlavl/datasets/wdsf/wdsf_turns.csv')
    old = f'{F}/turns'
    if os.path.isdir(old) and not os.path.isdir(f'{F}/turns_v1'):
        os.rename(old, f'{F}/turns_v1')                  # first (BRACE-detector) segmentation, superseded
    os.makedirs(old, exist_ok=True)
    for vid, g in t.groupby('video_id'):
        d = np.load(f'{F}/clips/{vid}.npz')
        sec = float(d['clip_len']) / float(d['fps'])
        centre = (np.arange(len(d['rgb'])) + 0.5) * sec
        lab = np.zeros(len(centre), np.int8)
        for r in g.itertuples():
            lab[(centre >= r.start_s) & (centre <= r.end_s)] = 1 if r.athlete == 'red' else 2
        np.savez(f'{old}/{vid}.npz', dancer=lab)
    print(t.video_id.nunique(), 'videos')


def features():
    clf = pickle.load(open('/data/mlavl/features/BRACE/dancer_selector.pkl', 'rb'))
    os.makedirs(f'{F}/pose', exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(f'{F}/pose_est') if f.endswith('.npz'))
    for k, vid in enumerate(ids):
        dst = f'{F}/pose/{vid}.npz'
        if os.path.exists(dst):
            continue
        d = dict(np.load(f'{F}/pose_est/{vid}.npz'))
        ap = np.load(f'{F}/appearance/{vid}.npz')
        c = np.load(f'{F}/clips/{vid}.npz')
        T, fps = len(c['rgb']), float(c['fps'])
        sel = learned_select(d, ap['box'], ap['has'], clf) if len(d['frame']) else np.array([], int)
        kps = {int(f): (d['kp'][i][j][:, :2], d['box'][i][j], '') for i, (f, j) in enumerate(zip(d['frame'], sel)) if j >= 0}
        beats = music_beats(f'{V}/{vid}.mp4')
        pose = np.zeros((T, DIM), np.float32)
        has = np.zeros(T, np.int8)
        for t in range(T):
            pose[t], has[t] = clip_feature(list(range(32 * t, 32 * t + 32)), kps, fps, beats)
        np.savez(dst, pose=pose, has_pose=has)
        print(f'[{k + 1}/{len(ids)}] {vid}: {has.sum()} / {T} clips with pose', flush=True)


if __name__ == '__main__':
    {'turns': turns, 'features': features}[sys.argv[1]]()
