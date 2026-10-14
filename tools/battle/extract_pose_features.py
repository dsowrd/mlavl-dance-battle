"""Clip-level pose and kick-beat features for the BRACE battles (ours, Phase 5).

Aligned to the same 32-frame clip grid as tools/battle/extract_features.py (clip i = frames [32 i, 32 i + 32) at the
video's native fps), so they can be added to the video tokens clip by clip. Per clip (only frames that have BRACE
keypoints, i.e. annotated dancing; other clips are all-zero with has_pose = 0):

    pose_mean (34), pose_std (34)   joints relative to the bbox centre, divided by bbox height (scale/position free)
    speed_mean (17), speed_max (17) per-joint speed per frame, in bbox heights
    global (8)                      mean / p90 speed, bbox height change, mean vertical position of the hips
                                    (floor vs standing), fraction of frames with keypoints, toprock / footwork /
                                    powermove share (BRACE segment labels)
    beat (4)                        kick-beat alignment: kinematic beats = local minima of the smoothed mean joint
                                    speed (Bailando, Siyao et al. CVPR 2022); BAS = mean over kinematic beats of
                                    exp(-d^2 / (2 sigma^2)), d = distance to the nearest music beat, sigma = 0.1 s;
                                    plus music beats per second, kinematic beats per second, has_beat

Music beats: librosa beat tracker on each video's full audio (BRACE's Essentia beats exist for only 146 of the 465
sequences; agreement with them is printed for validation).

    $MLAVL_PY -I tools/battle/extract_pose_features.py --out /data/mlavl/features/BRACE
Output: {out}/pose/{video_id}.npz with pose (T, 114) float32 (T = number of clips in clips/{video_id}.npz), has_pose (T,)
"""
import argparse
import json
import os
import re
import subprocess
import zipfile

import numpy as np

SEG_RE = re.compile(r'dataset/(\d{4})/([^/]+)/(.+)_(\d+)-(\d+)_(toprock|footwork|powermove)\.json')
MOVES = ['toprock', 'footwork', 'powermove']
FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
DIM = 34 + 34 + 17 + 17 + 8 + 4


def music_beats(path, sr=22050):
    import librosa
    wav = np.frombuffer(subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), '-f',
                                        'f32le', '-'], capture_output=True, check=True).stdout, np.float32)
    _, beats = librosa.beat.beat_track(y=wav, sr=sr, units='time')
    return np.asarray(beats, float)


def load_keypoints(z, names):
    """frame -> (kp (17,2), box (4), move) for one video."""
    out = {}
    for name in names:
        m = SEG_RE.match(name)
        move = m.group(6)
        d = json.loads(z.read(name))
        for k, v in d.items():
            f = int(k.split('img-')[1].split('.')[0])
            out[f] = (np.asarray(v['keypoints'], np.float32)[:, :2], np.asarray(v['box'][:4], np.float32), move)
    return out


def clip_feature(frames, kps, fps, beats_s, sigma=0.1):
    have = [f for f in frames if f in kps]
    if len(have) < 4:
        return np.zeros(DIM, np.float32), 0
    kp = np.stack([kps[f][0] for f in have])                       # F x 17 x 2
    box = np.stack([kps[f][1] for f in have])                      # x y w h
    h = np.maximum(box[:, 3], 1.0)[:, None, None]
    centre = (box[:, None, :2] + box[:, None, 2:4] / 2)
    rel = (kp - centre) / h                                        # F x 17 x 2
    sp = np.linalg.norm(np.diff(kp, axis=0), axis=2) / h[1:, :, 0]  # (F-1) x 17
    msp = sp.mean(1)
    hips = rel[:, 11:13, 1].mean()                                 # COCO 11/12 = hips, y down
    moves = [kps[f][2] for f in have]
    g = [msp.mean(), np.percentile(msp, 90), (box[-1, 3] - box[0, 3]) / max(box[0, 3], 1), hips,
         len(have) / len(frames)] + [moves.count(m) / len(moves) for m in MOVES]
    # kick-beat: local minima of smoothed mean joint speed, in seconds
    k = np.convolve(msp, np.ones(3) / 3, mode='same')
    idx = [i for i in range(1, len(k) - 1) if k[i] < k[i - 1] and k[i] <= k[i + 1]]
    t_kin = np.array([have[1 + i] for i in idx], float) / fps
    t0, t1 = frames[0] / fps, (frames[-1] + 1) / fps
    mb = beats_s[(beats_s >= t0 - 0.5) & (beats_s < t1 + 0.5)]
    if len(t_kin) and len(mb):
        d = np.abs(t_kin[:, None] - mb[None]).min(1)
        bas, has_beat = float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean()), 1.0
    else:
        bas, has_beat = 0.0, 0.0
    dur = t1 - t0
    beat = [bas, ((mb >= t0) & (mb < t1)).sum() / dur, len(t_kin) / dur, has_beat]
    f = np.concatenate([rel.mean(0).ravel(), rel.std(0).ravel(), sp.mean(0), sp.max(0), g, beat]).astype(np.float32)
    return f, 1


def validate_beats(brace, vid, beats_s, seqs):
    """Mean distance (s) from BRACE Essentia beats to the nearest librosa beat, for this video's sequences that have them."""
    ab = json.load(open(os.path.join(brace, 'repo/annotations/audio_beats.json')))
    ds = []
    for s in seqs:
        uid, start_s = s
        if uid in ab and len(beats_s):
            ref = np.asarray(ab[uid]['beats_sec']) + start_s
            ds.append(np.abs(ref[:, None] - beats_s[None]).min(1))
    return np.concatenate(ds) if ds else None


def main(a):
    import pandas as pd
    seqs = pd.read_csv(os.path.join(a.brace, 'repo/annotations/sequences.csv'))
    fps_of = pd.read_csv(os.path.join(a.brace, 'repo/videos_info.csv')).set_index('video_id').fps.to_dict()
    z = zipfile.ZipFile(os.path.join(a.brace, 'dataset.zip'))
    by_vid = {}
    for n in z.namelist():
        m = SEG_RE.match(n)
        if m:
            by_vid.setdefault(m.group(2), []).append(n)
    os.makedirs(os.path.join(a.out, 'pose'), exist_ok=True)
    dists = []
    for k, vid in enumerate(sorted(by_vid)):
        clips = np.load(os.path.join(a.out, 'clips', vid + '.npz'))
        T, L, fps = len(clips['rgb']), int(clips['clip_len']), float(fps_of[vid])
        beats_s = music_beats(os.path.join(a.videos, vid + '.mp4'))
        kps = load_keypoints(z, by_vid[vid])
        feats, has = zip(*[clip_feature(list(range(i * L, i * L + L)), kps, fps, beats_s) for i in range(T)])
        np.savez(os.path.join(a.out, 'pose', vid + '.npz'), pose=np.stack(feats), has_pose=np.array(has, np.int8),
                 music_beats_s=beats_s)
        g = seqs[seqs.video_id == vid]
        d = validate_beats(a.brace, vid, beats_s, [(r.uid, r.start_frame / fps) for r in g.itertuples()])
        if d is not None:
            dists.append(d)
        print(f'[{k + 1}/{len(by_vid)}] {vid}: T={T}, clips with pose {sum(has)}, music beats {len(beats_s)}', flush=True)
    if dists:
        d = np.concatenate(dists)
        print(f'librosa vs BRACE Essentia beats: median |dt| {np.median(d) * 1000:.0f} ms, '
              f'{(d < 0.07).mean():.1%} within 70 ms, {(d < 0.15).mean():.1%} within 150 ms (n={len(d)})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--brace', default='/data/mlavl/datasets/brace')
    p.add_argument('--videos', default='/data/mlavl/videos/brace')
    p.add_argument('--out', default='/data/mlavl/features/BRACE')
    main(p.parse_args())
