"""2D pose of the dancing dancer for every battle video (ours, Phase 5/6: pose for battles without BRACE keypoints).

torchvision Keypoint R-CNN (ResNet-50 FPN, COCO-17 keypoints, the same joint layout as BRACE) on every `stride`-th
frame of the clips in which someone is dancing (BRACE: annotated rounds; others: automatic turns). The dancer in a
frame is the detected person with the largest IoU with that clip's dancer box (tools/battle/appearance.py, clip
centre), falling back to the large-and-central heuristic.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/estimate_pose.py --set brace  [--shard 0 --nshards 3]
    CUDA_VISIBLE_DEVICES=1 $MLAVL_PY -I tools/battle/estimate_pose.py --set extra  [--shard ...]
    $MLAVL_PY -I tools/battle/estimate_pose.py --set brace --check      # accuracy vs the BRACE keypoints (PCK)

Output: {features}/pose_est/{video_id}.npz with frame (F,) int and, for up to 6 persons per frame,
kp (F, 6, 17, 3) x,y,score in 1920x1080 px, box (F, 6, 4) x,y,w,h, det_score (F, 6) (0 = no person).
The dancer is chosen afterwards by select_dancer (tracking + motion), tuned against the BRACE keypoints.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from appearance import iou, pick_dancer  # noqa: E402

FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
W, H = 960, 540
MAXP = 6     # persons kept per frame (score >= 0.5, highest first); the dancer is selected offline (select_dancer)
SETS = {'brace': ('/data/mlavl/videos/brace', '/data/mlavl/features/BRACE'),
        'extra': ('/data/mlavl/videos/bcone_extra', '/data/mlavl/features/BCONE_EXTRA'),
        'wdsf': ('/data/mlavl/videos/wdsf', '/data/mlavl/features/WDSF')}   # turns/ from wdsf_pose.py turns


def dancing_clips(set_, features, vid, T):
    if set_ == 'brace':
        seqs = pd.read_csv('/data/mlavl/datasets/brace/repo/annotations/sequences.csv')
        g = seqs[seqs.video_id == vid]
        c = np.arange(T) * 32 + 16
        m = np.zeros(T, bool)
        for s, e in g[['start_frame', 'end_frame']].values:
            m |= (c >= s) & (c <= e)
        return m
    return np.load(os.path.join(features, 'turns', vid + '.npz'))['dancer'] > 0


def frames_iter(path, stride):
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-i', path, '-vf', f"select='not(mod(n\\,{stride}))',scale={W}:{H}",
                          '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    n, k = W * H * 3, 0
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield k * stride, np.frombuffer(b, np.uint8).reshape(H, W, 3)
        k += 1
    p.stdout.close()
    p.wait()


def run(a):
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
    device = torch.device('cuda')
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).eval().to(device)
    videos, features = SETS[a.set]
    os.makedirs(os.path.join(features, 'pose_est'), exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(features, 'appearance')) if f.endswith('.npz'))
    ids = ids[a.shard::a.nshards]
    for k, vid in enumerate(ids):
        dst = os.path.join(features, 'pose_est', vid + '.npz')
        if os.path.exists(dst):
            continue
        ap = np.load(os.path.join(features, 'appearance', vid + '.npz'))
        T = len(ap['has'])
        on = dancing_clips(a.set, features, vid, T)
        out_f, out_kp, out_box, out_s = [], [], [], []
        batch = []

        def flush():
            if not batch:
                return
            x = [torch.from_numpy(fr).to(device).permute(2, 0, 1).float() / 255 for _, fr in batch]
            with torch.no_grad():
                res = model(x)
            for (f, _), r in zip(batch, res):
                keep = (r['labels'] == 1) & (r['scores'] >= 0.5)
                n = int(min(keep.sum().item(), MAXP))
                kp = np.zeros((MAXP, 17, 3), np.float32)
                bx = np.zeros((MAXP, 4), np.float32)
                sc = np.zeros(MAXP, np.float32)
                if n:
                    b = r['boxes'][keep][:n].cpu().numpy() * 2
                    b[:, 2:] -= b[:, :2]
                    k_ = r['keypoints'][keep][:n].cpu().numpy()
                    k_[..., :2] *= 2
                    kp[:n, :, :2] = k_[..., :2]
                    kp[:n, :, 2] = r['keypoints_scores'][keep][:n].cpu().numpy()
                    bx[:n] = b
                    sc[:n] = r['scores'][keep][:n].cpu().numpy()
                out_f.append(f)
                out_kp.append(kp)
                out_box.append(bx)
                out_s.append(sc)
            batch.clear()

        for f, fr in frames_iter(os.path.join(videos, vid + '.mp4'), a.stride):
            if f // 32 < T and on[f // 32]:
                batch.append((f, fr))
                if len(batch) == 16:
                    flush()
        flush()
        np.savez(dst, frame=np.array(out_f, np.int32), kp=np.array(out_kp, np.float32).reshape(-1, MAXP, 17, 3),
                 box=np.array(out_box, np.float32).reshape(-1, MAXP, 4), det_score=np.array(out_s, np.float32).reshape(-1, MAXP))
        print(f'[{k + 1}/{len(ids)}] {vid}: {len(out_f)} frames ({on.sum()} dancing clips)', flush=True)


def select_dancer(d, ap_box=None, ap_has=None, w_iou=2.0, w_motion=1.0, w_size=0.5, w_centre=0.5, w_app=1.0):
    """Pick one person per frame. Greedy tracking: score = w_iou * IoU(previous dancer) + w_motion * keypoint motion
    (normalised by bbox height, relative to the frame's max) + w_size * relative area + w_centre * centrality
    + w_app * IoU(appearance box of the clip). Returns indices (F,), -1 where no person."""
    F = len(d['frame'])
    sel = np.full(F, -1, int)
    prev_box, prev_kp = None, {}
    for i in range(F):
        sc = d['det_score'][i]
        valid = np.where(sc > 0)[0]
        if len(valid) == 0:
            continue
        b, kp = d['box'][i][valid], d['kp'][i][valid]
        area = b[:, 2] * b[:, 3]
        s = w_size * area / area.max()
        cx = b[:, 0] + b[:, 2] / 2
        s = s + w_centre * np.exp(-((cx - 960) / 480) ** 2)
        if prev_box is not None:
            s = s + w_iou * iou(b, prev_box)
        if i > 0 and sel[i - 1] >= 0:
            # motion of each candidate w.r.t. the closest person of the previous frame
            pk = d['kp'][i - 1][:, :, :2]
            pv = d['det_score'][i - 1] > 0
            mot = []
            for j in range(len(valid)):
                dd = np.linalg.norm(pk[pv] - kp[j][None, :, :2], axis=2).mean(1)
                mot.append(dd.min() / max(b[j, 3], 1) if len(dd) else 0)
            mot = np.array(mot)
            s = s + w_motion * mot / (mot.max() + 1e-6)
        if ap_box is not None:
            t = min(int(d['frame'][i]) // 32, len(ap_has) - 1)
            if ap_has[t]:
                s = s + w_app * iou(b, ap_box[t])
        j = valid[int(np.argmax(s))]
        sel[i] = j
        prev_box = d['box'][i][j]
    return sel


def candidate_features(d, ap_box=None, ap_has=None):
    """Per frame and candidate person: features for the learned dancer selector. Returns list of (frame idx,
    candidate idx array, feature matrix)."""
    out = []
    F = len(d['frame'])
    for i in range(F):
        valid = np.where(d['det_score'][i] > 0)[0]
        if len(valid) == 0:
            continue
        b, kp = d['box'][i][valid], d['kp'][i][valid]
        area = b[:, 2] * b[:, 3]
        cx, cy = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
        # motion: distance to the closest person in the previous and next sampled frame (keypoints, / bbox height)
        mot = np.zeros(len(valid))
        for k in (i - 1, i + 1):
            if 0 <= k < F:
                pv = d['det_score'][k] > 0
                if pv.any():
                    dd = np.linalg.norm(d['kp'][k][pv][None, :, :, :2] - kp[:, None, :, :2], axis=3).mean(2).min(1)
                    mot += dd / np.maximum(b[:, 3], 1)
        app = np.zeros(len(valid))
        if ap_box is not None:
            t = min(int(d['frame'][i]) // 32, len(ap_has) - 1)
            if ap_has[t]:
                app = iou(b, ap_box[t])
        hip_y = (kp[:, 11, 1] + kp[:, 12, 1]) / 2
        f = np.stack([area / area.max(), np.log(area + 1), np.exp(-((cx - 960) / 480) ** 2), cy / 1080,
                      b[:, 2] / np.maximum(b[:, 3], 1), d['det_score'][i][valid], mot, mot / (mot.max() + 1e-6),
                      app, (np.argsort(np.argsort(-area)) == 0).astype(float), kp[:, :, 2].mean(1) / 10,
                      hip_y / 1080, np.full(len(valid), len(valid))], 1)
        out.append((i, valid, f))
    return out


def learned_select(d, ap_box, ap_has, clf, smooth=1.0):
    """Viterbi over frames: emission = classifier log-probability of 'is the dancer', transition = smooth * IoU of
    consecutive choices (stay on the same person). Returns indices (F,), -1 where no person."""
    cand = candidate_features(d, ap_box, ap_has)
    sel = np.full(len(d['frame']), -1, int)
    if not cand:
        return sel
    em = [clf.predict_log_proba(f)[:, 1] for _, _, f in cand]
    score, back = [em[0]], [None]
    for n in range(1, len(cand)):
        i0, v0, _ = cand[n - 1]
        i1, v1, _ = cand[n]
        trans = np.stack([iou(d['box'][i0][v0], d['box'][i1][j]) for j in v1])     # (n1, n0)
        tot = score[-1][None, :] + smooth * trans
        back.append(tot.argmax(1))
        score.append(tot.max(1) + em[n])
    j = int(np.argmax(score[-1]))
    for n in range(len(cand) - 1, -1, -1):
        sel[cand[n][0]] = cand[n][1][j]
        if n:
            j = int(back[n][j])
    return sel


def train_selector(a):
    """Learn the dancer selector on BRACE (target = candidate closest to the annotated dancer keypoints); report
    leave-one-year-out wrong-person rate of greedy rule vs learned + Viterbi; save the model fitted on all years."""
    import pickle
    from sklearn.ensemble import GradientBoostingClassifier
    gt = brace_gt()
    years = pd.read_csv('/data/mlavl/datasets/brace/repo/videos_info.csv').set_index('video_id').year
    per_vid = {}
    for f in sorted(os.listdir('/data/mlavl/features/BRACE/pose_est')):
        vid = f[:-4]
        d = dict(np.load('/data/mlavl/features/BRACE/pose_est/' + f))
        ap = np.load('/data/mlavl/features/BRACE/appearance/' + f)
        X, Y = [], []
        tgt = np.full(len(d['frame']), -1)
        for i, valid, feat in candidate_features(d, ap['box'], ap['has']):
            key = (vid, int(d['frame'][i]))
            if key not in gt:
                continue
            g, h = gt[key]
            err = np.array([np.median(np.linalg.norm(d['kp'][i][j][:, :2] - g, axis=1)) / max(h, 1) for j in valid])
            if err.min() > 0.5:
                continue
            tgt[i] = valid[int(np.argmin(err))]
            X.append(feat)
            Y.append((np.arange(len(valid)) == np.argmin(err)).astype(int))
        per_vid[vid] = (d, ap, tgt, np.concatenate(X) if X else None, np.concatenate(Y) if Y else None)
    res = {'greedy': [], 'learned': []}
    for yr in sorted(set(years[list(per_vid)])):
        tr = [v for v in per_vid if years[v] != yr and per_vid[v][3] is not None]
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(
            np.concatenate([per_vid[v][3] for v in tr]), np.concatenate([per_vid[v][4] for v in tr]))
        for v in per_vid:
            if years[v] != yr:
                continue
            d, ap, tgt, _, _ = per_vid[v]
            m = tgt >= 0
            res['greedy'].append(((select_dancer(d, ap['box'], ap['has'], w_app=3.0, w_centre=1.5)[m] != tgt[m]).sum(), m.sum()))
            res['learned'].append(((learned_select(d, ap['box'], ap['has'], clf)[m] != tgt[m]).sum(), m.sum()))
    for k, v in res.items():
        print(f'{k}: wrong person {sum(x[0] for x in v) / sum(x[1] for x in v):.1%} of {sum(x[1] for x in v)} frames '
              f'where the dancer was detected')
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(
        np.concatenate([x[3] for x in per_vid.values() if x[3] is not None]),
        np.concatenate([x[4] for x in per_vid.values() if x[4] is not None]))
    pickle.dump(clf, open('/data/mlavl/features/BRACE/dancer_selector.pkl', 'wb'))


def brace_gt():
    z = zipfile.ZipFile('/data/mlavl/datasets/brace/dataset.zip')
    seg = re.compile(r'dataset/\d{4}/([^/]+)/.+\.json')
    gt = {}
    for n in z.namelist():
        m = seg.match(n)
        if m:
            for key, v in json.loads(z.read(n)).items():
                gt[(m.group(1), int(key.split('img-')[1].split('.')[0]))] = (np.array(v['keypoints'])[:, :2], v['box'][3])
    return gt


def check(a, **w):
    """Against the BRACE keypoints: (1) is the dancer among the detections at all (oracle choice), (2) PCK@0.2 and the
    wrong-person rate of select_dancer."""
    gt = brace_gt()
    pck, wrong, oracle_ok = [], 0, 0
    for f in sorted(os.listdir('/data/mlavl/features/BRACE/pose_est')):
        vid = f[:-4]
        d = np.load('/data/mlavl/features/BRACE/pose_est/' + f)
        ap = np.load('/data/mlavl/features/BRACE/appearance/' + f)
        sel = select_dancer(d, ap['box'], ap['has'], **w)
        for i, fr in enumerate(d['frame']):
            if (vid, int(fr)) not in gt or sel[i] < 0:
                continue
            g, h = gt[(vid, int(fr))]
            errs = [np.median(np.linalg.norm(d['kp'][i][j][:, :2] - g, axis=1)) / max(h, 1)
                    for j in np.where(d['det_score'][i] > 0)[0]]
            oracle_ok += min(errs) < 0.5
            dist = np.linalg.norm(d['kp'][i][sel[i]][:, :2] - g, axis=1) / max(h, 1)
            pck.append((dist < 0.2).mean())
            wrong += np.median(dist) > 0.5
    pck = np.array(pck)
    n = max(1, len(pck))
    print(f'frames {len(pck)}: dancer detected at all {oracle_ok / n:.1%}; selected: PCK@0.2 {pck.mean():.3f}, '
          f'wrong person {wrong / n:.1%}  ({w})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--set', choices=list(SETS), required=True)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--check', action='store_true')
    p.add_argument('--train_selector', action='store_true')
    a = p.parse_args()
    train_selector(a) if a.train_selector else (check(a) if a.check else run(a))
