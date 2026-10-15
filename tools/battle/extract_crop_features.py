"""Dancer-centred Video Swin features (ours, Phase 6 improvement).

tools/battle/extract_features.py resizes the whole frame to 224 px, where the dancer is a small part of a picture of
crowd and stage. Here each 32-frame clip is cropped around the dancer before Video Swin-B: the crop is a square of
side 1.4 x max(w, h) of the clip's median dancer box (from the learned dancer selector on the estimated pose,
tools/battle/estimate_pose.py; fallback: the appearance box at the clip centre; fallback: the centre square), fixed
for the 32 frames, resized to 224 x 224. Audio (AST) features are copied unchanged.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/extract_crop_features.py --set brace [--shard 0 --nshards 3]
Output: {features}/clips_crop/{video_id}.npz, same keys as clips/ (rgb (T, 1024) = dancer-centred), plus box_src (T,)
0 = centre square, 1 = appearance box, 2 = pose-selected dancer boxes.
"""
import argparse
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_pose import SETS, learned_select  # noqa: E402
from extract_features import MEAN, STD  # noqa: E402

FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
W, H = 960, 540          # decode size; boxes are in 1920x1080 px


def clip_boxes(vid, features, T):
    """(T, 4) box per clip in 1920x1080 px and its source."""
    ap = np.load(os.path.join(features, 'appearance', vid + '.npz'))
    boxes = np.tile(np.array([420., 0., 1080., 1080.]), (T, 1))      # centre square
    src = np.zeros(T, np.int8)
    m = ap['has'] > 0
    boxes[m], src[m] = ap['box'][m], 1
    pf = os.path.join(features, 'pose_est', vid + '.npz')
    if os.path.exists(pf):
        d = dict(np.load(pf))
        sel = learned_select(d, ap['box'], ap['has'], pickle.load(open('/data/mlavl/features/BRACE/dancer_selector.pkl', 'rb')))
        clip = np.minimum(d['frame'] // 32, T - 1)
        for t in np.unique(clip[sel >= 0]):
            k = np.where((clip == t) & (sel >= 0))[0]
            boxes[t] = np.median(d['box'][k, sel[k]], 0)
            src[t] = 2
    return boxes, src


def square(box, scale=1.4):
    x, y, w, h = box / 2                                           # to the 960x540 decode
    s = max(w, h) * scale
    cx, cy = x + w / 2, y + h / 2
    s = min(max(s, 64), H)
    x0 = int(np.clip(cx - s / 2, 0, W - s))
    y0 = int(np.clip(cy - s / 2, 0, H - s))
    return x0, y0, int(s)


def frames(path):
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-i', path, '-vf', f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt',
                          'rgb24', '-'], stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    p.stdout.close()
    p.wait()


@torch.no_grad()
def main(a):
    from torchvision.models.video import swin3d_b, Swin3D_B_Weights
    device = torch.device('cuda')
    vm = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
    vm.head = torch.nn.Identity()
    vm = vm.eval().to(device)
    videos, features = SETS[a.set]
    os.makedirs(os.path.join(features, 'clips_crop'), exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(features, 'appearance')) if f.endswith('.npz'))[a.shard::a.nshards]
    for k, vid in enumerate(ids):
        dst = os.path.join(features, 'clips_crop', vid + '.npz')
        if os.path.exists(dst):
            continue
        c = dict(np.load(os.path.join(features, 'clips', vid + '.npz')))
        T = len(c['rgb'])
        boxes, src = clip_boxes(vid, features, T)
        feats, buf, batch = [], [], []
        t = 0
        for f in frames(os.path.join(videos, vid + '.mp4')):
            if t >= T:
                break
            if not buf:
                x0, y0, s = square(boxes[t])
            buf.append(torch.from_numpy(f[y0:y0 + s, x0:x0 + s].copy()))
            if len(buf) == 32:
                x = torch.stack(buf).to(device).permute(3, 0, 1, 2).float().div_(255)    # C T h w
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                batch.append(x)
                buf, t = [], t + 1
                if len(batch) == 8:
                    xb = (torch.stack(batch) - MEAN.to(device)) / STD.to(device)
                    with torch.autocast('cuda', dtype=torch.float16):
                        feats.append(vm(xb).float().cpu().numpy())
                    batch = []
        if batch:
            xb = (torch.stack(batch) - MEAN.to(device)) / STD.to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                feats.append(vm(xb).float().cpu().numpy())
        rgb = np.concatenate(feats)
        assert len(rgb) == T, (vid, len(rgb), T)
        c['rgb'] = rgb.astype(np.float32)
        c['box_src'] = src
        np.savez(dst + '.tmp.npz', **c)
        os.replace(dst + '.tmp.npz', dst)
        print(f'[{k + 1}/{len(ids)}] {vid}: T={T}, boxes from pose {np.mean(src == 2):.2f} / appearance {np.mean(src == 1):.2f}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--set', choices=list(SETS), required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    main(p.parse_args())
