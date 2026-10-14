"""Dancer identity from appearance: who is dancing in each clip (ours, Phase 4 extension).

Whole-frame Video Swin features cannot tell the two dancers apart (2-means on them: 50% on BRACE), because the
frame is mostly crowd and stage. Here we crop the dancer at each clip's centre frame and embed the crop with the
CLIP ViT-B/16 image encoder; clothing and body shape make the two dancers separable.

    # oracle crops from the BRACE keypoint boxes: can appearance separate the dancers at all?
    $MLAVL_PY -I tools/battle/appearance.py oracle
    # detected crops (torchvision Faster R-CNN person boxes; the dancer = the person whose box is closest to the
    # frame centre among the large ones), for any video list
    $MLAVL_PY -I tools/battle/appearance.py detect --videos DIR --features FEATS [--ids ...]

Output: {features}/appearance/{video_id}.npz with emb (T, 512) float16 (zeros where no person), box (T, 4) in
1920x1080-equivalent pixels, has (T,) int8.
"""
import argparse
import json
import os
import re
import subprocess
import zipfile

import numpy as np
import pandas as pd
import torch

FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
SEG_RE = re.compile(r'dataset/(\d{4})/([^/]+)/(.+)_(\d+)-(\d+)_(toprock|footwork|powermove)\.json')
W, H = 960, 540        # decode size (half of 1080p)


def centre_frames(path, T, L=32):
    """Frames n = 16, 48, ... (clip centres) as uint8 (T, H, W, 3) at 960x540."""
    p = subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vf', f"select='not(mod(n-{L // 2}\\,{L}))',scale={W}:{H}",
                        '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
    fr = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
    if len(fr) < T:
        fr = np.concatenate([fr, np.zeros((T - len(fr), H, W, 3), np.uint8)])
    return fr[:T]


class Embedder:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPImageProcessor
        name = 'openai/clip-vit-base-patch16'
        self.m = CLIPModel.from_pretrained(name).eval().to(device)
        self.pp = CLIPImageProcessor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def __call__(self, crops):
        out = []
        for i in range(0, len(crops), 64):
            x = self.pp(images=crops[i:i + 64], return_tensors='pt')['pixel_values'].to(self.device)
            out.append(torch.nn.functional.normalize(self.m.get_image_features(pixel_values=x), dim=1).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, 512), np.float32)


def crop(frame, box, pad=0.1):
    """box x, y, w, h in 1920x1080 px -> crop of the half-size frame."""
    x, y, w, h = (np.asarray(box, float) / 2)
    x0, y0 = max(0, int(x - pad * w)), max(0, int(y - pad * h))
    x1, y1 = min(W, int(x + w * (1 + pad))), min(H, int(y + h * (1 + pad)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return frame[y0:y1, x0:x1]


def two_means(e, iters=30):
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    c = np.stack([e[0], e[np.argmin(e @ e[0])]])
    for _ in range(iters):
        a = np.argmax(e @ c.T, 1)
        c = np.stack([e[a == k].mean(0) if (a == k).any() else c[k] for k in range(2)])
        c /= np.linalg.norm(c, axis=1, keepdims=True)
    return np.argmax(e @ c.T, 1)


def oracle(a):
    """BRACE: crop the annotated dancer box at each clip centre that lies inside a round, cluster, compare."""
    device = torch.device('cuda')
    emb = Embedder(device)
    seqs = pd.read_csv(os.path.join(a.brace, 'repo/annotations/sequences.csv'))
    z = zipfile.ZipFile(os.path.join(a.brace, 'dataset.zip'))
    by_vid = {}
    for n in z.namelist():
        m = SEG_RE.match(n)
        if m:
            by_vid.setdefault(m.group(2), []).append(n)
    rows = []
    for k, (vid, g) in enumerate(seqs.groupby('video_id')):
        clips = np.load(os.path.join(a.features, 'clips', vid + '.npz'))
        T, L = len(clips['rgb']), int(clips['clip_len'])
        boxes = {}
        for n in by_vid.get(vid, []):
            for key, v in json.loads(z.read(n)).items():
                boxes[int(key.split('img-')[1].split('.')[0])] = v['box'][:4]
        g = g.sort_values('start_frame')
        first = g.dancer.iloc[0]
        truth = np.zeros(T, np.int8)
        for r in g.itertuples():
            c = np.arange(T) * L + L // 2
            truth[(c >= r.start_frame) & (c <= r.end_frame)] = 1 if r.dancer == first else 2
        frames = centre_frames(os.path.join(a.videos, vid + '.mp4'), T, L)
        idx, crops = [], []
        for t in np.where(truth > 0)[0]:
            f = t * L + L // 2
            near = sorted(range(f - 15, f + 16), key=lambda x: abs(x - f))       # nearest annotated frame in the clip
            b = next((boxes[x] for x in near if x in boxes), None)
            if b is None:
                continue
            cr = crop(frames[t], b)
            if cr is not None:
                idx.append(t)
                crops.append(cr)
        if len(crops) < 4:
            continue
        e = emb(crops)
        lab = two_means(e)
        tr = truth[idx] - 1
        acc = max((lab == tr).mean(), (lab != tr).mean())          # cluster ids are arbitrary
        rows.append({'video_id': vid, 'year': int(g.year.iloc[0]), 'clips': len(idx), 'identity_acc': acc})
        print(f'[{k + 1}] {vid}: {len(idx)} crops, identity acc {acc:.3f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.features, 'appearance_oracle_eval.csv'), index=False)
    print(df.identity_acc.describe().round(3).to_string())
    print('videos >= 0.9:', (df.identity_acc >= 0.9).mean().round(3))


class PersonDetector:
    def __init__(self, device):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        self.m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, frames, thr=0.6):
        """frames uint8 (N, H, W, 3) -> list of (boxes x,y,w,h in 1920x1080 px, scores) for persons."""
        out = []
        for i in range(0, len(frames), 16):
            x = torch.from_numpy(frames[i:i + 16]).to(self.device).permute(0, 3, 1, 2).float() / 255
            for r in self.m(list(x)):
                k = (r['labels'] == 1) & (r['scores'] >= thr)
                b = r['boxes'][k].cpu().numpy() * 2                     # back to 1080p pixels
                b[:, 2:] -= b[:, :2]
                out.append((b, r['scores'][k].cpu().numpy()))
        return out


def pick_dancer(boxes, prev=None):
    """The dancer = large person near the horizontal centre (crowd/opponent/MC stand at the edges); if a previous
    dancer box is given, prefer continuity. Score = area * centrality (* overlap bonus)."""
    if len(boxes) == 0:
        return None
    cx = boxes[:, 0] + boxes[:, 2] / 2
    area = boxes[:, 2] * boxes[:, 3]
    central = np.exp(-((cx - 960) / 480) ** 2)
    s = area / area.max() * central
    if prev is not None:
        s = s * (1 + iou(boxes, prev))
    return boxes[int(np.argmax(s))]


def iou(b, ref):
    x0 = np.maximum(b[:, 0], ref[0]); y0 = np.maximum(b[:, 1], ref[1])
    x1 = np.minimum(b[:, 0] + b[:, 2], ref[0] + ref[2]); y1 = np.minimum(b[:, 1] + b[:, 3], ref[1] + ref[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / (b[:, 2] * b[:, 3] + ref[2] * ref[3] - inter + 1e-6)


def detect(a):
    """Detected dancer crop embedding per clip centre for every video; on BRACE also report box hit rate vs the
    annotated dancer box and identity accuracy of 2-means on the detected crops (within annotated rounds)."""
    device = torch.device('cuda')
    det, emb = PersonDetector(device), Embedder(device)
    os.makedirs(os.path.join(a.features, 'appearance'), exist_ok=True)
    ids = a.ids or sorted(f[:-4] for f in os.listdir(os.path.join(a.features, 'clips')) if f.endswith('.npz'))
    for k, vid in enumerate(ids):
        dst = os.path.join(a.features, 'appearance', vid + '.npz')
        if os.path.exists(dst):
            continue
        clips = np.load(os.path.join(a.features, 'clips', vid + '.npz'))
        T, L = len(clips['rgb']), int(clips['clip_len'])
        frames = centre_frames(os.path.join(a.videos, vid + '.mp4'), T, L)
        dets = det(frames)
        E = np.zeros((T, 512), np.float16)
        B = np.zeros((T, 4), np.float32)
        has = np.zeros(T, np.int8)
        prev, idx, crops = None, [], []
        for t, (b, _) in enumerate(dets):
            d = pick_dancer(b, prev)
            if d is None:
                continue
            cr = crop(frames[t], d)
            if cr is None:
                continue
            prev = d
            B[t], has[t] = d, 1
            idx.append(t)
            crops.append(cr)
        if crops:
            E[idx] = emb(crops).astype(np.float16)
        np.savez(dst, emb=E, box=B, has=has)
        print(f'[{k + 1}/{len(ids)}] {vid}: T={T}, dancer found in {has.mean():.2f} of clips', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['oracle', 'detect'])
    p.add_argument('--brace', default='/data/mlavl/datasets/brace')
    p.add_argument('--videos', default='/data/mlavl/videos/brace')
    p.add_argument('--features', default='/data/mlavl/features/BRACE')
    p.add_argument('--ids', nargs='*')
    a = p.parse_args()
    oracle(a) if a.mode == 'oracle' else detect(a)
