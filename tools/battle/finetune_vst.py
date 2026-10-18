"""Fine-tune Video Swin-B (Kinetics-600, the PAMFN backbone) on BRACE move types, then re-extract battle features
(ours, Phase 5b step 2: adapt the video backbone to breaking).

Frozen features already separate the three BRACE move types to some extent (leave-one-year-out logistic regression on
clips inside the 1,352 annotated segments: whole frame 64% of clips / 74% of segments, dancer-centred crop 70% / 82%;
chance 34%). Here the last Swin stage (features.6, 2 blocks) + final norm are trained on dancer-centred clips with a
3-way head (toprock / powermove / footwork); everything before stays frozen (and runs without gradients).

Input = exactly what tools/battle/extract_crop_features.py feeds the backbone: the 32 frames of grid clip t
(BRACE frames [32 t, 32 t + 32)) cropped with the clip's dancer box. The cache stores the crop at box scale 1.6 as
256 x 256 JPEGs; training takes a random square of 75-100 % (box scale 1.2-1.6) + horizontal flip, evaluation the
centre 224 (= box scale 1.4, the extraction crop). Split: BRACE's own sequence-level train / test split (move labels
only; winner labels are never used, and the 201 extra battles have no move labels at all).

    $MLAVL_PY -I tools/battle/finetune_vst.py cache                         # CPU, ~/data/mlavl/features/BRACE/vst_ft_cache
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/finetune_vst.py train --name vst_move
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/finetune_vst.py extract --name vst_move --set brace   (and extra)
Extract output: {features}/clips_ft/{video_id}.npz: rgb (T, 1024) from the fine-tuned backbone (pooled, before the
head), audio copied from clips_pamfn (PAMFN-matched AST), other keys as clips/.
"""
import argparse
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_crop_features import H, W, clip_boxes, frames  # noqa: E402
from estimate_pose import SETS  # noqa: E402
from vst_k600 import load_vst_k600  # noqa: E402

BRACE = '/data/mlavl/datasets/brace/repo/annotations'
FEAT = '/data/mlavl/features/BRACE'
CACHE = os.path.join(FEAT, 'vst_ft_cache')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
CACHE_SCALE, EXTRACT_SCALE = 1.6, 1.4


def square(box, scale):
    """Same as extract_crop_features.square, with the scale as an argument (box in 1920x1080 px, crop in 960x540)."""
    x, y, w, h = box / 2
    s = min(max(max(w, h) * scale, 64), H)
    cx, cy = x + w / 2, y + h / 2
    return int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s)), int(s)


# ---------------------------------------------------------------- cache
def cache_video(args):
    from torchvision.io import encode_jpeg
    torch.set_num_threads(1)
    vid, rows = args
    dst = os.path.join(CACHE, vid + '.pkl')
    if os.path.exists(dst):
        return vid, -1
    T = len(np.load(os.path.join(FEAT, 'clips', vid + '.npz'))['rgb'])
    boxes, _ = clip_boxes(vid, FEAT, T)
    want = {}
    for r in rows:
        for t in range(r['start_frame'] // 32, min(r['end_frame'] // 32 + 1, T)):
            if 32 * t >= r['start_frame'] and 32 * t + 32 <= r['end_frame']:
                want[t] = r
    out, buf, last = [], [], max(want, default=-1)
    for fi, f in enumerate(frames(os.path.join('/data/mlavl/videos/brace', vid + '.mp4'))):
        t = fi // 32
        if t >= T or t > last:
            break
        if t not in want:
            continue
        x0, y0, s = square(boxes[t], CACHE_SCALE)
        im = torch.from_numpy(f[y0:y0 + s, x0:x0 + s].copy()).permute(2, 0, 1)
        im = nn.functional.interpolate(im[None].float(), size=(256, 256), mode='bilinear', align_corners=False)[0]
        buf.append(encode_jpeg(im.round().clamp(0, 255).to(torch.uint8), quality=90).numpy().tobytes())
        if fi % 32 == 31:
            r = want[t]
            out.append({'t': t, 'label': r['dance_type_id'], 'seg': r['uid'], 'split': r['split'], 'year': r['year'],
                        'jpg': buf})
            buf = []
    pickle.dump(out, open(dst + '.tmp', 'wb'))
    os.replace(dst + '.tmp', dst)
    return vid, len(out)


def cache(a):
    os.makedirs(CACHE, exist_ok=True)
    seg = pd.read_csv(f'{BRACE}/segments.csv')
    test = set(pd.read_csv(f'{BRACE}/sequences_test.csv').uid)
    seg['split'] = [('test' if f'{v}.{s}' in test else 'train') for v, s in zip(seg.video_id, seg.seq_idx)]
    jobs = [(v, g.to_dict('records')) for v, g in seg.groupby('video_id')]
    with Pool(a.workers) as p:
        for k, (vid, n) in enumerate(p.imap_unordered(cache_video, jobs)):
            print(f'[{k + 1}/{len(jobs)}] {vid}: {n} clips', flush=True)


# ---------------------------------------------------------------- train
class Clips(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from torchvision.io import decode_jpeg
        it = self.items[i]
        x = torch.stack([decode_jpeg(torch.frombuffer(b, dtype=torch.uint8)) for b in it['jpg']])   # T C 256 256
        if self.train:
            s = int(256 * np.random.uniform(0.75, 1.0))
            x0, y0 = np.random.randint(0, 256 - s + 1, 2)
            x = x[:, :, y0:y0 + s, x0:x0 + s]
            if np.random.rand() < 0.5:
                x = x.flip(-1)
        else:
            s = round(256 * EXTRACT_SCALE / CACHE_SCALE)
            o = (256 - s) // 2
            x = x[:, :, o:o + s, o:o + s]
        x = nn.functional.interpolate(x.float(), size=(224, 224), mode='bilinear', align_corners=False)
        return x.permute(1, 0, 2, 3).div_(255), it['label'], i                                 # C T 224 224


class MoveNet(nn.Module):
    def __init__(self, train_from=6, n_cls=3):
        super().__init__()
        self.vm = load_vst_k600()
        self.vm.head = nn.Identity()
        self.k = train_from
        for n, p in self.vm.named_parameters():
            p.requires_grad = n.startswith('norm.') or any(n.startswith(f'features.{j}.') for j in range(train_from, 7))
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, n_cls))

    def embed(self, x):
        vm = self.vm
        with torch.no_grad():
            h = vm.pos_drop(vm.patch_embed(x))
            for j in range(self.k):
                h = vm.features[j](h)
        for j in range(self.k, 7):
            h = vm.features[j](h)
        h = vm.norm(h).permute(0, 4, 1, 2, 3)
        return torch.flatten(vm.avgpool(h), 1)

    def forward(self, x):
        return self.head(self.embed(x))


def norm(x, device):
    return (x.to(device, non_blocking=True) - MEAN.to(device)) / STD.to(device)


@torch.no_grad()
def evaluate(model, dl, items, device):
    model.eval()
    P = np.zeros((len(items), 3))
    for x, y, i in dl:
        with torch.autocast('cuda', dtype=torch.float16):
            P[i.numpy()] = model(norm(x, device)).float().softmax(1).cpu().numpy()
    y = np.array([it['label'] for it in items])
    d = pd.DataFrame(P)
    d['seg'], d['y'] = [it['seg'] for it in items], y
    g = d.groupby('seg').mean()
    return float((P.argmax(1) == y).mean()), float((g[[0, 1, 2]].values.argmax(1) == g.y.values).mean())


def train(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device('cuda')
    items = [it for f in sorted(glob.glob(f'{CACHE}/*.pkl')) for it in pickle.load(open(f, 'rb'))]
    tr = [it for it in items if it['split'] == 'train']
    te = [it for it in items if it['split'] == 'test']
    print(f'{len(tr)} train clips / {len({i["seg"] for i in tr})} segments, {len(te)} test clips / '
          f'{len({i["seg"] for i in te})} segments', flush=True)
    dl = torch.utils.data.DataLoader(Clips(tr, True), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     drop_last=True, pin_memory=True, persistent_workers=True)
    dte = torch.utils.data.DataLoader(Clips(te, False), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    model = MoveNet(a.train_from).to(device)
    bb = [p for p in model.vm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': bb, 'lr': a.lr}, {'params': model.head.parameters(), 'lr': a.lr * 10}],
                            weight_decay=0.05)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr * 10], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    print(f'trainable backbone params {sum(p.numel() for p in bb) / 1e6:.1f}M', flush=True)
    c0, s0 = evaluate(model, dte, te, device)
    print(f'epoch -1 (random head): test clip {c0:.3f} segment {s0:.3f}', flush=True)
    best = -1
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in dl:
            with torch.autocast('cuda', dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(norm(x, device)), y.to(device), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        c, s = evaluate(model, dte, te, device)
        print(f'epoch {ep}: train loss {tot / n:.3f} | test clip {c:.3f} segment {s:.3f}', flush=True)
        if s > best:
            best = s
            sd = {k: v for k, v in model.state_dict().items()
                  if k.startswith('head.') or k.startswith('vm.norm.') or
                  any(k.startswith(f'vm.features.{j}.') for j in range(a.train_from, 7))}
            torch.save({'state': sd, 'train_from': a.train_from, 'epoch': ep, 'clip_acc': c, 'seg_acc': s},
                       f'ckpt/{a.name}.pt')
    print(f'best test segment acc {best:.3f} -> ckpt/{a.name}.pt', flush=True)


# ---------------------------------------------------------------- extract
@torch.no_grad()
def extract(a):
    device = torch.device('cuda')
    torch.set_num_threads(4)
    ck = torch.load(f'ckpt/{a.name}.pt', map_location='cpu', weights_only=False)
    model = MoveNet(ck['train_from'])
    missing, unexpected = model.load_state_dict(ck['state'], strict=False)
    assert not unexpected and all(k.startswith('vm.') for k in missing)
    model = model.eval().to(device)
    print(f"loaded ckpt/{a.name}.pt (epoch {ck['epoch']}, test segment acc {ck['seg_acc']:.3f})", flush=True)
    videos, features = SETS[a.set]
    out_dir = os.path.join(features, a.out)
    os.makedirs(out_dir, exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(features, 'clips_pamfn')) if f.endswith('.npz'))[a.shard::a.nshards]
    for k, vid in enumerate(ids):
        dst = os.path.join(out_dir, vid + '.npz')
        if os.path.exists(dst):
            continue
        c = dict(np.load(os.path.join(features, 'clips_pamfn', vid + '.npz')))
        T = len(c['rgb'])
        boxes, src = clip_boxes(vid, features, T)
        feats, buf, batch, t = [], [], [], 0

        def flush():
            xb = norm(torch.stack(batch), device)
            with torch.autocast('cuda', dtype=torch.float16):
                feats.append(model.embed(xb).float().cpu().numpy())

        for f in frames(os.path.join(videos, vid + '.mp4')):
            if t >= T:
                break
            if not buf:
                x0, y0, s = square(boxes[t], EXTRACT_SCALE)
            buf.append(f[y0:y0 + s, x0:x0 + s])
            if len(buf) == 32:   # numpy stack: torch.stack of 32 CPU frames took 0.6 s under CPU contention
                x = torch.from_numpy(np.stack(buf)).to(device).permute(3, 0, 1, 2).float().div_(255)
                batch.append(nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False))
                buf, t = [], t + 1
                if len(batch) == 8:
                    flush()
                    batch = []
        if batch:
            flush()
        rgb = np.concatenate(feats)
        assert len(rgb) == T, (vid, len(rgb), T)
        c['rgb'], c['box_src'] = rgb.astype(np.float32), src
        np.savez(dst + '.tmp.npz', **c)
        os.replace(dst + '.tmp.npz', dst)
        print(f'[{k + 1}/{len(ids)}] {vid}: T={T}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['cache', 'train', 'extract'])
    p.add_argument('--name', default='vst_move')
    p.add_argument('--train_from', type=int, default=6, help='first torchvision features.<i> to train (6 = last stage)')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--set', choices=list(SETS), default='brace')
    p.add_argument('--out', default='clips_ft')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    {'cache': cache, 'train': train, 'extract': extract}[a.mode](a)
