"""Fine-tune AST (AudioSet 0.4593, PAMFN preprocessing) toward breaking music: beat position and tempo per clip
(ours, Phase 5b step 2: adapt the audio backbone).

The battle model sees one 768-d AST vector per 32-frame clip. AudioSet AST is a sound-event tagger; for judging
musicality the audio stream should carry *where the beats are* inside the clip, so the clip-wise AV fusion can line
them up with the dancer's motion. Targets come from the librosa beat tracker run on every video
({features}/beats/{video_id}.npy, beat times in s; median |dt| to BRACE's Essentia beats checked in
tools/battle/extract_pose_features.py). They are signal-processing labels, never judges' decisions, so all 282
battle videos can be used.

Per clip (PAMFN split: the waveform cut into T equal segments, one per video clip; the AST sees the first tm_dim = 64
fbank frames = 0.64 s of each segment):
  beat  8 bins of 80 ms over the 0.64 s window, 1 if a beat falls in the bin (BCE)
  tempo log2(BPM) from the median inter-beat interval within +-5 s (MSE), masked where fewer than 4 beats
The last --blocks AST layers + final LayerNorm are trained with two small heads; the earlier layers stay frozen.
--blocks 0 trains the heads only (= linear probe of the frozen features, the baseline the fine-tuning must beat).
Hold-out: BRACE 2020 videos + every 10th extra video (by sorted id).

    $MLAVL_PY -I tools/battle/finetune_ast.py fbank --workers 12        # CPU; {features}/ast_fbank/{vid}.npz
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/finetune_ast.py train --name ast_beat --blocks 2
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/finetune_ast.py extract --name ast_beat --set brace --src clips_ft --out clips_ftva
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features_pamfn import PamfnAST, load_wave  # noqa: E402

SETS = {'brace': ('/data/mlavl/videos/brace', '/data/mlavl/features/BRACE'),
        'extra': ('/data/mlavl/videos/bcone_extra', '/data/mlavl/features/BCONE_EXTRA')}
TM, SR, NB = 64, 32000, 8
WIN = TM * 0.01                       # seconds of audio the AST sees per clip


# ---------------------------------------------------------------- fbank + targets
def fbank_video(job):
    import torchaudio
    torch.set_num_threads(1)
    s, vid = job
    videos, feats = SETS[s]
    dst = os.path.join(feats, 'ast_fbank', vid + '.npz')
    if os.path.exists(dst):
        return vid, -1
    T = len(np.load(os.path.join(feats, 'clips_pamfn', vid + '.npz'))['audio'])
    wave = load_wave(os.path.join(videos, vid + '.mp4'), SR)
    wave = wave - wave.mean()
    L = wave.shape[1] // T
    fb = [torchaudio.compliance.kaldi.fbank(wave[:, i * L:(i + 1) * L], htk_compat=True, sample_frequency=SR,
                                            use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
                                            frame_shift=10) for i in range(T)]
    fb = (torch.stack(fb) - 4.26) / (4.57 * 2)
    fb = nn.functional.pad(fb, (0, 0, 0, max(0, TM - fb.shape[1])))[:, :TM]
    beats = np.load(os.path.join(feats, 'beats', vid + '.npy'))
    start = np.arange(T) * L / SR
    beat = np.zeros((T, NB), np.float32)
    tempo = np.full(T, np.nan, np.float32)
    for i, t0 in enumerate(start):
        b = beats[(beats >= t0) & (beats < t0 + WIN)]
        beat[i, ((b - t0) / WIN * NB).astype(int).clip(0, NB - 1)] = 1
        near = beats[(beats >= t0 - 5) & (beats < t0 + WIN + 5)]
        if len(near) >= 4:
            tempo[i] = np.log2(60 / np.median(np.diff(near)))
    np.savez(dst, fbank=fb.numpy().astype(np.float16), beat=beat, tempo=tempo)
    return vid, T


def fbank(a):
    jobs = []
    for s, (_, feats) in SETS.items():
        os.makedirs(os.path.join(feats, 'ast_fbank'), exist_ok=True)
        jobs += [(s, f[:-4]) for f in sorted(os.listdir(os.path.join(feats, 'clips_pamfn'))) if f.endswith('.npz')]
    with Pool(a.workers) as p:
        for k, (vid, T) in enumerate(p.imap_unordered(fbank_video, jobs)):
            print(f'[{k + 1}/{len(jobs)}] {vid}: T={T}', flush=True)


# ---------------------------------------------------------------- model
class BeatAST(nn.Module):
    def __init__(self, blocks, device):
        super().__init__()
        self.ast = PamfnAST(TM, device).m           # HF ASTModel with PAMFN's tm_dim position embeddings
        n = len(self.ast.encoder.layer)
        self.k = n - blocks
        for name, p in self.ast.named_parameters():
            p.requires_grad = name.startswith('layernorm.') and blocks > 0 or any(
                name.startswith(f'encoder.layer.{j}.') for j in range(self.k, n))
        self.beat = nn.Linear(768, NB)
        self.tempo = nn.Linear(768, 1)

    def embed(self, x):
        m = self.ast
        with torch.no_grad():
            h = m.embeddings(x)
            for j in range(self.k):
                h = m.encoder.layer[j](h)[0]
        for j in range(self.k, len(m.encoder.layer)):
            h = m.encoder.layer[j](h)[0]
        h = m.layernorm(h)
        return (h[:, 0] + h[:, 1]) / 2               # = ASTModel.pooler_output, PAMFN's feature

    def forward(self, x):
        e = self.embed(x)
        return self.beat(e), self.tempo(e)[:, 0]


def load_split():
    tr, te = [], []
    for s, (_, feats) in SETS.items():
        files = sorted(glob.glob(os.path.join(feats, 'ast_fbank', '*.npz')))
        for i, f in enumerate(files):
            vid = os.path.basename(f)[:-4]
            hold = (s == 'extra' and i % 10 == 0) or (s == 'brace' and vid in BRACE_2020)
            d = np.load(f)
            item = (torch.from_numpy(d['fbank']), torch.from_numpy(d['beat']), torch.from_numpy(d['tempo']))
            (te if hold else tr).append(item)
    cat = lambda xs: tuple(torch.cat([x[j] for x in xs]) for j in range(3))
    return cat(tr), cat(te)


BRACE_2020 = set()


@torch.no_grad()
def evaluate(model, data, device, bs=256):
    from sklearn.metrics import roc_auc_score
    model.eval()
    X, B, Tm = data
    P, Q = [], []
    for i in range(0, len(X), bs):
        with torch.autocast('cuda', dtype=torch.float16):
            p, q = model(X[i:i + bs].to(device).float())
        P.append(p.float().sigmoid().cpu())
        Q.append(q.float().cpu())
    P, Q = torch.cat(P).numpy(), torch.cat(Q).numpy()
    m = ~np.isnan(Tm.numpy())
    err = np.abs(Q[m] - Tm.numpy()[m])                                # in octaves
    return roc_auc_score(B.numpy().ravel(), P.ravel()), float(np.median(err) * 100), float(np.mean(err < np.log2(1.04)))


def train(a):
    import pandas as pd
    torch.manual_seed(a.seed)
    device = torch.device('cuda')
    vi = pd.read_csv('/data/mlavl/datasets/brace/repo/videos_info.csv')
    BRACE_2020.update(vi[vi.year == 2020].video_id)
    (X, B, Tm), te = load_split()
    print(f'{len(X)} train clips, {len(te[0])} held-out clips; beat rate {B.mean():.3f} per 80 ms bin; '
          f'tempo known for {np.mean(~np.isnan(Tm.numpy())):.2f}', flush=True)
    model = BeatAST(a.blocks, device).to(device)
    bb = [p for p in model.ast.parameters() if p.requires_grad]
    heads = list(model.beat.parameters()) + list(model.tempo.parameters())
    opt = torch.optim.AdamW([{'params': bb, 'lr': a.lr}, {'params': heads, 'lr': a.lr * 20}], weight_decay=0.01)
    steps = a.epochs * (len(X) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr * 20], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    print(f'trainable AST params {sum(p.numel() for p in bb) / 1e6:.1f}M', flush=True)
    best = -1
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X) - a.batch + 1, a.batch):
            j = perm[i:i + a.batch]
            x, b, t = X[j].to(device).float(), B[j].to(device), Tm[j].to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                pb, pt = model(x)
            m = ~torch.isnan(t)
            loss = nn.functional.binary_cross_entropy_with_logits(pb.float(), b) + \
                (((pt.float() - t.nan_to_num()) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        auc, terr, tacc = evaluate(model, te, device)
        print(f'epoch {ep}: train loss {tot / (len(X) // a.batch):.4f} | held-out beat AUC {auc:.3f}, tempo median error '
              f'{terr:.1f} cents/100 (octave x100), within 4%: {tacc:.3f}', flush=True)
        if auc > best:
            best = auc
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith('ast.') or
                  k.startswith('ast.layernorm.') or any(k.startswith(f'ast.encoder.layer.{j}.')
                                                         for j in range(model.k, len(model.ast.encoder.layer)))}
            torch.save({'state': sd, 'blocks': a.blocks, 'epoch': ep, 'auc': auc, 'tempo_acc': tacc}, f'ckpt/{a.name}.pt')
    print(f'best held-out beat AUC {best:.3f} -> ckpt/{a.name}.pt', flush=True)


@torch.no_grad()
def extract(a):
    device = torch.device('cuda')
    ck = torch.load(f'ckpt/{a.name}.pt', map_location='cpu', weights_only=False)
    model = BeatAST(ck['blocks'], device)
    missing, unexpected = model.load_state_dict(ck['state'], strict=False)
    assert not unexpected and all(k.startswith('ast.') for k in missing)
    model = model.eval().to(device)
    print(f"loaded ckpt/{a.name}.pt (epoch {ck['epoch']}, held-out beat AUC {ck['auc']:.3f})", flush=True)
    _, feats = SETS[a.set]
    os.makedirs(os.path.join(feats, a.out), exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(feats, a.src)) if f.endswith('.npz'))
    for k, vid in enumerate(ids):
        dst = os.path.join(feats, a.out, vid + '.npz')
        if os.path.exists(dst):
            continue
        c = dict(np.load(os.path.join(feats, a.src, vid + '.npz')))
        x = torch.from_numpy(np.load(os.path.join(feats, 'ast_fbank', vid + '.npz'))['fbank'])
        out = []
        for i in range(0, len(x), 256):
            with torch.autocast('cuda', dtype=torch.float16):
                out.append(model.embed(x[i:i + 256].to(device).float()).float().cpu())
        c['audio'] = torch.cat(out).numpy().astype(np.float32)
        assert len(c['audio']) == len(c['rgb'])
        np.savez(dst + '.tmp.npz', **c)
        os.replace(dst + '.tmp.npz', dst)
        print(f'[{k + 1}/{len(ids)}] {vid}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['fbank', 'train', 'extract'])
    p.add_argument('--name', default='ast_beat')
    p.add_argument('--blocks', type=int, default=2, help='last AST layers to train (0 = heads only, linear probe)')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--set', choices=list(SETS), default='brace')
    p.add_argument('--src', default='clips_ft', help='clip-feature subdir whose rgb is kept')
    p.add_argument('--out', default='clips_ftva')
    a = p.parse_args()
    {'fbank': fbank, 'train': train, 'extract': extract}[a.mode](a)
