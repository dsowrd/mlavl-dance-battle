"""Clip-level video (Video Swin-B) and audio (AST) features for the BRACE battle videos (ours, Phase 4).

Mirrors the feature layout MLAVL is trained on for RG / Fis-V (PAMFN release): one feature vector per
non-overlapping 32-frame clip, video 1024-d (Video Swin-B, Kinetics-400 + ImageNet-22k), audio 768-d
(AST fine-tuned on AudioSet) computed on the audio of the same clip. Clips follow each video's native frame rate
(25 or 29.97 fps, see videos_info.csv), so clip i covers BRACE frames [32*i, 32*i + 32).

    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=3 $MLAVL_PY -I tools/battle/extract_features.py \
        --videos $MLAVL_DATA/videos/brace --out $MLAVL_DATA/features/BRACE [--ids ID ...]

Output per video: {out}/clips/{video_id}.npz with
    rgb (T, 1024) float32, audio (T, 768) float32, clip_start_frame (T,) int, fps, clip_len
Resumable: videos whose .npz exists are skipped.

Preprocessing: frames are resized to 224 px on the short side and centre-cropped to 224x224 (battle cameras keep
the dancer near the centre), normalised with ImageNet mean/std as in torchvision's Swin3D preset. A trailing partial
clip shorter than 32 frames is dropped. Audio: mono 16 kHz; each clip's 1.07-1.28 s segment goes through the AST
feature extractor (zero-padded to its 10.24 s input) and the pooled 768-d output is kept.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
FFPROBE = os.environ.get('FFPROBE', '/home/irl/anaconda3/bin/ffprobe')


def probe(path):
    out = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                          '-show_entries', 'stream=r_frame_rate,width,height,nb_read_packets', '-of', 'json', path],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    return float(num) / float(den), int(s['width']), int(s['height']), int(s['nb_read_packets'])


def frames(path, size=224):
    """Yield uint8 frames (size, size, 3), short side resized to `size`, centre crop."""
    vf = f"scale='if(gt(iw,ih),-2,{size})':'if(gt(iw,ih),{size},-2)',crop={size}:{size}"
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-i', path, '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         stdout=subprocess.PIPE)
    n = size * size * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(size, size, 3)
    p.stdout.close()
    p.wait()


def audio(path, sr=16000):
    out = subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True, check=True)
    return np.frombuffer(out.stdout, np.float32)


def build_models(device):
    from torchvision.models.video import swin3d_b, Swin3D_B_Weights
    from transformers import ASTFeatureExtractor, ASTModel
    vm = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
    vm.head = torch.nn.Identity()           # 1024-d pooled feature
    vm = vm.eval().to(device)
    name = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    fe = ASTFeatureExtractor.from_pretrained(name)
    am = ASTModel.from_pretrained(name).eval().to(device)
    return vm, fe, am


@torch.no_grad()
def video_features(vm, path, clip_len, batch, device):
    clips, cur, feats = [], [], []
    for f in frames(path):
        cur.append(f)
        if len(cur) == clip_len:
            clips.append(np.stack(cur))
            cur = []
            if len(clips) == batch:
                feats.append(run_video(vm, clips, device))
                clips = []
    if clips:
        feats.append(run_video(vm, clips, device))
    return np.concatenate(feats) if feats else np.zeros((0, 1024), np.float32)


def run_video(vm, clips, device):
    x = torch.from_numpy(np.stack(clips)).to(device)          # B T H W C uint8
    x = x.permute(0, 4, 1, 2, 3).float().div_(255)             # B C T H W
    x = (x - MEAN.to(device)) / STD.to(device)
    with torch.autocast('cuda', dtype=torch.float16):
        y = vm(x)
    return y.float().cpu().numpy()


@torch.no_grad()
def audio_features(fe, am, wav, starts, clip_len, fps, device, sr=16000, batch=32):
    segs = []
    for s in starts:
        a, b = int(round(s / fps * sr)), int(round((s + clip_len) / fps * sr))
        seg = wav[a:b]
        segs.append(seg if len(seg) else np.zeros(1, np.float32))
    out = []
    for i in range(0, len(segs), batch):
        inp = fe(segs[i:i + batch], sampling_rate=sr, return_tensors='pt')
        with torch.autocast('cuda', dtype=torch.float16):
            y = am(input_values=inp['input_values'].to(device)).pooler_output
        out.append(y.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def main(a):
    device = torch.device('cuda')
    os.makedirs(os.path.join(a.out, 'clips'), exist_ok=True)
    # finished downloads only: yt-dlp's intermediate files are '<id>.f137.mp4' / '<id>.temp.mp4' (ids have no '.')
    ids = a.ids or sorted(f[:-4] for f in os.listdir(a.videos) if f.endswith('.mp4') and '.' not in f[:-4])
    ids = ids[a.shard::a.nshards]          # ids starting with '-' cannot go through --ids; shard the default list
    vm, fe, am = build_models(device)
    for k, vid in enumerate(ids):
        dst = os.path.join(a.out, 'clips', vid + '.npz')
        if os.path.exists(dst):
            continue
        path = os.path.join(a.videos, vid + '.mp4')
        t0 = time.time()
        fps, w, h, nf = probe(path)
        rgb = video_features(vm, path, a.clip_len, a.batch, device)
        starts = np.arange(len(rgb)) * a.clip_len
        aud = audio_features(fe, am, audio(path), starts, a.clip_len, fps, device)
        assert len(aud) == len(rgb), (len(aud), len(rgb))
        np.savez(dst + '.tmp.npz', rgb=rgb.astype(np.float32), audio=aud.astype(np.float32),
                 clip_start_frame=starts, fps=fps, clip_len=a.clip_len, n_frames=nf)
        os.replace(dst + '.tmp.npz', dst)
        print(f'[{k + 1}/{len(ids)}] {vid}: {nf} frames @ {fps:.2f} fps -> T={len(rgb)} '
              f'({time.time() - t0:.0f}s, max VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--videos', default='/data/mlavl/videos/brace')
    p.add_argument('--out', default='/data/mlavl/features/BRACE')
    p.add_argument('--ids', nargs='*')
    p.add_argument('--clip-len', type=int, default=32)
    p.add_argument('--batch', type=int, default=8, help='clips per Video Swin forward')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    main(p.parse_args())
