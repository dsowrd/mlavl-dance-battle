"""Clip features for battle videos made exactly like the PAMFN RG / Fis-V features (ours, Phase 6 fix).

Our first extraction (tools/battle/extract_features.py: Swin-B Kinetics-400, short side 224 + centre square, HF AST
default preprocessing) does not live in the same feature space as the PAMFN features the reproduced MLAVL models were
trained on (correlation of per-dimension mean profiles with PAMFN: video 0.12, audio 0.45, vs 0.52 / 0.95 between
PAMFN RG and Fis-V). This follows PAMFN tools/extract_features.py:

  video  Video Swin-B Kinetics-600 + ImageNet-22k (tools/battle/vst_k600.py), 32 consecutive frames per clip,
         Resize(256) -> CenterCrop(224 h x 300 w) -> Resize(224 x 224), ImageNet mean / std, backbone + avg pool
  audio  AST AudioSet 0.4593 (same weights as HF MIT/ast-finetuned-audioset-10-10-0.4593); the waveform is split into
         as many equal segments as there are clips; per segment kaldi fbank (128 mel, hanning, 10 ms shift,
         htk_compat), normalised (x - 4.26) / (4.57 * 2), padded / cut to tm_dim frames; AST position embeddings cut
         to tm_dim time steps (centre slice, as in the original AST code); output (cls + dist) / 2

PAMFN's tm_dim and audio sample-rate arguments are not recorded; --tm_dim / --audio_sr are chosen by `probe` (the
setting whose per-dimension mean profile on BRACE correlates best with PAMFN's). Probe result (10 BRACE videos,
2026-10-18): audio tm_dim 64, 32 kHz: 0.892 / 0.890 with PAMFN RG / Fis-V (HF default preprocessing 0.45; PAMFN RG vs
Fis-V 0.95). Video: the mean-profile test is dominated by content (K600 + PAMFN preprocessing 0.21, K400 0.12), and
Kinetics-400 / -600 heads both classify PAMFN and our features correctly, so the video spaces are compatible.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/extract_features_pamfn.py probe
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/extract_features_pamfn.py extract --set brace --tm_dim T --audio_sr S
Output: {features}/clips_pamfn/{video_id}.npz, same keys as clips/.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vst_k600 import load_vst_k600  # noqa: E402

FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
SETS = {'brace': ('/data/mlavl/videos/brace', '/data/mlavl/features/BRACE'),
        'extra': ('/data/mlavl/videos/bcone_extra', '/data/mlavl/features/BCONE_EXTRA'),
        'wdsf': ('/data/mlavl/videos/wdsf', '/data/mlavl/features/WDSF')}   # Phase 5c: no clips/ pass, see extract()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def video_feats(vm, path, T, device, batch=8):
    vf = 'scale=-2:256,crop=300:224,scale=224:224'
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-i', path, '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         stdout=subprocess.PIPE)
    n = 224 * 224 * 3
    feats, clip, clips = [], [], []
    while len(feats) * batch + len(clips) < T:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        clip.append(np.frombuffer(b, np.uint8).reshape(224, 224, 3))
        if len(clip) == 32:
            clips.append(np.stack(clip))
            clip = []
            if len(clips) == batch:
                feats.append(run(vm, clips, device))
                clips = []
    p.kill()
    if clips:
        feats.append(run(vm, clips, device))
    return np.concatenate(feats)[:T]


@torch.no_grad()
def run(vm, clips, device):
    x = torch.from_numpy(np.stack(clips)).to(device).permute(0, 4, 1, 2, 3).float().div_(255)
    x = (x - MEAN.to(device)) / STD.to(device)
    with torch.autocast('cuda', dtype=torch.float16):
        return vm(x).float().cpu().numpy()


class PamfnAST:
    def __init__(self, tm_dim, device):
        from transformers import ASTConfig, ASTModel
        name = 'MIT/ast-finetuned-audioset-10-10-0.4593'
        full = ASTModel.from_pretrained(name)
        cfg = ASTConfig.from_pretrained(name)
        cfg.max_length = tm_dim
        m = ASTModel(cfg)
        sd = full.state_dict()
        pos = sd['embeddings.position_embeddings']                     # (1, 2 + 12 * 101, 768)
        f, t0 = 12, 101
        t = (tm_dim - 16) // 10 + 1
        grid = pos[:, 2:].reshape(1, f, t0, -1)
        s = t0 // 2 - t // 2
        sd['embeddings.position_embeddings'] = torch.cat([pos[:, :2], grid[:, :, s:s + t].reshape(1, f * t, -1)], 1)
        m.load_state_dict(sd)
        self.m, self.tm, self.device = m.eval().to(device), tm_dim, device

    @torch.no_grad()
    def __call__(self, wave, sr_fbank, num_segments):
        import torchaudio
        wave = wave - wave.mean()
        L = wave.shape[1] // num_segments
        fb = [torchaudio.compliance.kaldi.fbank(wave[:, i * L:(i + 1) * L], htk_compat=True, sample_frequency=sr_fbank,
                                                use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
                                                frame_shift=10) for i in range(num_segments)]
        fb = (torch.stack(fb) - 4.26) / (4.57 * 2)
        if fb.shape[1] < self.tm:
            fb = torch.nn.functional.pad(fb, (0, 0, 0, self.tm - fb.shape[1]))
        fb = fb[:, :self.tm]
        out = []
        for i in range(0, len(fb), 32):
            out.append(self.m(input_values=fb[i:i + 32].to(self.device)).pooler_output.float().cpu())
        return torch.cat(out).numpy()


def load_wave(path, sr):
    import torchaudio
    raw = subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', '48000', '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    w = torch.from_numpy(np.frombuffer(raw, np.float32).copy())[None]
    return torchaudio.transforms.Resample(48000, sr)(w) if sr != 48000 else w


def pamfn_profile(kind):
    import pickle  # noqa: F401
    fs = {'video': ['/data/mlavl/features/RG/Ball_rgb_VST.npy', '/data/mlavl/features/FISV/FISV_rgb_VST.npy'],
          'audio': ['/data/mlavl/features/RG/Ball_audio_AST.npy', '/data/mlavl/features/FISV/FISV_audio_AST.npy']}[kind]
    return [np.concatenate(list(np.load(f, allow_pickle=True).item().values())).mean(0) for f in fs]


def probe(a):
    device = torch.device('cuda')
    vids = sorted(glob.glob('/data/mlavl/videos/brace/*.mp4'))[:10]
    T = {v: len(np.load(v.replace('/videos/brace/', '/features/BRACE/clips/').replace('.mp4', '.npz'))['rgb']) for v in vids}
    pv = pamfn_profile('video')
    vm = load_vst_k600().eval().to(device)
    V = np.concatenate([video_feats(vm, v, T[v], device) for v in vids])
    print('video K600 + PAMFN preprocessing: corr with PAMFN RG %.3f, Fis-V %.3f (old extraction 0.12)' %
          tuple(np.corrcoef(V.mean(0), p)[0, 1] for p in pv))
    pa = pamfn_profile('audio')
    for tm in [64, 100, 128]:
        ast = PamfnAST(tm, device)
        for sr, sr_fb in [(16000, 16000), (32000, 32000), (16000, 48000), (32000, 48000)]:
            A = np.concatenate([ast(load_wave(v, sr), sr_fb, T[v]) for v in vids])
            print(f'audio tm_dim {tm}, resample {sr}, fbank sample_frequency {sr_fb}: corr with PAMFN RG '
                  f'{np.corrcoef(A.mean(0), pa[0])[0, 1]:.3f}, Fis-V {np.corrcoef(A.mean(0), pa[1])[0, 1]:.3f}', flush=True)


def extract(a):
    device = torch.device('cuda')
    videos, feats = SETS[a.set]
    os.makedirs(os.path.join(feats, a.out), exist_ok=True)
    vm = None if a.audio_only else load_vst_k600().eval().to(device)
    ast = PamfnAST(a.tm_dim, device)
    # Sets without an earlier clips/ pass (wdsf) take the clip grid from the video itself: clip i = frames
    # [32 i, 32 i + 32) at the native fps, trailing partial clip dropped (same grid as extract_features.py).
    standalone = not os.path.isdir(os.path.join(feats, 'clips'))
    if standalone:
        from extract_features import probe as probe_video
        archive = os.path.join(videos, 'downloaded.txt')      # only finished downloads (yt-dlp archive)
        ids = sorted(line.split()[1] for line in open(archive) if line.strip())
        ids = [v for v in ids if os.path.exists(os.path.join(videos, v + '.mp4'))]
    else:
        ids = sorted(f[:-4] for f in os.listdir(os.path.join(feats, 'clips')) if f.endswith('.npz'))
    ids = ids[a.shard::a.nshards]
    for k, vid in enumerate(ids):
        dst = os.path.join(feats, a.out, vid + '.npz')
        if os.path.exists(dst):
            continue
        if standalone:
            fps, _, _, n_frames = probe_video(os.path.join(videos, vid + '.mp4'))
            T = n_frames // 32
            c = {'clip_start_frame': np.arange(T) * 32, 'fps': fps, 'clip_len': 32}
        else:
            c = dict(np.load(os.path.join(feats, 'clips', vid + '.npz')))
            T = len(c['rgb'])
        path = os.path.join(videos, vid + '.mp4')
        if not a.audio_only:
            c['rgb'] = video_feats(vm, path, T, device).astype(np.float32)
        c['audio'] = ast(load_wave(path, a.audio_sr), a.fbank_sr, T).astype(np.float32)
        assert len(c['rgb']) == T and len(c['audio']) == T, (vid, len(c['rgb']), len(c['audio']), T)
        np.savez(dst + '.tmp.npz', **c)
        os.replace(dst + '.tmp.npz', dst)
        print(f'[{k + 1}/{len(ids)}] {vid}: T={T}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['probe', 'extract'])
    p.add_argument('--set', choices=list(SETS), default='brace')
    p.add_argument('--tm_dim', type=int, default=128)
    p.add_argument('--audio_sr', type=int, default=16000)
    p.add_argument('--fbank_sr', type=int, default=16000)
    p.add_argument('--audio_only', action='store_true',
                   help='keep the existing clips/ video features (K400 and K600 Swin-B share a feature space: Kinetics heads are '
                        'interchangeable on both PAMFN and our features) and only redo the audio the PAMFN way')
    p.add_argument('--out', default='clips_pamfn')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    probe(a) if a.mode == 'probe' else extract(a)
