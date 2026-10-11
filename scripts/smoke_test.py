"""Smoke test: build MLAVL with the precomputed text features and run one train step on random data.

Checks (1) the CLIP-free text-feature path, (2) --n_text/--n_query vs. feature-file consistency,
(3) PyG SAGEConv under torch.use_deterministic_algorithms(True), (4) backward pass.

    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY scripts/smoke_test.py --dataset rg --n_text 32 --clip-num 68
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY scripts/smoke_test.py --dataset fs1000 --n_text 42 --clip-num 95
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import options  # noqa: E402
from main import setup_seed  # noqa: E402
from models import loss as loss_mod  # noqa: E402
from models import model as model_mod  # noqa: E402


def main():
    args = options.parser.parse_args()
    setup_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} dataset={args.dataset} n_text={args.n_text} n_query={args.n_query} clip_num={args.clip_num}')

    net = model_mod.MLAVL(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder, args.n_decoder,
                          args.n_query, args.dropout, args, class_names=None).to(device)
    n_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'trainable params: {n_param / 1e6:.2f}M (paper: ~3.82M)')
    print(f'text_feature buffer: {tuple(net.text_feature.shape)} on {net.text_feature.device}')

    loss_fn = loss_mod.LossFun(args.alpha, args.margin, alpha_ce=args.alpha_ce)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    b, t = min(args.batch, 8), args.clip_num
    v = torch.randn(b, t, args.in_dim, device=device)
    a = torch.randn(b, t, 768, device=device)
    y = torch.rand(b, device=device)

    net.train()
    t0 = time.time()
    out = net(v, a)
    l = loss_fn(out['output'], y, out['embed'], out['embed2'], out['embed3'], out['embed4'],
                out['embed5'], out['embed6'], out['logits1'], out['logits2'], args)
    optim.zero_grad()
    l.backward()
    optim.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        print(f'max VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    print(f'output {tuple(out["output"].shape)} loss {l.item():.4f} step time {time.time() - t0:.2f}s')
    print('SMOKE TEST OK')


if __name__ == '__main__':
    main()
