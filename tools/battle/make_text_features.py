"""CLIP text features for an MLAVL action/grade label CSV (ours, Phase 5).

MLAVL ships precomputed ViFi-CLIP text features (FS_/RG_/LOGO_text_feature.npy) but not the ViFi-CLIP weights.
For a new vocabulary (breaking moves) we encode the label sentences with the original OpenAI CLIP ViT-B/16 text
encoder (same backbone ViFi-CLIP fine-tunes; prompts are the raw CSV sentences, as in ViFi-CLIP's zero-shot path),
and save them in the same layout and format: torch.save of a (rows, 512) float tensor, rows in CSV order
[action-V M][action-A M][grade-AV 2N][grade-V N].

    HF_HOME=$MLAVL_DATA/cache/hf $MLAVL_PY -I tools/battle/make_text_features.py \
        ours/vocab/action-label_BREAKING.csv ours/vocab/BREAKING_text_feature.npy --match-norm RG_text_feature.npy [--compare RG_text_feature.npy --compare-csv action-label_RG.csv]
"""
import argparse

import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer

NAME = 'openai/clip-vit-base-patch16'


@torch.no_grad()
def encode(sentences):
    tok = CLIPTokenizer.from_pretrained(NAME)
    model = CLIPModel.from_pretrained(NAME).eval()
    x = tok(sentences, padding=True, return_tensors='pt')
    return model.get_text_features(**x).float()


def main(a):
    names = pd.read_csv(a.csv)['name'].tolist()
    feat = encode(names)
    if a.match_norm:  # rescale rows to the mean row norm of a shipped ViFi-CLIP file (vanilla CLIP norms are ~25% smaller)
        target = torch.load(a.match_norm, map_location='cpu', weights_only=True).float().norm(dim=1).mean()
        feat = feat * (target / feat.norm(dim=1).mean())
    torch.save(feat, a.out)
    print(f'{a.out}: {tuple(feat.shape)} from {len(names)} sentences, row norm {feat.norm(dim=1).mean():.2f}')
    if a.compare:
        ref = torch.load(a.compare, map_location='cpu', weights_only=True).float()
        mine = encode(pd.read_csv(a.compare_csv)['name'].tolist())
        cos = torch.nn.functional.cosine_similarity(ref, mine, dim=1)
        print(f'vanilla CLIP vs shipped {a.compare}: per-row cosine mean {cos.mean():.3f} (min {cos.min():.3f}); '
              f'row norms shipped {ref.norm(dim=1).mean():.2f} vs vanilla {mine.norm(dim=1).mean():.2f}')
        r = lambda m: torch.nn.functional.normalize(m, dim=1) @ torch.nn.functional.normalize(m, dim=1).T
        iu = torch.triu_indices(len(ref), len(ref), 1)
        print(f'similarity structure agreement (pearson of pairwise cosines): '
              f'{torch.corrcoef(torch.stack([r(ref)[iu[0], iu[1]], r(mine)[iu[0], iu[1]]]))[0, 1]:.3f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('out')
    p.add_argument('--compare')
    p.add_argument('--compare-csv')
    p.add_argument('--match-norm', help='shipped text feature file whose mean row norm the output is scaled to')
    main(p.parse_args())
