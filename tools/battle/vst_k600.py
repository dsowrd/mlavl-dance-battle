"""Video Swin-B Kinetics-600 + ImageNet-22k (the backbone of the PAMFN RG / Fis-V features) in torchvision's swin3d_b.

PAMFN (tools/extract_features.py in https://github.com/qinghuannn/PAMFN) extracts its VST features with
Video-Swin-Transformer `swin_base_patch244_window877_kinetics600_22k.pth` (mmaction2), backbone + global average
pooling. This maps that checkpoint onto torchvision's swin3d_b (same architecture, patch 2x4x4, window 8x7x7):
  backbone.patch_embed.*            -> patch_embed.*
  backbone.layers.i.blocks.j.*      -> features.{2i}.{j}.*   (mlp.fc1 -> mlp.0, mlp.fc2 -> mlp.3)
  backbone.layers.i.downsample.*    -> features.{2i+1}.*
  backbone.norm.*                   -> norm.*
  cls_head.fc_cls.*                 -> head.* (600 Kinetics classes; replaced by Identity for features)
Checkpoint: https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth
"""
import re

import torch

CKPT = '/data/mlavl/cache/vst/swin_base_patch244_window877_kinetics600_22k.pth'


def load_vst_k600(with_head=False):
    from torchvision.models.video import swin3d_b
    m = swin3d_b(num_classes=600)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    out = {}
    for k, v in sd.items():
        if k.startswith('cls_head.fc_cls.'):
            out[k.replace('cls_head.fc_cls.', 'head.')] = v
            continue
        k = k.replace('backbone.', '')
        m_ = re.match(r'layers\.(\d+)\.blocks\.(\d+)\.(.*)', k)
        if m_:
            i, j, rest = int(m_.group(1)), m_.group(2), m_.group(3)
            rest = rest.replace('mlp.fc1', 'mlp.0').replace('mlp.fc2', 'mlp.3')
            k = f'features.{2 * i}.{j}.{rest}'
        m_ = re.match(r'layers\.(\d+)\.downsample\.(.*)', k)
        if m_:
            k = f'features.{2 * int(m_.group(1)) + 1}.{m_.group(2)}'
        out[k] = v
    missing, unexpected = m.load_state_dict(out, strict=False)
    missing = [k for k in missing if not k.endswith('relative_position_index')]
    assert not missing and not unexpected, (missing, unexpected)
    if not with_head:
        m.head = torch.nn.Identity()
    return m
