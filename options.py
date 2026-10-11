# Modified from MLAVL (Xu et al., CVPR 2025), https://github.com/XuHuangbiao/MLAVL
# Changes: add --dataset, --text_feature, --use_clip, --alpha_ce, --seed so RG/Fis-V/LOGO can be selected
#          and the model runs without ViFi-CLIP weights (precomputed text features are loaded instead).
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--video-path', type=str, default='../FS1000/output_feature_fs1000_new')
parser.add_argument('--audio-path', type=str, default='../FS1000/ast_feature_fs1000_new')
parser.add_argument('--clip-num', type=int, default=95)

parser.add_argument('--train-label-path', type=str, default='../FS1000/train_fs1000_new.txt')
parser.add_argument('--test-label-path', type=str, default='../FS1000/val_fs1000_new.txt')

parser.add_argument('--action-type', type=str, default='TES')
parser.add_argument('--score-type', type=str, default='Total_Score')

parser.add_argument('--model-name', type=str, default='action_net', help='name used to save model and logs')
parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument("--test", action='store_true', help="only evaluate, don't train")

parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--optim', type=str, default='adam')

parser.add_argument("--lr-decay", type=str, default='cos', help='use what decay scheduler')
parser.add_argument("--decay-rate", type=float, default=0.01, help="lr decay rate")
parser.add_argument("--warmup", type=int, default=0, help="warmup epoch")

parser.add_argument('--in_dim', type=int, default=768)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_encoder', type=int, default=1)
parser.add_argument('--n_decoder', type=int, default=3)
parser.add_argument('--n_query', type=int, default=6)
parser.add_argument('--n_text', type=int, default=42)
parser.add_argument("--use_pe", type=bool, default=False)

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--margin', type=float, default=1.0)

parser.add_argument('--dropout', type=float, default=0.0)

# CLIP
parser.add_argument('--ARCH', type=str, default='ViT-B/16')
parser.add_argument('--DROP_PATH_RATE', type=float, default=0.)
parser.add_argument("--RETRAINED", type=bool, default=None)
parser.add_argument("--FIX_TEXT", type=bool, default=True)
parser.add_argument("--MULTI_VIEW_INFERENCE", type=bool, default=True)
parser.add_argument("--ZS_EVAL", type=bool, default=False)
parser.add_argument('--USE', type=str, default='both')
parser.add_argument("--PROMPT_MODEL", type=bool, default=False)
parser.add_argument('--N_CTX_VISION', type=int, default=0)
parser.add_argument('--N_CTX_TEXT', type=int, default=0)
parser.add_argument('--CTX_INIT', type=str, default=None)
parser.add_argument('--PROMPT_DEPTH_VISION', type=int, default=0)
parser.add_argument('--PROMPT_DEPTH_TEXT', type=int, default=1)
parser.add_argument('--pretrained_clip_weight', type=str,
                    default='./weights/k400_clip_complete_finetuned_30_epochs.pth')
parser.add_argument('--action_list', type=str, default='./action-label_FS.csv')
# ---- added for this project (see header) ----
parser.add_argument('--dataset', type=str, default='fs1000', choices=['fs1000', 'fisv', 'rg', 'logo'],
                    help='which Dataset class in datasets.py to use (upstream hard-coded FS1000Dataset)')
parser.add_argument('--text_feature', type=str, default=None,
                    help='precomputed CLIP text feature (torch.save file, e.g. ./RG_text_feature.npy). '
                         'Default: picked by --dataset (FS_/RG_/LOGO_text_feature.npy)')
parser.add_argument('--use_clip', action='store_true',
                    help='re-enable the ViFi-CLIP text encoder (needs --pretrained_clip_weight) instead of --text_feature')
parser.add_argument('--alpha_ce', type=float, default=0.01,
                    help='weight of the label-smoothing CE loss (paper: 0.01 FS1000, 0.1 Fis-V/RG; upstream fixed 0.01)')
parser.add_argument('--seed', type=int, default=0, help='random seed (upstream hard-coded 0)')
parser.add_argument('--score_range', type=int, default=None)
