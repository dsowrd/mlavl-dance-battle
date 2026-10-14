"""Pose / kick-beat branch for MLAVL (ours, Phase 5).

The battle dataset appends the 115-d clip pose feature (tools/battle/extract_pose_features.py: 114 values + has_pose)
to the 1024-d Video Swin feature. `add_pose_branch(model)` replaces MLAVL's `proj_visual` with a module that splits
that input, projects the video part with the original (pre-trained) projection and the pose part with a new 1x1-conv
projection, and adds them: visual token = proj_visual(video) + g * proj_pose(pose). The gate g starts at 0, so the
model starts exactly as the initialisation checkpoint and learns how much pose to use; g is trained with the scalar
lr (see ours/train_battle.py). Upstream models/model.py is not modified.
"""
import torch
from torch import nn

POSE_DIM = 115


class PoseAugmentedProjection(nn.Module):
    def __init__(self, base, in_dim=1024, pose_dim=POSE_DIM, hidden=256):
        super().__init__()
        self.base, self.in_dim = base, in_dim
        self.proj_pose = nn.Sequential(
            nn.Conv1d(pose_dim, 128, kernel_size=1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, hidden, kernel_size=1), nn.BatchNorm1d(hidden))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):                       # x: (B, in_dim + pose_dim, T), as MLAVL calls proj_visual
        return self.base(x[:, :self.in_dim]) + self.gate * self.proj_pose(x[:, self.in_dim:])


def add_pose_branch(model, in_dim=1024):
    model.proj_visual = PoseAugmentedProjection(model.proj_visual, in_dim).to(next(model.parameters()).device)
    return model
