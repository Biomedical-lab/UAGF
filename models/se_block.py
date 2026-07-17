"""Squeeze-and-Excitation (SE) block for channel recalibration.

Reference:
    Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition (CVPR), pp. 7132-7141.
"""

import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block that recalibrates channel-wise features.

    Args:
        in_dim: Number of input features.
        reduction: Reduction ratio for the bottleneck layer.
    """

    def __init__(self, in_dim: int, reduction: int = 16):
        super().__init__()
        mid_dim = max(1, in_dim // reduction)
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mid_dim, in_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.fc1(x)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w
