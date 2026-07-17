"""Adaptive Fusion Module with uncertainty-guided gating.

This module implements the core UAGF pipeline:
  1. SE-refined backbone embeddings
  2. Branch-level uncertainty estimation (entropy, max probability, margin)
  3. MLP-based gating network for adaptive per-sample fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .se_block import SEBlock


class AdaptiveFusionModule(nn.Module):
    """Uncertainty-Guided Adaptive Gated Fusion module.

    Given three backbone embeddings, this module:
      - Refines each branch with an SE block,
      - Computes per-branch uncertainty features from calibrated softmax,
      - Learns per-sample gating weights via an MLP,
      - Produces a weighted fusion of the refined embeddings.

    Args:
        c_dim: ConvNeXt embedding dimension.
        s_dim: Swin Transformer embedding dimension.
        e_dim: EfficientNet embedding dimension.
        num_classes: Number of target classes.
        temperature: Gating softmax temperature.
        branch_temps: Tuple (T_c, T_s, T_e) for per-branch calibration.
        se_reduction: SE block reduction ratio.
    """

    def __init__(
        self,
        c_dim: int,
        s_dim: int,
        e_dim: int,
        num_classes: int,
        temperature: float = 1.0,
        branch_temps: tuple = None,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature

        # Per-branch calibration temperatures
        if branch_temps is None:
            branch_temps = (1.0, 1.0, 1.0)
        self.register_buffer("T_c", torch.tensor(float(branch_temps[0])))
        self.register_buffer("T_s", torch.tensor(float(branch_temps[1])))
        self.register_buffer("T_e", torch.tensor(float(branch_temps[2])))

        # SE blocks for channel recalibration
        self.se_c = SEBlock(c_dim, se_reduction)
        self.se_s = SEBlock(s_dim, se_reduction)
        self.se_e = SEBlock(e_dim, se_reduction)

        # Branch classification heads
        self.head_c = nn.Linear(c_dim, num_classes)
        self.head_s = nn.Linear(s_dim, num_classes)
        self.head_e = nn.Linear(e_dim, num_classes)

        # Gating MLP: input = concatenated SE features + 9D uncertainty vector
        gate_in = c_dim + s_dim + e_dim + 9
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    @staticmethod
    def _uncertainty_from_probs(probs: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty features from softmax probabilities.

        Returns a (N, 3) tensor: [entropy, max_prob, margin].
        """
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=1, keepdim=True)
        max_prob = torch.max(probs, dim=1, keepdim=True)[0]
        sorted_p = torch.topk(probs, 2, dim=1)[0]
        margin = sorted_p[:, 0:1] - sorted_p[:, 1:2]
        return torch.cat([entropy, max_prob, margin], dim=1)

    def forward(self, xc, xs, xe):
        """Forward pass.

        Args:
            xc: ConvNeXt features (N, c_dim).
            xs: Swin features (N, s_dim).
            xe: EfficientNet features (N, e_dim).

        Returns:
            fused: Weighted fused features (N, c_dim + s_dim + e_dim).
            weights: Per-sample gating weights (N, 3).
        """
        # SE refinement
        xc_att = self.se_c(xc)
        xs_att = self.se_s(xs)
        xe_att = self.se_e(xe)

        # Branch logits and temperature-scaled softmax
        prob_c = F.softmax(self.head_c(xc_att) / self.T_c, dim=1)
        prob_s = F.softmax(self.head_s(xs_att) / self.T_s, dim=1)
        prob_e = F.softmax(self.head_e(xe_att) / self.T_e, dim=1)

        # Uncertainty features (3 measures x 3 branches = 9D)
        u_c = self._uncertainty_from_probs(prob_c)
        u_s = self._uncertainty_from_probs(prob_s)
        u_e = self._uncertainty_from_probs(prob_e)

        # Gating
        combined = torch.cat([xc_att, xs_att, xe_att, u_c, u_s, u_e], dim=1)
        z = self.gate_mlp(combined)
        weights = F.softmax(z / self.temperature, dim=1)

        # Weighted fusion
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        fused = torch.cat([w1 * xc_att, w2 * xs_att, w3 * xe_att], dim=1)
        return fused, weights

    def set_branch_temps(self, T_c: float, T_s: float, T_e: float):
        """Update per-branch calibration temperatures."""
        self.T_c.fill_(T_c)
        self.T_s.fill_(T_s)
        self.T_e.fill_(T_e)


class LabelSmoothingCE(nn.Module):
    """Label smoothing cross-entropy loss."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        logprobs = F.log_softmax(pred, dim=-1)
        nll = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -logprobs.mean(dim=-1)
        return self.confidence * nll.mean() + self.smoothing * smooth.mean()


class ProxyClassifier(nn.Module):
    """Lightweight linear classifier used during gating module training."""

    def __init__(self, fused_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(fused_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
