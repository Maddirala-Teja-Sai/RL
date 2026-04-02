"""LIDAR-based observation encoder.

Encodes LIDAR ray readings + agent state vector into a feature representation.
"""

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Encodes LIDAR + state observations into a fixed-size feature vector.

    Architecture:
        obs (LIDAR+state) → Linear(256) → ReLU → Linear(256) → ReLU → 256-dim
    """

    def __init__(self, obs_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        """
        Args:
            obs: (B, obs_dim) — LIDAR readings + state vector concatenated
        Returns:
            features: (B, output_dim)
        """
        return self.net(obs)
