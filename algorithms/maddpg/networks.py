"""Actor and Critic networks for MADDPG."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MADDPGActor(nn.Module):
    """Decentralized actor: local obs → deterministic continuous action."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)


class MADDPGCritic(nn.Module):
    """Centralized critic: (all_obs, all_actions) → Q-value."""

    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)
