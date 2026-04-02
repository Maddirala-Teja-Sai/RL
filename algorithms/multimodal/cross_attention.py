"""Cross-attention fusion module for combining LIDAR and vision features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-attention between LIDAR and vision modalities.

    Uses multi-head attention where:
    - Query: one modality
    - Key/Value: the other modality
    Then fuses bidirectionally.
    """

    def __init__(self, lidar_dim=256, vision_dim=128, fused_dim=256, n_heads=4):
        super().__init__()
        self.lidar_proj = nn.Linear(lidar_dim, fused_dim)
        self.vision_proj = nn.Linear(vision_dim, fused_dim)

        # Cross-attention: LIDAR attends to vision
        self.cross_attn_l2v = nn.MultiheadAttention(fused_dim, n_heads, batch_first=True)
        # Cross-attention: vision attends to LIDAR
        self.cross_attn_v2l = nn.MultiheadAttention(fused_dim, n_heads, batch_first=True)

        self.layer_norm1 = nn.LayerNorm(fused_dim)
        self.layer_norm2 = nn.LayerNorm(fused_dim)

        self.output_net = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, lidar_features, vision_features):
        """
        Args:
            lidar_features: (B, lidar_dim)
            vision_features: (B, vision_dim)
        Returns:
            fused: (B, fused_dim)
        """
        l_proj = self.lidar_proj(lidar_features).unsqueeze(1)   # (B, 1, D)
        v_proj = self.vision_proj(vision_features).unsqueeze(1)  # (B, 1, D)

        # LIDAR attends to vision
        l2v, _ = self.cross_attn_l2v(l_proj, v_proj, v_proj)
        l2v = self.layer_norm1(l2v + l_proj).squeeze(1)

        # Vision attends to LIDAR
        v2l, _ = self.cross_attn_v2l(v_proj, l_proj, l_proj)
        v2l = self.layer_norm2(v2l + v_proj).squeeze(1)

        # Concatenate and fuse
        combined = torch.cat([l2v, v2l], dim=-1)
        return self.output_net(combined)
