"""Bilinear fusion of LIDAR and vision features.

Computes the outer product interaction between LIDAR features and vision features,
then applies low-rank projection to create a compact fused representation.

This captures multiplicative relationships between modalities that simple
concatenation or addition would miss.
"""

import torch
import torch.nn as nn


class BilinearFusion(nn.Module):
    """Low-rank bilinear fusion of two feature vectors.

    Given LIDAR features f_l ∈ R^d_l and vision features f_v ∈ R^d_v:
    1. Project both to common dimension k
    2. Compute element-wise product (Hadamard product, efficient bilinear approx)
    3. Project to output dimension

    This is a low-rank approximation of full bilinear interaction:
        f_fused = W * (f_l ⊗ f_v) ≈ (U * f_l) ⊙ (V * f_v)
    """

    def __init__(self, lidar_dim=256, vision_dim=128, rank=128, output_dim=256):
        super().__init__()
        # Project to common rank space
        self.lidar_proj = nn.Sequential(
            nn.Linear(lidar_dim, rank),
            nn.ReLU(),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, rank),
            nn.ReLU(),
        )

        # Output projection with residual
        self.output = nn.Sequential(
            nn.Linear(rank, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Residual connections from original features
        self.lidar_residual = nn.Linear(lidar_dim, output_dim)
        self.vision_residual = nn.Linear(vision_dim, output_dim)

    def forward(self, lidar_features, vision_features):
        """
        Args:
            lidar_features: (B, lidar_dim) from ObservationEncoder
            vision_features: (B, vision_dim) from VisionEncoder
        Returns:
            fused: (B, output_dim) bilinear-fused features
        """
        # Low-rank bilinear interaction
        l_proj = self.lidar_proj(lidar_features)    # (B, rank)
        v_proj = self.vision_proj(vision_features)  # (B, rank)
        bilinear = l_proj * v_proj  # Hadamard product (B, rank)

        # Output with residual
        fused = self.output(bilinear)
        fused = fused + self.lidar_residual(lidar_features) + self.vision_residual(vision_features)

        return fused
