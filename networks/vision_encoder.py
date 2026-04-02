"""Faster R-CNN inspired vision encoder for RL observations.

Uses a ResNet-like CNN backbone with Feature Pyramid Network (FPN) style
multi-scale features, and RoI Align for per-agent feature extraction from
a top-down rendered environment image.

Based on: https://arxiv.org/abs/1506.01497 (Faster R-CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class FPNBackbone(nn.Module):
    """Feature Pyramid Network-style CNN backbone.

    Produces multi-scale feature maps from the input image.
    Simplified version inspired by ResNet+FPN from Faster R-CNN.
    """

    def __init__(self):
        super().__init__()
        # Bottom-up pathway (C2 → C5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Top-down pathway (FPN)
        self.lateral4 = nn.Conv2d(512, 256, 1)
        self.lateral3 = nn.Conv2d(256, 256, 1)
        self.lateral2 = nn.Conv2d(128, 256, 1)

        self.fpn_conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_conv2 = nn.Conv2d(256, 256, 3, padding=1)

    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 128, 128) image
        Returns:
            feature_map: (B, 256, H, W) — the P3 level feature map
        """
        c2 = self.conv1(x)      # (B, 64, 32, 32)
        c3 = self.layer2(c2)    # (B, 128, 16, 16)
        c4 = self.layer3(c3)    # (B, 256, 8, 8)
        c5 = self.layer4(c4)    # (B, 512, 4, 4)

        # FPN top-down
        p5 = self.lateral4(c5)  # (B, 256, 4, 4)
        p4 = self.lateral3(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p3 = self.fpn_conv3(p3)  # (B, 256, 16, 16)

        return p3


class VisionEncoder(nn.Module):
    """Faster R-CNN-style vision encoder for per-agent features.

    Pipeline:
        128×128 RGB → FPN backbone → feature map
        → RoI Align per agent region → FC → 128-dim per-agent feature

    Agent positions are mapped to RoI boxes on the feature map.
    """

    def __init__(self, output_dim=128, roi_size=7):
        super().__init__()
        self.backbone = FPNBackbone()
        self.roi_size = roi_size

        # Head after RoI Align
        self.head = nn.Sequential(
            nn.Linear(256 * roi_size * roi_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, image, agent_positions=None, agent_radius=0.1):
        """
        Args:
            image: (B, 3, 128, 128) top-down environment render
            agent_positions: (B, N, 2) normalized positions in [0,1] or None
            agent_radius: agent region radius in normalized coords
        Returns:
            features: (B*N, output_dim) per-agent features, or
                      (B, output_dim) global features if no positions given
        """
        feature_map = self.backbone(image)  # (B, 256, H, W)

        if agent_positions is None:
            # Global average pooling fallback
            pooled = F.adaptive_avg_pool2d(feature_map, self.roi_size)
            pooled = pooled.view(pooled.size(0), -1)
            return self.head(pooled)

        B, N, _ = agent_positions.shape
        H, W = feature_map.shape[2], feature_map.shape[3]

        # Convert agent positions to RoI boxes (x1, y1, x2, y2) in feature map coords
        boxes_list = []
        for b in range(B):
            for n in range(N):
                cx, cy = agent_positions[b, n]
                x1 = max(0, (cx - agent_radius) * W)
                y1 = max(0, (cy - agent_radius) * H)
                x2 = min(W, (cx + agent_radius) * W)
                y2 = min(H, (cy + agent_radius) * H)
                boxes_list.append([float(b), float(x1), float(y1), float(x2), float(y2)])

        rois = torch.tensor(boxes_list, dtype=torch.float32, device=image.device)

        # RoI Align
        pooled = roi_align(feature_map, rois, output_size=self.roi_size, spatial_scale=1.0)
        pooled = pooled.view(pooled.size(0), -1)  # (B*N, 256*roi_size*roi_size)

        return self.head(pooled)  # (B*N, output_dim)
