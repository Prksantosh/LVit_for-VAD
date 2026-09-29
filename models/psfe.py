from __future__ import annotations

import torch
import torch.nn as nn

def _make_norm_2d(channels: int, norm: str = "batch") -> nn.Module:
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
    
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm='{norm}'. Use 'batch' or 'group'.")

# -----------------------------------------------------------------------------
# 5. Progressive Pixel-Shuffle Feature Expansion decoder
# -----------------------------------------------------------------------------
class PixelShuffleFeatureExpansion(nn.Module):
    """Learned x2 feature expansion followed by spatial refinement."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch"):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(upscale_factor=2),
            _make_norm_2d(out_channels, norm),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _make_norm_2d(out_channels, norm),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.refine(x)
        return x
