from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm_2d(channels: int, norm: str = "batch") -> nn.Module:
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        # Choose a valid number of groups while keeping groups reasonably small.
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm='{norm}'. Use 'batch' or 'group'.")

# -----------------------------------------------------------------------------
# 4. Feature-Aligned Skip Connection (FASC)
# -----------------------------------------------------------------------------
class FeatureAlignedSkipConnection(nn.Module):


    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm: str = "batch",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.channel_projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # Local feature refinement with lightweight depthwise spatial processing.
        self.local_refine = nn.Sequential(
            nn.Conv2d(
                out_dim,
                out_dim,
                kernel_size=3,
                padding=1,
                groups=out_dim,
                bias=False,
            ),
            _make_norm_2d(out_dim, norm),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            _make_norm_2d(out_dim, norm),
        )

        # Spatial attention/refinement gate generated from average + max maps.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.out_act = nn.GELU()

    def forward(
        self,
        tokens: torch.Tensor,
        token_hw: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        b, n, _ = tokens.shape
        h_t, w_t = token_hw
        if h_t * w_t != n:
            raise ValueError(
                f"token_hw={token_hw} is incompatible with N={n} tokens."
            )

        x = self.channel_projection(tokens)
        x = x.transpose(1, 2).reshape(b, self.out_dim, h_t, w_t)

        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        refined = self.local_refine(x)
        avg_map = refined.mean(dim=1, keepdim=True)
        max_map = refined.amax(dim=1, keepdim=True)
        gate = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        # Preserve aligned features and add locally refined details.
        return self.out_act(x + gate * refined)
