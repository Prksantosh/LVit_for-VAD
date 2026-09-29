# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:36:30 2026

@author: USER
"""

from __future__ import annotations

#from dataclasses import dataclass
#from typing import List, Sequence, Tuple, Optional
#import math

import torch
import torch.nn as nn
#import torch.nn.functional as F
from models.psfe import PixelShuffleFeatureExpansion
from models.FASC import FeatureAlignedSkipConnection

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


class ProgressivePixelShuffleDecoder(nn.Module):
    """
    Five x2 PSFE stages for the default 8x8 -> 256x256 reconstruction path.

    Four FASC connections are fused at decoder stages 1..4. The final stage is
    intentionally skip-free so the output must be synthesized from the decoder
    representation rather than receiving a direct high-resolution bypass.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        out_channels: int = 3,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        num_fasc: int = 4,
        norm: str = "batch",
    ):
        super().__init__()

        decoder_channels = tuple(int(c) for c in decoder_channels)
        if len(decoder_channels) < 1:
            raise ValueError("decoder_channels cannot be empty.")
        if num_fasc < 0 or num_fasc > len(decoder_channels):
            raise ValueError("num_fasc must be between 0 and number of decoder stages.")

        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels
        self.num_fasc = num_fasc

        in_channels = [embed_dim] + list(decoder_channels[:-1])
        self.psfe_blocks = nn.ModuleList(
            [
                PixelShuffleFeatureExpansion(cin, cout, norm=norm)
                for cin, cout in zip(in_channels, decoder_channels)
            ]
        )

        self.fasc = nn.ModuleList(
            [
                FeatureAlignedSkipConnection(embed_dim, decoder_channels[i], norm=norm)
                for i in range(num_fasc)
            ]
        )

        # Refinement after U_i + S_i^aligned fusion.
        self.fusion_refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        decoder_channels[i],
                        decoder_channels[i],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    _make_norm_2d(decoder_channels[i], norm),
                    nn.GELU(),
                )
                for i in range(num_fasc)
            ]
        )

        final_ch = decoder_channels[-1]
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(final_ch, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        bottleneck_tokens: torch.Tensor,
        skip_tokens: Sequence[torch.Tensor],
        token_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, n, c = bottleneck_tokens.shape
        h_t, w_t = token_hw

        if h_t * w_t != n:
            raise ValueError(
                f"token_hw={token_hw} is incompatible with bottleneck N={n}."
            )
        if c != self.embed_dim:
            raise ValueError(
                f"Expected bottleneck dim={self.embed_dim}, got C={c}."
            )
        if len(skip_tokens) < self.num_fasc:
            raise ValueError(
                f"Decoder requires {self.num_fasc} FASC skip features, "
                f"but encoder returned {len(skip_tokens)}."
            )

        x = bottleneck_tokens.transpose(1, 2).reshape(b, c, h_t, w_t)

        # Deepest encoder feature is used first; progressively earlier features
        # are injected as spatial resolution grows.
        skips = list(reversed(skip_tokens[-self.num_fasc :])) if self.num_fasc else []

        for i, psfe in enumerate(self.psfe_blocks):
            x = psfe(x)

            if i < self.num_fasc:
                aligned = self.fasc[i](
                    skips[i],
                    token_hw=token_hw,
                    target_size=x.shape[-2:],
                )
                x = self.fusion_refine[i](x + aligned)

        return self.reconstruction_head(x)
