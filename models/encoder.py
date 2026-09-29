# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:34:41 2026

@author: USER
"""

from __future__ import annotations

#from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
#import math

import torch
import torch.nn as nn
#import torch.nn.functional as F
from models.LVit import LinearTransformerBlock

# -----------------------------------------------------------------------------
# 3. LViT encoder for reconstruction (no classifier / no CLS token)
# -----------------------------------------------------------------------------
class LViTEncoder(nn.Module):
    """
    Stack of corrected linear transformer blocks.

    Returns:
        final_tokens: normalized output of the last block
        skip_tokens:  selected intermediate block outputs for FASC

    All skip features preserve the token grid; FASC performs stage-specific
    channel projection and spatial alignment for the decoder.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        reduction_ratio: int = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.0,
        kernel: str = "exp",
        skip_indices: Optional[Sequence[int]] = (1, 3, 5, 7),
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1.")

        if skip_indices is None:
            # Up to four evenly distributed taps.
            n_taps = min(4, depth)
            skip_indices = tuple(
                sorted(set(round(i) for i in torch.linspace(0, depth - 1, n_taps).tolist()))
            )
        else:
            skip_indices = tuple(int(i) for i in skip_indices)

        if not skip_indices:
            raise ValueError("At least one skip index is required for FASC.")
        if min(skip_indices) < 0 or max(skip_indices) >= depth:
            raise ValueError(
                f"skip_indices={skip_indices} are invalid for depth={depth}."
            )

        self.embed_dim = embed_dim
        self.depth = depth
        self.skip_indices = skip_indices

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                LinearTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    reduction_ratio=reduction_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    kernel=kernel,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_tokens: List[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.skip_indices:
                skip_tokens.append(x)

        final_tokens = self.norm(x)

        # If the last selected skip is from the final block, use the normalized
        # representation there as well so bottleneck and deepest skip are coherent.
        if self.skip_indices[-1] == self.depth - 1:
            skip_tokens[-1] = final_tokens

        return final_tokens, skip_tokens