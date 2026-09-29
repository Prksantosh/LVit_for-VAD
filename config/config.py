from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple #, Optional,List, Sequence

@dataclass
class FALAPSConfig:
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    out_channels: int = 3
    embed_dim: int = 512

    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    reduction_ratio: int = 4
    qkv_bias: bool = True
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.0
    kernel: str = "exp"

    # Four feature-aligned skips from the 8-layer LViT.
    skip_indices: Tuple[int, ...] = (1, 3, 5, 7)

    # 8x8 -> 16 -> 32 -> 64 -> 128 -> 256
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    num_fasc: int = 4

    # Keep BatchNorm to remain close to the supplied model; switch to "group"
    # if very small batches make BatchNorm unstable.
    conv_norm: str = "batch"
