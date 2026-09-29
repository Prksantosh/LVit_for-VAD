"""
Feature-Aligned Linear-Attention Pixel-Shuffle Autoencoder (updated implementation)

Core pipeline preserved:
    Input frame
      -> Progressive Patch Embedding
      -> Corrected Linear Transformer Block x L
      -> LViT encoder features
      -> Feature-Aligned Skip Connections (FASC)
      -> Progressive Pixel-Shuffle Feature Expansion (PSFE) Decoder
      -> Reconstructed frame

The linear-attention implementation never materializes an N x N attention matrix.
It computes K^T V first and then Q(K^T V), with positive kernel feature maps
and the corresponding normalization denominator.
"""

from __future__ import annotations

#from dataclasses import dataclass
#from typing import List, Sequence, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _check_square_tokens(n_tokens: int) -> int:
    side = int(math.isqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(
            f"Token count must form a square 2-D grid for FASC/decoder; got N={n_tokens}."
        )
    return side


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


class DropPath(nn.Module):
    """Stochastic depth. Identity when drop_prob=0 or during evaluation."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -----------------------------------------------------------------------------
# 1. Progressive Patch Embedding
# -----------------------------------------------------------------------------
class ProgressivePatchEmbedding(nn.Module):
    """
    Three-stage convolutional progressive patch embedding.

    Default 256x256 configuration:
        256 -> 128  (7x7, stride 2)
        128 ->  64  (3x3, stride 2)
         64 ->   8  (8x8, stride 8 when patch_size=16)

    Output:
        tokens:   [B, N, D]
        token_hw: (H_t, W_t)

    No CLS token is used because the encoder feeds a reconstruction decoder.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
        norm: str = "batch",
    ):
        super().__init__()
        if patch_size < 4 or patch_size % 2 != 0:
            raise ValueError("patch_size must be an even integer >= 4.")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        mid_dim = embed_dim // 2
        final_kernel = patch_size // 2

        self.local_context = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_dim,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            _make_norm_2d(mid_dim, norm),
            nn.GELU(),
        )

        self.regional_context = nn.Sequential(
            nn.Conv2d(
                mid_dim,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            _make_norm_2d(embed_dim, norm),
            nn.GELU(),
        )

        self.global_projection = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=final_kernel,
                stride=final_kernel,
                padding=0,
                bias=False,
            ),
            _make_norm_2d(embed_dim, norm),
        )

        # LayerNorm is applied after 2-D features are rearranged as tokens.
        self.token_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.local_context(x)
        x = self.regional_context(x)
        x = self.global_projection(x)

        h_t, w_t = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]
        tokens = self.token_norm(tokens)
        return tokens, (h_t, w_t)


# -----------------------------------------------------------------------------
# 2. Corrected kernel linear attention
# -----------------------------------------------------------------------------
class KernelLinearAttention(nn.Module):
    r"""
    Reduced-dimensional multi-head kernel linear attention.

    This adapts the original LViT's reduced Q/K/V projection while correcting
    the quadratic attention computation.

    For positive feature map phi(.):

        Y = phi(Q) [phi(K)^T V]
            ---------------------------------
            phi(Q) [phi(K)^T 1] + eps

    K^T V is evaluated BEFORE multiplying by Q, so no [N x N] attention matrix
    is formed. For fixed head dimension d, sequence complexity is linear in N.

    Default feature map is an element-wise exponential map to stay consistent
    with the manuscript's exponential-kernel formulation. Values are clamped
    before exponentiation for numerical stability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        reduction_ratio: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        kernel: str = "exp",
        exp_clamp: float = 8.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        if dim % reduction_ratio != 0:
            raise ValueError(
                f"dim={dim} must be divisible by reduction_ratio={reduction_ratio}."
            )

        reduced_dim = dim // reduction_ratio
        if reduced_dim % num_heads != 0:
            raise ValueError(
                f"Reduced dim={reduced_dim} must be divisible by num_heads={num_heads}."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio
        self.reduced_dim = reduced_dim
        self.head_dim = reduced_dim // num_heads
        self.kernel = kernel.lower()
        self.exp_clamp = float(exp_clamp)
        self.eps = float(eps)

        # Splitting the scale symmetrically between Q and K is convenient for
        # a dot-product-inspired kernel while retaining associative computation.
        self.qk_scale = self.head_dim ** -0.25

        # Preserve the original LViT idea: Q is projected separately and K/V
        # share one projection layer, all in reduced latent dimension.
        self.q = nn.Linear(dim, reduced_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * reduced_dim, bias=qkv_bias)

        # There is no explicit N x N attention probability matrix on which to
        # apply dropout. Dropout is therefore applied to the linear-attention
        # output before the final projection.
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(reduced_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "exp":
            # Positive deterministic exponential feature map.
            x = torch.clamp(x, min=-self.exp_clamp, max=self.exp_clamp)
            return torch.exp(x)
        if self.kernel == "elu":
            # Optional stable positive map for experimentation/ablation.
            return F.elu(x) + 1.0
        raise ValueError(f"Unsupported kernel='{self.kernel}'. Use 'exp' or 'elu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape

        q = self.q(x)
        kv = self.kv(x)
        k, v = kv.chunk(2, dim=-1)

        # [B, H, N, d]
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q = self._feature_map(q * self.qk_scale)
        k = self._feature_map(k * self.qk_scale)

        # Associative linear attention:
        #   KV = phi(K)^T V      -> [B,H,d,d]
        #   normalizer uses sum_N phi(K), never QK^T.
        kv_context = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)  # [B,H,d]

        denominator = torch.einsum("bhnd,bhd->bhn", q, k_sum)
        denominator = denominator.clamp_min(self.eps)

        out = torch.einsum("bhnd,bhde->bhne", q, kv_context)
        out = out / denominator.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(b, n, self.reduced_dim)
        out = self.attn_drop(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearTransformerBlock(nn.Module):
    """Pre-norm residual transformer block with corrected kernel linear attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        reduction_ratio: int = 4,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        kernel: str = "exp",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = KernelLinearAttention(
            dim=dim,
            num_heads=num_heads,
            reduction_ratio=reduction_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            kernel=kernel,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x



