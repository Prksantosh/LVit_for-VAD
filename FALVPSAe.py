from __future__ import annotations

from typing import List, Sequence, Tuple, Optional

import torch
import torch.nn as nn

from config.config import FALAPSConfig
from models.LVit import ProgressivePatchEmbedding
from models.encoder import LViTEncoder
from models.decoder import ProgressivePixelShuffleDecoder


class FeatureAlignedLViTPixelShuffleAutoencoder(nn.Module):
    """Complete reconstruction model matching the requested paper pipeline."""

    def __init__(self, config: FALAPSConfig = FALAPSConfig()):
        super().__init__()
        self.config = config

        self.patch_embed = ProgressivePatchEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            norm=config.conv_norm,
        )

        self.encoder = LViTEncoder(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            reduction_ratio=config.reduction_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            kernel=config.kernel,
            skip_indices=config.skip_indices,
        )

        self.decoder = ProgressivePixelShuffleDecoder(
            embed_dim=config.embed_dim,
            out_channels=config.out_channels,
            decoder_channels=config.decoder_channels,
            num_fasc=config.num_fasc,
            norm=config.conv_norm,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        input_hw = x.shape[-2:]

        tokens, token_hw = self.patch_embed(x)
        bottleneck, skip_tokens = self.encoder(tokens)
        recon = self.decoder(bottleneck, skip_tokens, token_hw)

        # For the stated 256x256 / patch_size=16 configuration, reconstruction
        # is produced exactly by PSFE. Raise instead of silently using bilinear
        # interpolation, preserving the progressive pixel-shuffle claim.
        if recon.shape[-2:] != input_hw:
            raise RuntimeError(
                "Decoder output resolution does not match input. "
                f"Input={input_hw}, output={recon.shape[-2:]}, token_hw={token_hw}. "
                "Tune decoder stage count/channels or patch embedding stride; "
                "do not silently interpolate if a pure PSFE decoder is intended."
            )

        if return_features:
            return {
                "reconstruction": recon,
                "tokens": tokens,
                "token_hw": token_hw,
                "bottleneck": bottleneck,
                "skip_tokens": skip_tokens,
            }
        return recon


# -----------------------------------------------------------------------------
# 7. Convenience functions / smoke test
# -----------------------------------------------------------------------------
def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_falaps_model(**overrides) -> FeatureAlignedLViTPixelShuffleAutoencoder:
    """Build the default model while allowing selected config values to be overridden."""
    cfg = FALAPSConfig(**overrides)
    return FeatureAlignedLViTPixelShuffleAutoencoder(cfg)


def smoke_test(device: Optional[str] = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_falaps_model().to(device)
    model.eval()

    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        result = model(x, return_features=True)

    y = result["reconstruction"]
    print("Model: FeatureAlignedLViTPixelShuffleAutoencoder")
    print(f"Device: {device}")
    print(f"Input shape:          {tuple(x.shape)}")
    print(f"Token grid:           {result['token_hw']}")
    print(f"Token shape:          {tuple(result['tokens'].shape)}")
    print(f"Bottleneck shape:     {tuple(result['bottleneck'].shape)}")
    print("Skip shapes:          ", [tuple(s.shape) for s in result["skip_tokens"]])
    print(f"Reconstruction shape: {tuple(y.shape)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Output range:         [{y.min().item():.4f}, {y.max().item():.4f}]")


if __name__ == "__main__":
    smoke_test()
