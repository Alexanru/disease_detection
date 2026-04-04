"""raresight/models/mae.py — Masked Autoencoder (He et al., 2022).

Implements MAE pretraining:
  - ViT encoder that sees only unmasked patches
  - Lightweight decoder that reconstructs masked patches
  - Pixel-level MSE reconstruction loss

Reference: https://arxiv.org/abs/2111.06377
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block


class PatchEmbed(nn.Module):
    """Image → sequence of patch embeddings."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), "b c h w -> b (h w) c")


class MAEEncoder(nn.Module):
    """ViT encoder — processes only visible (unmasked) patches."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Sinusoidal position embeddings (fixed)
        pos_embed = self._sinusoidal_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5))
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

    @staticmethod
    def _sinusoidal_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
        grid = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid, grid, indexing="xy")
        grid = torch.stack(grid).reshape(2, -1)
        emb_h = _1d_sincos(embed_dim // 2, grid[0])
        emb_w = _1d_sincos(embed_dim // 2, grid[1])
        pos = torch.cat([emb_h, emb_w], dim=1)          # (num_patches, embed_dim)
        cls = torch.zeros(1, embed_dim)
        return torch.cat([cls, pos], dim=0)              # (1 + num_patches, embed_dim)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask patches; return visible patches + mask + restore indices."""
        B, N, D = x.shape
        keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)       # 1 = masked, 0 = visible

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]              # add position (no cls)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore


class MAEDecoder(nn.Module):
    """Lightweight Transformer decoder that reconstructs masked patches."""

    def __init__(
        self,
        num_patches: int,
        encoder_dim: int = 768,
        decoder_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size ** 2 * 3, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        B, N_vis_plus1, D = x.shape

        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - N_vis_plus1 + 1, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)               # drop cls
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)                          # re-add cls

        x = x + self.decoder_pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.pred(x[:, 1:, :])                                        # remove cls
        return x


class MaskedAutoencoder(nn.Module):
    """Full MAE: encoder + decoder + reconstruction loss."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.encoder = MAEEncoder(
            img_size=cfg.get("img_size", 224),
            patch_size=cfg.get("patch_size", 16),
            embed_dim=cfg["encoder"]["embed_dim"],
            depth=cfg["encoder"]["depth"],
            num_heads=cfg["encoder"]["num_heads"],
        )
        self.decoder = MAEDecoder(
            num_patches=self.encoder.patch_embed.num_patches,
            encoder_dim=cfg["encoder"]["embed_dim"],
            decoder_dim=cfg["decoder"]["embed_dim"],
            depth=cfg["decoder"]["depth"],
            num_heads=cfg["decoder"]["num_heads"],
            patch_size=cfg.get("patch_size", 16),
        )
        self.mask_ratio = cfg.get("mask_ratio", 0.75)
        self.patch_size = cfg.get("patch_size", 16)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        return rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, mask, ids_restore = self.encoder(imgs, self.mask_ratio)
        pred = self.decoder(latent, ids_restore)
        target = self.patchify(imgs)

        # MSE loss on masked patches only
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return {"loss": loss, "pred": pred, "mask": mask}

    def get_encoder(self) -> MAEEncoder:
        """Return encoder for Stage 2 fine-tuning."""
        return self.encoder


# ── Utility ───────────────────────────────────────────────────────────────────

def _1d_sincos(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2)
    omega = 1.0 / (10000 ** omega)
    out = torch.einsum("n,d->nd", pos.float(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
