"""raresight/models/classifier.py — ViT classifier for Stage 2 fine-tuning.

Loads the MAE-pretrained encoder and adds a classification head.
Supports layer-wise learning rate decay (LLRD) for stable fine-tuning.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from raresight.models.mae import MAEEncoder


class RareDiseaseClassifier(nn.Module):
    """ViT backbone (MAE-pretrained) + MLP classification head."""

    def __init__(
        self,
        num_classes: int = 8,
        dropout: float = 0.1,
        encoder_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        encoder_cfg = encoder_cfg or {
            "embed_dim": 768, "depth": 12, "num_heads": 12,
            "patch_size": 16, "img_size": 224,
        }
        self.encoder = MAEEncoder(
            img_size=encoder_cfg.get("img_size", 224),
            patch_size=encoder_cfg.get("patch_size", 16),
            embed_dim=encoder_cfg["embed_dim"],
            depth=encoder_cfg["depth"],
            num_heads=encoder_cfg["num_heads"],
        )
        embed_dim = encoder_cfg["embed_dim"]
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def load_mae_weights(self, checkpoint_path: str | Path) -> None:
        """Load encoder weights from a Stage 1 MAE checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)

        # Extract only encoder weights (strip "encoder." prefix if present)
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"[MAE load] missing={len(missing)}  unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode — use cls token representation (mask_ratio=0 at inference)
        enc_out, _, _ = self.encoder(x, mask_ratio=0.0)
        cls = enc_out[:, 0]                  # [B, embed_dim]
        return self.head(cls)

    def get_layer_groups(self, layer_decay: float = 0.75) -> list[dict]:
        """Parameter groups with layer-wise LR decay (LLRD)."""
        depth = len(self.encoder.blocks)
        param_groups = []

        for i, block in enumerate(self.encoder.blocks):
            lr_scale = layer_decay ** (depth - i)
            param_groups.append({
                "params": list(block.parameters()),
                "lr_scale": lr_scale,
                "name": f"block_{i}",
            })

        param_groups.append({"params": list(self.encoder.patch_embed.parameters()), "lr_scale": layer_decay ** (depth + 1), "name": "patch_embed"})
        param_groups.append({"params": [self.encoder.cls_token, self.encoder.pos_embed],  "lr_scale": layer_decay ** (depth + 1), "name": "embeddings"})
        param_groups.append({"params": list(self.head.parameters()), "lr_scale": 1.0, "name": "head"})

        return param_groups
