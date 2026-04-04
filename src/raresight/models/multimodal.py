"""raresight/models/multimodal.py — Late-fusion multimodal classifier.

Architecture:
  ┌─────────────────┐    ┌────────────────────┐
  │  Image (224×224) │    │ Clinical features   │
  │  ViT-B/16 encoder│    │ (age, sex, loc, ...) │
  │  [MAE-pretrained]│    │  MLP encoder        │
  └────────┬─────────┘    └─────────┬──────────┘
           │   cls token [768]       │  [256]
           └──────────┬─────────────┘
                      │  concat → [1024]
                 Fusion MLP
                      │
              Classifier head [num_classes]

This late fusion approach:
  - Allows ablation: image-only, clinical-only, multimodal
  - Each modality can be trained/frozen independently
  - Modality dropout for robustness (clinical data sometimes missing)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from raresight.models.classifier import RareDiseaseClassifier


class ClinicalMLP(nn.Module):
    """Small MLP to encode tabular clinical features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFusionModel(nn.Module):
    """Late-fusion: ViT image encoder + Clinical MLP → shared head.

    Supports three inference modes:
      - "full":     image + clinical (default)
      - "image":    image only (clinical zeroed — ablation)
      - "clinical": clinical only (image zeroed — ablation)
    """

    def __init__(
        self,
        num_classes: int,
        clinical_dim: int,
        encoder_cfg: dict | None = None,
        clinical_hidden: int = 128,
        clinical_embed: int = 256,
        fusion_hidden: int = 512,
        dropout: float = 0.2,
        modality_dropout_p: float = 0.1,    # randomly drop one modality during training
    ) -> None:
        super().__init__()

        # ── Image branch (ViT encoder only, no head) ─────────────────────────
        encoder_cfg = encoder_cfg or {"embed_dim": 768, "depth": 12, "num_heads": 12}
        _cls = RareDiseaseClassifier(num_classes=1, encoder_cfg=encoder_cfg)
        self.image_encoder = _cls.encoder
        image_embed_dim = encoder_cfg["embed_dim"]

        # ── Clinical branch ───────────────────────────────────────────────────
        self.clinical_encoder = ClinicalMLP(clinical_dim, clinical_hidden, clinical_embed, dropout)

        # ── Fusion ────────────────────────────────────────────────────────────
        fusion_input = image_embed_dim + clinical_embed
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_classes),
        )

        self.modality_dropout_p = modality_dropout_p
        self.image_embed_dim = image_embed_dim
        self.clinical_embed = clinical_embed

    def load_mae_weights(self, checkpoint_path: str) -> None:
        """Load MAE pretrained encoder into image branch."""
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        encoder_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = self.image_encoder.load_state_dict(encoder_state, strict=False)
        print(f"[MAE load] missing={len(missing)}  unexpected={len(unexpected)}")

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        enc_out, _, _ = self.image_encoder(images, mask_ratio=0.0)
        return enc_out[:, 0]   # cls token [B, image_embed_dim]

    def forward(
        self,
        images: torch.Tensor,
        clinical: torch.Tensor,
        mode: str = "full",
    ) -> torch.Tensor:
        B = images.shape[0]

        img_emb = self.encode_image(images)           # [B, 768]
        clin_emb = self.clinical_encoder(clinical)    # [B, 256]

        # Modality dropout during training (robustness to missing data)
        if self.training and self.modality_dropout_p > 0:
            drop_img  = torch.rand(B, 1, device=images.device) < self.modality_dropout_p
            drop_clin = torch.rand(B, 1, device=images.device) < self.modality_dropout_p
            img_emb  = img_emb  * (~drop_img).float()
            clin_emb = clin_emb * (~drop_clin).float()

        # Ablation modes
        if mode == "image":
            clin_emb = torch.zeros_like(clin_emb)
        elif mode == "clinical":
            img_emb = torch.zeros_like(img_emb)

        fused = torch.cat([img_emb, clin_emb], dim=1)   # [B, 1024]
        return self.fusion(fused)

    def get_optimizer_groups(self, base_lr: float, layer_decay: float = 0.75, wd: float = 0.01) -> list[dict]:
        """Separate LR groups: frozen backbone vs. new heads."""
        backbone_params = list(self.image_encoder.parameters())
        new_params = list(self.clinical_encoder.parameters()) + list(self.fusion.parameters())
        return [
            {"params": backbone_params, "lr": base_lr * layer_decay, "weight_decay": wd},
            {"params": new_params,       "lr": base_lr,               "weight_decay": wd},
        ]
