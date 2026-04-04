"""tests/test_models.py — Unit tests for RareSight models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import numpy as np


# ── MAE tests ─────────────────────────────────────────────────────────────────

class TestMAE:
    @pytest.fixture
    def mae(self):
        from raresight.models.mae import MaskedAutoencoder
        cfg = {
            "img_size": 64, "patch_size": 16, "mask_ratio": 0.75,
            "encoder": {"embed_dim": 192, "depth": 2, "num_heads": 3},
            "decoder": {"embed_dim": 128, "depth": 2, "num_heads": 4},
        }
        return MaskedAutoencoder(cfg)

    def test_forward_shape(self, mae):
        x = torch.randn(2, 3, 64, 64)
        out = mae(x)
        assert "loss" in out
        assert "pred" in out
        assert "mask" in out

    def test_loss_is_scalar(self, mae):
        x = torch.randn(2, 3, 64, 64)
        out = mae(x)
        assert out["loss"].shape == torch.Size([])

    def test_loss_is_positive(self, mae):
        x = torch.randn(2, 3, 64, 64)
        out = mae(x)
        assert out["loss"].item() > 0

    def test_mask_ratio(self, mae):
        x = torch.randn(2, 3, 64, 64)
        out = mae(x)
        # 16 patches for 64×64 with patch_size=16; 75% should be masked
        expected_masked = int(16 * 0.75)
        actual_masked = out["mask"][0].sum().item()
        assert abs(actual_masked - expected_masked) <= 1  # allow ±1 due to floor

    def test_patchify_shape(self, mae):
        x = torch.randn(2, 3, 64, 64)
        patches = mae.patchify(x)
        # 4×4 = 16 patches, each patch 16×16×3 = 768
        assert patches.shape == (2, 16, 768)


# ── Classifier tests ───────────────────────────────────────────────────────────

class TestClassifier:
    @pytest.fixture
    def model(self):
        from raresight.models.classifier import RareDiseaseClassifier
        cfg = {"embed_dim": 192, "depth": 2, "num_heads": 3, "img_size": 64, "patch_size": 16}
        return RareDiseaseClassifier(num_classes=8, encoder_cfg=cfg)

    def test_forward_shape(self, model):
        x = torch.randn(4, 3, 64, 64)
        logits = model(x)
        assert logits.shape == (4, 8)

    def test_no_nan(self, model):
        x = torch.randn(4, 3, 64, 64)
        logits = model(x)
        assert not torch.isnan(logits).any()


# ── Multimodal tests ──────────────────────────────────────────────────────────

class TestMultimodal:
    @pytest.fixture
    def model(self):
        from raresight.models.multimodal import MultimodalFusionModel
        enc_cfg = {"embed_dim": 192, "depth": 2, "num_heads": 3, "img_size": 64, "patch_size": 16}
        return MultimodalFusionModel(num_classes=7, clinical_dim=13, encoder_cfg=enc_cfg)

    def test_forward_full(self, model):
        img = torch.randn(2, 3, 64, 64)
        clin = torch.randn(2, 13)
        out = model(img, clin, mode="full")
        assert out.shape == (2, 7)

    def test_forward_image_only(self, model):
        img = torch.randn(2, 3, 64, 64)
        clin = torch.zeros(2, 13)
        out = model(img, clin, mode="image")
        assert out.shape == (2, 7)

    def test_forward_clinical_only(self, model):
        img = torch.zeros(2, 3, 64, 64)
        clin = torch.randn(2, 13)
        out = model(img, clin, mode="clinical")
        assert out.shape == (2, 7)


# ── Loss tests ────────────────────────────────────────────────────────────────

class TestFocalLoss:
    def test_focal_basic(self):
        from raresight.training.losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0
        assert loss.shape == torch.Size([])

    def test_focal_with_alpha(self):
        from raresight.training.losses import FocalLoss
        alpha = torch.tensor([0.1, 0.3, 0.4, 0.2])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)

    def test_build_loss_factory(self):
        from raresight.training.losses import build_loss
        cfg = {"type": "focal", "gamma": 2.0, "alpha": "auto", "label_smoothing": 0.1}
        loss_fn = build_loss(cfg, class_counts=[100, 50, 200, 30])
        assert loss_fn is not None


# ── Augmentation tests ────────────────────────────────────────────────────────

class TestAugmentation:
    def test_train_transform(self):
        from raresight.data.dataset import build_train_transform
        import numpy as np
        tf = build_train_transform(224)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out = tf(image=img)["image"]
        assert out.shape == (3, 224, 224)

    def test_val_transform(self):
        from raresight.data.dataset import build_val_transform
        import numpy as np
        tf = build_val_transform(224)
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        out = tf(image=img)["image"]
        assert out.shape == (3, 224, 224)
