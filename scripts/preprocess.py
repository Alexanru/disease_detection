"""scripts/preprocess.py — Dataset preprocessing stub.

This script prepares raw datasets for training. It handles:
- Download verification
- Image resizing and normalization
- Train/val/test split
- Clinical data alignment

For now, this is a stub. Full implementation requires manual dataset downloads.
See README.md for dataset acquisition instructions.
"""

import os
import sys
from pathlib import Path

import torch
from loguru import logger

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
CHECKPOINTS_DIR = Path("checkpoints")


def setup_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "raw").mkdir(exist_ok=True)
    (DATA_DIR / "processed").mkdir(exist_ok=True)
    logger.info(f"✅ Directories ready: {DATA_DIR}, {CHECKPOINTS_DIR}")


def check_datasets():
    """Verify that datasets are downloaded."""
    datasets = {
        "ISIC 2019": DATA_DIR / "raw" / "ISIC_2019_Training_Input",
        "HAM10000": DATA_DIR / "raw" / "ham10000",
        "PAD-UFES-20": DATA_DIR / "raw" / "pad_ufes_20",
    }

    logger.info("📋 Dataset check:")
    for name, path in datasets.items():
        if path.exists():
            logger.success(f"  ✅ {name}: found at {path}")
        else:
            logger.warning(f"  ⚠️  {name}: not found at {path}")

    return all(p.exists() for p in datasets.values())


def check_pytorch():
    """Verify PyTorch and GPU availability."""
    logger.info(f"🔧 PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    logger.info(f"🖥️  CUDA available: {cuda_available}")
    if cuda_available:
        logger.success(f"   GPU: {torch.cuda.get_device_name(0)}")


def main():
    """Run preprocessing checks."""
    logger.info("🚀 RareSight Preprocessing")
    logger.info("=" * 60)

    setup_directories()
    logger.info("")

    check_pytorch()
    logger.info("")

    all_ready = check_datasets()
    logger.info("")

    if all_ready:
        logger.success("✅ All datasets ready for training!")
    else:
        logger.warning("""
⚠️  Some datasets are missing. To proceed:

1. **ISIC 2019**: https://isic-archive.com
   - Download training input → extract to data/raw/ISIC_2019_Training_Input

2. **HAM10000**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Extract to data/raw/ham10000

3. **PAD-UFES-20**: https://data.mendeley.com/datasets/znsxvx2xyd
   - Extract to data/raw/pad_ufes_20

Or run scripts/download_data.py for automated download (if available).
        """)


if __name__ == "__main__":
    main()
