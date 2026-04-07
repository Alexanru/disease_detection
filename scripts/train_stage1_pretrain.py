#!/usr/bin/env python3
"""scripts/train_stage1_pretrain.py - Stage 1: MAE self-supervised pretraining.

Usage:
    python scripts/train_stage1_pretrain.py
    python scripts/train_stage1_pretrain.py stage1.epochs=100 hardware.device=cpu
    python scripts/train_stage1_pretrain.py --resume              # Resume from best checkpoint
    python scripts/train_stage1_pretrain.py --resume-from outputs/mae_epoch050.pth
"""
import os
os.environ["TMPDIR"] = "/tmp"
os.environ["WANDB_DISABLED"] = "true"

import torch
torch.multiprocessing.set_sharing_strategy("file_system")


import argparse
import sys

import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import torch
torch.multiprocessing.set_sharing_strategy("file_system")

import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from raresight.data.dataset import UnlabeledImageDataset, build_mae_transform
from raresight.models.mae import MaskedAutoencoder
from raresight.training.trainer import Trainer


def _extract_resume_args(argv: list[str]) -> argparse.Namespace:
    """Strip custom resume args before Hydra parses CLI options."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint")
    args, remaining = parser.parse_known_args(argv[1:])
    sys.argv = [argv[0], *remaining]
    return args


RESUME_ARGS = _extract_resume_args(sys.argv.copy())


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.project.seed)
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("  RareSight - Stage 1: MAE pretraining")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))

    # ── Dataset ────────────────────────────────────────────────────────────────
    transform = build_mae_transform(cfg.data.image_size)
    dataset = UnlabeledImageDataset(
        Path(cfg.data.root),
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.stage1.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
    )
    logger.info(f"Dataset: {len(dataset):,} images | batches: {len(loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = MaskedAutoencoder(OmegaConf.to_container(cfg.stage1, resolve=True))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"MAE parameters: {n_params:.1f}M")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.stage1.training.lr,
        weight_decay=cfg.stage1.training.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = cfg.stage1.training.epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── W&B ───────────────────────────────────────────────────────────────────
    run = None
    try:
        run = wandb.init(
            project=cfg.stage1.wandb.project,
            tags=list(cfg.stage1.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    except Exception:
        logger.warning("W&B not available — skipping.")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=None,                            # MAE computes its own loss
        device=device,
        checkpoint_dir=Path(cfg.project.checkpoint_dir),
        mixed_precision=cfg.hardware.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        wandb_run=run,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    if RESUME_ARGS.resume_from:
        logger.info(f"Resuming from: {RESUME_ARGS.resume_from}")
        start_epoch = trainer.load_checkpoint(RESUME_ARGS.resume_from, load_optimizer=True)
    elif RESUME_ARGS.resume:
        best_checkpoint = Path(cfg.project.checkpoint_dir) / f"{cfg.stage1.training.checkpoint_name}_best.pth"
        if best_checkpoint.exists():
            logger.info("Resuming from best checkpoint")
            start_epoch = trainer.load_checkpoint(best_checkpoint, load_optimizer=True)
        else:
            logger.warning("No best checkpoint found, starting from epoch 1")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.stage1.training.epochs + 1):
        train_loss = trainer.train_epoch(loader, epoch)
        trainer.step_scheduler(epoch)

        logger.info(f"Epoch {epoch:3d}/{cfg.stage1.training.epochs} | loss={train_loss:.4f}")

        if run:
            run.log({"train/epoch_loss": train_loss, "epoch": epoch})

        # Save checkpoint every N epochs
        if epoch % cfg.stage1.training.save_every == 0:
            trainer.save_checkpoint(epoch, {"loss": train_loss}, cfg.stage1.training.checkpoint_name)

        # Always track best
        trainer.save_best(epoch, {"val/loss": train_loss}, cfg.stage1.training.checkpoint_name)

    logger.success("Stage 1 pretraining complete.")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
