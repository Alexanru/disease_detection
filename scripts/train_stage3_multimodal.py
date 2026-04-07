#!/usr/bin/env python3
"""scripts/train_stage3_multimodal.py - Stage 3: multimodal fusion training.

Loads MAE checkpoint from Stage 1, adds multimodal fusion (image + clinical).
Trains on HAM10000 with age, sex, localization features.

Usage:
    python scripts/train_stage3_multimodal.py
    python scripts/train_stage3_multimodal.py stage3.training.lr=1e-4 hardware.device=cpu
    python scripts/train_stage3_multimodal.py --resume              # Resume from best checkpoint
    python scripts/train_stage3_multimodal.py --resume-from outputs/multimodal_epoch010.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from raresight.data.multimodal_dataset import MultimodalDataset, build_multimodal_loaders
from raresight.evaluation.metrics import evaluate
from raresight.models.multimodal import MultimodalFusionModel
from raresight.training.losses import build_loss
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


@hydra.main(config_path="../configs", config_name="config_stage3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.project.seed)
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("  RareSight - Stage 3: multimodal fusion training")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = build_multimodal_loaders(
        data_root=cfg.data.root,
        dataset_name=cfg.data.name,
        batch_size=cfg.stage3.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        image_size=cfg.data.image_size,
        pin_memory=cfg.hardware.pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder_cfg = OmegaConf.to_container(cfg.stage1.encoder, resolve=True)
    encoder_cfg.update({"patch_size": 16, "img_size": cfg.data.image_size})

    model = MultimodalFusionModel(
        num_classes=cfg.stage3.model.num_classes,
        clinical_dim=cfg.data.clinical_dim,
        encoder_cfg=encoder_cfg,
        clinical_hidden=128,
        clinical_embed=256,
        fusion_hidden=512,
        dropout=cfg.stage3.model.dropout,
        modality_dropout_p=0.1,
    )

    # Load MAE weights from Stage 1
    ckpt = Path(cfg.project.checkpoint_dir) / f"{cfg.stage1.training.checkpoint_name}_best.pth"
    if Path(ckpt).exists():
        logger.info(f"Loading MAE weights from {ckpt}")
        model.load_mae_weights(str(ckpt))
    else:
        logger.warning(f"No checkpoint found at {ckpt}; training from scratch.")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Multimodal model parameters: {n_params:.1f}M")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = build_loss(
        OmegaConf.to_container(cfg.stage3.loss, resolve=True),
        class_counts=None,  # Will use default alpha
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.stage3.training.lr,
        weight_decay=cfg.stage3.training.weight_decay,
    )

    total_steps = cfg.stage3.training.epochs * len(loaders["train"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # ── W&B ───────────────────────────────────────────────────────────────────
    run = None
    try:
        run = wandb.init(
            project=cfg.stage3.wandb.project,
            tags=list(cfg.stage3.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if run:
            run.watch(model, log_freq=100)
    except Exception:
        logger.warning("W&B not available; skipping.")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=Path(cfg.project.checkpoint_dir),
        mixed_precision=cfg.hardware.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_clip=cfg.stage3.training.gradient_clip,
        wandb_run=run,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    epochs_no_improve = 0
    if RESUME_ARGS.resume_from:
        logger.info(f"Resuming from: {RESUME_ARGS.resume_from}")
        start_epoch = trainer.load_checkpoint(RESUME_ARGS.resume_from, load_optimizer=True)
    elif RESUME_ARGS.resume:
        best_checkpoint = Path(cfg.project.checkpoint_dir) / f"{cfg.stage3.training.checkpoint_name}_best.pth"
        if best_checkpoint.exists():
            logger.info("Resuming from best checkpoint")
            start_epoch = trainer.load_checkpoint(best_checkpoint, load_optimizer=True)
        else:
            logger.warning("No best checkpoint found, starting from epoch 1")

    patience = cfg.stage3.training.early_stopping_patience

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.stage3.training.epochs + 1):
        train_loss = trainer.train_epoch(loaders["train"], epoch)
        val_metrics = trainer.eval_epoch(loaders["val"])

        trainer.step_scheduler(epoch, val_loss=val_metrics["val/loss"])

        logger.info(
            f"E{epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val/loss']:.4f} | "
            f"val_acc={val_metrics['val/acc']:.4f}"
        )

        if run:
            run.log({"train/loss": train_loss, **val_metrics, "epoch": epoch})

        improved = trainer.save_best(epoch, val_metrics, cfg.stage3.training.checkpoint_name)
        if not improved:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement.")
                break
        else:
            epochs_no_improve = 0

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("Running final test evaluation")
    best_ckpt = Path(cfg.project.checkpoint_dir) / f"{cfg.stage3.training.checkpoint_name}_best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded best checkpoint from {best_ckpt}")

        test_metrics = trainer.eval_epoch(loaders["test"])
        logger.success(f"Test results: {test_metrics}")
        if run:
            run.log({"test/loss": test_metrics["val/loss"], "test/acc": test_metrics["val/acc"]})

    logger.success("Stage 3 multimodal training complete.")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
