#!/usr/bin/env python3
"""scripts/train_stage2_finetune.py — Stage 2: Supervised fine-tuning.

Loads the MAE-pretrained encoder, adds a classification head,
and fine-tunes with focal loss + weighted sampling for rare classes.

Usage:
    python scripts/train_stage2_finetune.py
    python scripts/train_stage2_finetune.py stage2.training.lr=5e-5
    python scripts/train_stage2_finetune.py --resume              # Resume from best checkpoint
    python scripts/train_stage2_finetune.py --resume-from outputs/finetune_epoch025.pth
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

from raresight.data.dataset import build_dataloaders
from raresight.evaluation.metrics import evaluate
from raresight.models.classifier import RareDiseaseClassifier
from raresight.training.losses import build_loss
from raresight.training.trainer import Trainer


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ── Resume argument parsing ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint")
    try:
        args, _ = parser.parse_known_args()
    except:
        args = argparse.Namespace(resume=False, resume_from=None)
    
    torch.manual_seed(cfg.project.seed)
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("  RareSight — Stage 2: Supervised Fine-Tuning")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = build_dataloaders(
        data_root=cfg.data.root,
        batch_size=cfg.stage2.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        image_size=cfg.data.image_size,
        use_weighted_sampler=(cfg.stage2.sampler.type == "weighted_random"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder_cfg = OmegaConf.to_container(cfg.stage1.encoder, resolve=True)
    encoder_cfg.update({"patch_size": 16, "img_size": cfg.data.image_size})

    model = RareDiseaseClassifier(
        num_classes=cfg.stage2.model.num_classes,
        dropout=cfg.stage2.model.dropout,
        encoder_cfg=encoder_cfg,
    )

    ckpt = cfg.stage2.pretrained_checkpoint
    if Path(ckpt).exists():
        model.load_mae_weights(ckpt)
        logger.info(f"Loaded MAE weights from {ckpt}")
    else:
        logger.warning(f"No checkpoint found at {ckpt} — training from scratch.")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Classifier parameters: {n_params:.1f}M")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Approximate class counts for alpha weighting
    class_counts = list(cfg.data.class_counts.values())[:cfg.stage2.model.num_classes]
    loss_fn = build_loss(
        OmegaConf.to_container(cfg.stage2.loss, resolve=True),
        class_counts=class_counts,
    )

    # ── Optimizer with layer-wise LR decay ────────────────────────────────────
    param_groups = model.get_layer_groups(cfg.stage2.layer_decay)
    base_lr = cfg.stage2.training.lr
    optimizer_groups = [
        {"params": g["params"], "lr": base_lr * g["lr_scale"], "weight_decay": cfg.stage2.training.weight_decay}
        for g in param_groups
    ]
    optimizer = AdamW(optimizer_groups)

    total_steps = cfg.stage2.training.epochs * len(loaders["train"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # ── W&B ───────────────────────────────────────────────────────────────────
    run = None
    try:
        run = wandb.init(
            project=cfg.stage2.wandb.project,
            tags=list(cfg.stage2.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if run:
            run.watch(model, log_freq=100)
    except Exception:
        logger.warning("W&B not available — skipping.")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=Path(cfg.project.checkpoint_dir),
        mixed_precision=cfg.hardware.mixed_precision,
        gradient_clip=cfg.stage2.training.gradient_clip,
        wandb_run=run,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    epochs_no_improve = 0
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from}")
        start_epoch = trainer.load_checkpoint(args.resume_from, load_optimizer=True)
    elif args.resume:
        best_checkpoint = Path(cfg.project.checkpoint_dir) / f"{cfg.stage2.training.checkpoint_name}_best.pth"
        if best_checkpoint.exists():
            logger.info("Resuming from best checkpoint")
            start_epoch = trainer.load_checkpoint(best_checkpoint, load_optimizer=True)
        else:
            logger.warning("No best checkpoint found, starting from epoch 1")

    patience = cfg.stage2.training.early_stopping_patience

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.stage2.training.epochs + 1):
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

        improved = trainer.save_best(epoch, val_metrics, cfg.stage2.training.checkpoint_name)
        if not improved:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement.")
                break
        else:
            epochs_no_improve = 0

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("Running final test evaluation …")
    best_ckpt = Path(cfg.project.checkpoint_dir) / f"{cfg.stage2.training.checkpoint_name}_best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    results = evaluate(model, loaders["test"], device, num_classes=cfg.stage2.model.num_classes)
    logger.success(f"\nTest Results:\n{results.classification_report}")
    logger.success(f"  AUC macro:    {results.macro_auc:.4f}")
    logger.success(f"  Balanced Acc: {results.balanced_accuracy:.4f}")
    logger.success(f"  Cohen Kappa:  {results.cohen_kappa:.4f}")

    if run:
        run.log(results.to_dict())
        run.finish()

    logger.success("Stage 2 fine-tuning complete ✓")


if __name__ == "__main__":
    main()
