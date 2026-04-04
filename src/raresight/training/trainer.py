"""raresight/training/trainer.py — Generic training engine."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm


class Trainer:
    """Reusable training loop with AMP, gradient clipping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        loss_fn: nn.Module | None,
        device: torch.device,
        checkpoint_dir: Path,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        log_every: int = 50,
        wandb_run=None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = GradScaler(enabled=mixed_precision)
        self.gradient_clip = gradient_clip
        self.log_every = log_every
        self.wandb = wandb_run
        self.best_metric = float("inf")

    # ── Training step ──────────────────────────────────────────────────────────

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(loader, desc=f"Train E{epoch}", leave=False)):
            images = batch["image"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                if self.loss_fn is None:
                    # MAE: model returns dict with 'loss'
                    out = self.model(images)
                    loss = out["loss"]
                else:
                    labels = batch["label"].to(self.device, non_blocking=True)
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"  step={step}  loss={loss.item():.4f}  lr={lr:.2e}")
                if self.wandb:
                    self.wandb.log({"train/loss_step": loss.item(), "lr": lr})

        return total_loss / len(loader)

    # ── Evaluation step (classification) ───────────────────────────────────────

    @torch.no_grad()
    def eval_epoch(self, loader) -> dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(loader, desc="Val", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            "val/loss": total_loss / len(loader),
            "val/acc":  correct / total,
        }

    # ── Checkpoint helpers ─────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, metrics: dict, name: str) -> Path:
        path = self.checkpoint_dir / f"{name}_epoch{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        logger.info(f"Saved checkpoint → {path}")
        return path

    def save_best(self, epoch: int, metrics: dict, name: str, monitor: str = "val/loss") -> bool:
        value = metrics[monitor]
        is_best = value < self.best_metric
        if is_best:
            self.best_metric = value
            path = self.checkpoint_dir / f"{name}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
            }, path)
            logger.success(f"New best ({monitor}={value:.4f}) → {path}")
        return is_best

    def load_checkpoint(self, checkpoint_path: str | Path, load_optimizer: bool = True) -> int:
        """Load checkpoint and resume training from saved epoch.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to restore optimizer state (disable for fine-tuning)
            
        Returns:
            Next epoch to start training from
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model weights from {checkpoint_path}")
        
        # Restore optimizer & scheduler (optional, for resuming training)
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored optimizer state")
        
        # Update best metric
        if "metrics" in checkpoint and "val/loss" in checkpoint["metrics"]:
            self.best_metric = checkpoint["metrics"]["val/loss"]
        
        next_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resuming from epoch {next_epoch}")
        return next_epoch

    # ── Scheduler step ────────────────────────────────────────────────────────

    def step_scheduler(self, epoch: int, val_loss: float | None = None) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
