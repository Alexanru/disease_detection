"""raresight/training/losses.py — Loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) — down-weights easy examples.

    Especially useful for rare/imbalanced classes.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: torch.Tensor | float | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(smooth_targets * log_probs).sum(dim=1)
            pt = (smooth_targets * probs).sum(dim=1)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device)[targets]
            else:
                alpha_t = self.alpha
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg: dict, class_counts: list[int] | None = None) -> nn.Module:
    """Factory function — builds loss from config."""
    loss_type = cfg.get("type", "focal")

    alpha = None
    if cfg.get("alpha") == "auto" and class_counts is not None:
        counts = torch.tensor(class_counts, dtype=torch.float32)
        alpha = 1.0 / counts
        alpha = alpha / alpha.sum()

    if loss_type == "focal":
        return FocalLoss(
            alpha=alpha,
            gamma=cfg.get("gamma", 2.0),
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    if loss_type == "cross_entropy":
        weight = alpha if alpha is not None else None
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.get("label_smoothing", 0.0))

    raise ValueError(f"Unknown loss type: {loss_type}")
