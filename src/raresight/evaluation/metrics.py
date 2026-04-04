"""raresight/evaluation/metrics.py — Comprehensive evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from raresight.data.dataset import CLASS_NAMES


@dataclass
class EvalResults:
    """Container for all evaluation metrics."""

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_auc: float = 0.0
    weighted_f1: float = 0.0
    macro_f1: float = 0.0
    cohen_kappa: float = 0.0
    per_class_auc: dict[str, float] = field(default_factory=dict)
    per_class_ap: dict[str, float] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    classification_report: str = ""

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_auc": self.macro_auc,
            "weighted_f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
            "cohen_kappa": self.cohen_kappa,
            **{f"auc_{k}": v for k, v in self.per_class_auc.items()},
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 8,
    class_names: list[str] | None = None,
) -> EvalResults:
    """Run full evaluation on a DataLoader."""
    class_names = class_names or CLASS_NAMES[:num_classes]
    model.eval()

    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = softmax(logits)
    preds = np.argmax(logits, axis=1)

    # ── Scalar metrics ──────────────────────────────────────────────────────────
    acc = np.mean(preds == labels)
    bal_acc = balanced_accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")

    # AUC / AP per class
    per_class_auc, per_class_ap = {}, {}
    for i, name in enumerate(class_names):
        bin_labels = (labels == i).astype(int)
        if bin_labels.sum() == 0:
            continue
        per_class_auc[name] = roc_auc_score(bin_labels, probs[:, i])
        per_class_ap[name]  = average_precision_score(bin_labels, probs[:, i])

    macro_auc = float(np.mean(list(per_class_auc.values())))

    # Full sklearn report
    report = classification_report(labels, preds, target_names=class_names, digits=4)

    # Extract F1 from report
    from sklearn.metrics import f1_score
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_f1    = f1_score(labels, preds, average="macro",    zero_division=0)

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    return EvalResults(
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
        macro_auc=macro_auc,
        weighted_f1=float(weighted_f1),
        macro_f1=float(macro_f1),
        cohen_kappa=float(kappa),
        per_class_auc=per_class_auc,
        per_class_ap=per_class_ap,
        confusion_matrix=cm,
        classification_report=report,
    )


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
