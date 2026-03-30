from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict:
    acc         = (preds == labels).mean()
    macro_f1    = f1_score(labels, preds, average="macro",  zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    # Perclass accuracy
    per_class = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == labels[mask]).mean()

    return {
        "acc":           float(acc),
        "macro_f1":      float(macro_f1),
        "weighted_f1":   float(weighted_f1),
        "per_class_acc": per_class,
    }


def compute_top5_accuracy(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Top-5 accuracy : relevant for artist classification (129 classes)."""
    return float(top_k_accuracy_score(labels, logits, k=5))


def get_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    cm = confusion_matrix(labels, preds)
    return cm, class_names


def find_outliers(
    features: np.ndarray,
    labels: np.ndarray,
    image_paths: Optional[List[str]] = None,
    n_sigma: float = 2.5,
) -> Dict[int, List[dict]]:
    # paintings whose distance to class centroid > n_sigma stds are flagged as outliers
    outliers: Dict[int, List[dict]] = {}

    for cls in np.unique(labels):
        mask  = labels == cls
        feats = features[mask]             # (n_cls, D)

        centroid = feats.mean(axis=0)
        dists    = np.linalg.norm(feats - centroid, axis=1)

        mean_d, std_d = dists.mean(), dists.std()
        if std_d < 1e-8:
            continue

        z_scores = (dists - mean_d) / std_d
        outlier_mask = z_scores > n_sigma

        if not outlier_mask.any():
            continue

        cls_paths = (
            [image_paths[i] for i, m in enumerate(mask) if m]
            if image_paths else [None] * mask.sum()
        )

        outliers[int(cls)] = [
            {
                "path":     cls_paths[i],
                "distance": float(dists[i]),
                "z_score":  float(z_scores[i]),
            }
            for i in np.where(outlier_mask)[0]
        ]

    total = sum(len(v) for v in outliers.values())
    pct   = 100 * total / len(labels)
    print(f" Found {total} outliers ({pct:.1f}%) across "
          f"{len(outliers)} classes at {n_sigma}σ threshold.")
    return outliers
