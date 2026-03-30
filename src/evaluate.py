import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA

from config import CFG
from dataset import build_activation_loaders, build_raw_loaders, load_class_names
from logger import Logger
from metrics import compute_metrics, compute_top5_accuracy, find_outliers
from models import build_model


def load_checkpoint(model, task: str, model_type: str):
    cpt_path = Path(CFG.output_dir) / "checkpoints" / f"{model_type}_{task}" / "best.pt"
    if not cpt_path.exists():
        raise FileNotFoundError(f"No checpoint at {cpt_path}")
    cpt = torch.load(cpt_path, map_location="cpu")
    model.load_state_dict(cpt["model_state"])
    print(f"[load] {cpt_path}  (epoch={cpt['epoch']}, val_acc={cpt['val_acc']:.4f})")
    return model


def get_num_classes(task: str) -> int:
    class_txt = Path(CFG.data.csv_root) / task / f"{task}_class.txt"
    with open(class_txt) as f:
        return sum(1 for line in f if line.strip())



@torch.no_grad()
def run_inference(model, loader, device) -> tuple:
    """Returns (all_logits, all_preds, all_labels) as numpy arrays."""
    model.eval()
    all_logits, all_preds, all_labels = [], [], []

    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1]
        with torch.cuda.amp.autocast():
            logits = model(inputs)
        all_logits.append(logits.cpu().float())
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Dict[int, str],
    title: str,
    out_path: Path,
    max_classes: int = 30,
):
    # Count samples per class, keep top-N by frequency
    unique, counts = np.unique(labels, return_counts=True)
    top_classes = unique[np.argsort(-counts)][:max_classes]

    mask  = np.isin(labels, top_classes)
    p, l  = preds[mask], labels[mask]
    names = [class_names.get(c, str(c)) for c in top_classes]

    cm = np.zeros((len(top_classes), len(top_classes)), dtype=int)
    idx_map = {c: i for i, c in enumerate(top_classes)}
    for pred, true in zip(p, l):
        if pred in idx_map and true in idx_map:
            cm[idx_map[true]][idx_map[pred]] += 1

    fig, ax = plt.subplots(figsize=(max(12, len(top_classes) * 0.5),
                                     max(10, len(top_classes) * 0.5)))
    sns.heatmap(cm, xticklabels=names, yticklabels=names,
                annot=len(top_classes) <= 15, fmt="d", ax=ax,
                cmap="Blues", linewidths=0.3)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  confusion matrix → {out_path}")


def plot_umap_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Dict[int, str],
    title: str,
    out_path: Path,
    n_classes_legend: int = 15,
):
    n_components = min(50, features.shape[1], features.shape[0] - 1)
    pca_feats = PCA(n_components=n_components).fit_transform(features)

    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedded = reducer.fit_transform(pca_feats)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        embedded = TSNE(n_components=2, random_state=42, perplexity=30
                        ).fit_transform(pca_feats)
        method = "t-SNE"

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(12, 10))
    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[cmap(i)], s=8, alpha=0.6,
                   label=class_names.get(cls, str(cls)))

    ax.set_title(f"{title} ({method})")
    ax.set_xticks([]); ax.set_yticks([])
    if len(unique_labels) <= n_classes_legend:
        ax.legend(fontsize=7, markerscale=2, loc="best",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  embedding plot at {out_path}")


def evaluate(model_type: str, task: str, use_wandb: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(CFG.output_dir) / "eval" / f"{model_type}_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = get_num_classes(task)
    class_names = load_class_names(
        Path(CFG.data.csv_root) / task / f"{task}_class.txt"
    )

    # Load model nd checpoint
    model = load_checkpoint(
        build_model(model_type, num_classes), task, model_type
    ).to(device)

    # Load data
    is_spatial = model_type == "convlstm"
    is_raw     = model_type == "resnet50"

    if is_raw:
        _, val_loader, _ = build_raw_loaders(task)
    else:
        _, val_loader = build_activation_loaders(task, spatial=is_spatial)

    # Run inference
    logits, preds, labels = run_inference(model, val_loader, device)
    #metrics:
    metrics = compute_metrics(preds, labels, num_classes)
    if task == "artist":   
        metrics["top5_acc"] = compute_top5_accuracy(logits, labels)

    print(f"\n[{model_type}/{task}]")
    print(f"  Accuracy:    {metrics['acc']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    if "top5_acc" in metrics:
        print(f"  Top-5 Acc:   {metrics['top5_acc']:.4f}")

    # Save metrics JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "per_class_acc"}, f, indent=2)

    plot_confusion_matrix(
        preds, labels, class_names,
        title=f"{model_type} | {task}",
        out_path=out_dir / "confusion_matrix.png",
    )

    if not is_raw:
        cache_path = Path(CFG.data.activation_cache_dir) / f"{task}_val_pooled.h5"
        if cache_path.exists():
            with h5py.File(cache_path, "r") as f:
                features = f["features"][:].astype(np.float32)
                paths    = [p.decode() for p in f["paths"][:]]
            plot_umap_embeddings(
                features, labels, class_names,
                title=f"SD Features | {task}",
                out_path=out_dir / "embeddings.png",
            )

            print(f"\n outliers in tas={task}")
            outliers = find_outliers(features, labels, paths, n_sigma=2.5)
            outlier_report = {}
            for cls_idx, items in outliers.items():
                cls_name = class_names.get(cls_idx, str(cls_idx))
                outlier_report[cls_name] = items[:10]  # top 10 outliers per class
                print(f"  {cls_name}: {len(items)} outliers")

            with open(out_dir / "outliers.json", "w") as f:
                json.dump(outlier_report, f, indent=2)

    logger = Logger(
        project   = CFG.wandb.project,
        name      = f"eval_{model_type}_{task}",
        config    = {"model": model_type, "task": task},
        use_wandb = use_wandb and CFG.wandb.enabled,
    )
    logger.log({
        f"eval/{task}/acc":         metrics["acc"],
        f"eval/{task}/macro_f1":    metrics["macro_f1"],
        f"eval/{task}/weighted_f1": metrics["weighted_f1"],
    })
    logger.log_image("confusion_matrix", str(out_dir / "confusion_matrix.png"))
    if (out_dir / "embeddings.png").exists():
        logger.log_image("embeddings", str(out_dir / "embeddings.png"))
    logger.finish()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tas 1")
    parser.add_argument("--model", choices=["convlstm", "mlp_probe", "resnet50"],
                        default="convlstm")
    parser.add_argument("--task", nargs="+",
                        choices=["style", "artist", "genre"],
                        default=CFG.data.tasks)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Log locally instead of W&B")
    args = parser.parse_args()
    for task in args.task:
        evaluate(args.model, task, use_wandb=not args.no_wandb)
