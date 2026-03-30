import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

def _plot_training_curves(df: pd.DataFrame, out_path: Path, title: str):
    if "epoch" not in df.columns:
        return
    edf = df.dropna(subset=["epoch"]).copy()
    if len(edf) < 2:
        return

    # Determine subplot groups
    groups = []
    if any(c in edf for c in ("train/loss", "val/loss")):
        groups.append(("Loss",     [c for c in ("train/loss", "val/loss")     if c in edf]))
    if any(c in edf for c in ("train/acc", "val/acc")):
        groups.append(("Accuracy", [c for c in ("train/acc", "val/acc")       if c in edf]))
    if "val/macro_f1" in edf:
        groups.append(("Macro F1", ["val/macro_f1"]))
    if not groups:
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    x = edf["epoch"].values
    for ax, (ylabel, cols) in zip(axes[0], groups):
        for col in cols:
            label = col.replace("train/", "train ").replace("val/", "val ")
            ax.plot(x, edf[col].values, marker="o", markersize=3, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _save_histogram(data, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(data, bins=50, edgecolor="k", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


class Logger:

    def __init__(
        self,
        project: str,
        name: str,
        config: dict = None,
        use_wandb: bool = True,
        log_dir: str = "results/logs",
    ):
        self.name     = name
        self.log_dir  = Path(log_dir) / name
        self._history: list = []
        self._summary: dict = {}

        if use_wandb and not _WANDB_AVAILABLE:
            print("no wnadb, using local logging ")
            use_wandb = False

        self.use_wandb = use_wandb

        if use_wandb:
            try:
                _wandb.init(
                    project=project, name=name,
                    config=config or {}, reinit=True,
                )
                self._wb = _wandb
            except Exception as exc:
                print(f"W&B init failed ({exc}) — using local logging.")
                self.use_wandb = False
                self._wb = None
        else:
            self._wb = None

        if not self.use_wandb:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if config:
                (self.log_dir / "config.json").write_text(
                    json.dumps(config, indent=2, default=str)
                )

    # ── Core logging ──────────────────────────────────────────────────────

    def log(self, metrics: dict):
        scalars = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating, np.integer))
        }
        if scalars:
            self._history.append(scalars)
        if self.use_wandb:
            self._wb.log(metrics)

    def log_image(self, key: str, image, caption: str = ""):
        # image: str/Path to PNG or numpy (H,W,3) uint8 array
        if self.use_wandb:
            img_obj = (self._wb.Image(str(image), caption=caption)
                       if isinstance(image, (str, Path))
                       else self._wb.Image(image, caption=caption))
            self._wb.log({key: img_obj})
        elif isinstance(image, np.ndarray):
            from PIL import Image as _PIL
            out = self.log_dir / f"{key.replace('/', '_')}.png"
            _PIL.fromarray(image.astype(np.uint8)).save(out)

    def log_histogram(self, key: str, data):
        if self.use_wandb:
            self._wb.log({key: self._wb.Histogram(data)})
        else:
            out = self.log_dir / f"{key.replace('/', '_')}_hist.png"
            _save_histogram(data, out, title=key)

    def watch(self, model, **kwargs):
        if self.use_wandb:
            self._wb.watch(model, **kwargs)

    def summary(self, key: str, value):
        self._summary[key] = value
        if self.use_wandb and self._wb is not None and self._wb.run is not None:
            self._wb.run.summary[key] = value

    # ── Finish ────────────────────────────────────────────────────────────

    def finish(self):
        if not self.use_wandb:
            if self._history:
                df = pd.DataFrame(self._history)
                df.to_csv(self.log_dir / "metrics.csv", index=False)
                _plot_training_curves(df, self.log_dir / "training_curves.png",
                                      title=self.name)
            if self._summary:
                (self.log_dir / "summary.json").write_text(
                    json.dumps(self._summary, indent=2, default=str)
                )
            print(f"'{self.name}' saved → {self.log_dir}/")
        else:
            self._wb.finish()
