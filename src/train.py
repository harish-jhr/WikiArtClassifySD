import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from config import CFG
from dataset import build_activation_loaders, build_raw_loaders, load_class_names
from models import build_model
from trainer import Trainer


def class_weights_from_loader(train_loader, num_classes: int, device) -> torch.Tensor:
    # slearn 'balanced' mode: upweights rare classes so they contribute equally to loss
    all_labels = []
    for batch in train_loader:
        all_labels.append(batch[1].numpy())
    all_labels = np.concatenate(all_labels)
    classes    = np.arange(num_classes)
    weights    = compute_class_weight("balanced", classes=classes, y=all_labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_num_classes(task: str) -> int:
    class_txt = Path(CFG.data.csv_root) / task / f"{task}_class.txt"
    with open(class_txt) as f:
        return sum(1 for line in f if line.strip())


def train_convlstm(task: str, use_wandb: bool = True):
    #Train Conv-LSTM on spatial SD activations :
    print(f"\n{'='*60}")
    print(f"  Conv-LSTM | task={task}")
    print(f"{'='*60}")

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes  = get_num_classes(task)
    train_loader, val_loader = build_activation_loaders(task, spatial=True)
    model        = build_model("convlstm", num_classes)
    cw           = class_weights_from_loader(train_loader, num_classes, device)

    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        num_classes    = num_classes,
        run_name       = f"convlstm_{task}",
        epochs         = CFG.train.epochs,
        lr             = CFG.train.lr,
        class_weights  = cw,
        use_wandb      = use_wandb,
    )
    metrics = trainer.fit()
    print(f"[{task}/convlstm] Best val acc={metrics['acc']:.4f} "
          f"F1={metrics['macro_f1']:.4f}")
    return metrics


def train_mlp_probe(task: str, use_wandb: bool = True):
    #Train MLP linear probe on pooled SD activations.
    print(f"\n{'='*60}")
    print(f"  MLP Probe | task={task}")
    print(f"{'='*60}")

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes  = get_num_classes(task)
    train_loader, val_loader = build_activation_loaders(task, spatial=False)
    model        = build_model("mlp", num_classes)
    cw           = class_weights_from_loader(train_loader, num_classes, device)

    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        num_classes    = num_classes,
        run_name       = f"mlp_probe_{task}",
        epochs         = 15,
        lr             = 1e-3,
        class_weights  = cw,
        use_wandb      = use_wandb,
    )
    metrics = trainer.fit()
    print(f"[{task}/mlp] Best val acc={metrics['acc']:.4f}")
    return metrics


def train_resnet50(task: str, use_wandb: bool = True):
    #Train ResNet50 baseline on raw images
    print(f"\n{'='*60}")
    print(f"  ResNet50 Baseline | task={task}")
    print(f"{'='*60}")

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes  = get_num_classes(task)
    class_names  = load_class_names(
        Path(CFG.data.csv_root) / task / f"{task}_class.txt"
    )
    train_loader, val_loader, _ = build_raw_loaders(task)
    model        = build_model("resnet50", num_classes)
    cw           = class_weights_from_loader(train_loader, num_classes, device)

    trainer = Trainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        num_classes     = num_classes,
        run_name        = f"resnet50_{task}",
        epochs          = CFG.train.baseline_epochs,
        lr              = CFG.train.baseline_lr,
        unfreeze_epoch  = 5,
        class_names     = class_names,
        class_weights   = cw,
        use_wandb       = use_wandb,
    )
    metrics = trainer.fit()
    print(f"[{task}/resnet50] Best val acc={metrics['acc']:.4f} "
          f"F1={metrics['macro_f1']:.4f}")
    return metrics


MODEL_TRAIN_FNS = {
    "convlstm": train_convlstm,
    "mlp":      train_mlp_probe,
    "resnet50": train_resnet50,
}

def main(models: List[str], tasks: List[str], use_wandb: bool = True):
    seed_everything(CFG.train.seed)
    Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)

    summary = {}
    for task in tasks:
        summary[task] = {}
        for model_type in models:
            fn = MODEL_TRAIN_FNS[model_type]
            try:
                metrics = fn(task, use_wandb=use_wandb)
                summary[task][model_type] = metrics
            except Exception as e:
                print(f"[ERROR] {model_type}/{task}: {e}")
                import traceback; traceback.print_exc()

    # Final summary table
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for task, task_results in summary.items():
        for model_type, m in task_results.items():
            print(f"  {task:10s} | {model_type:12s} | "
                  f"acc={m.get('acc', 0):.4f} | f1={m.get('macro_f1', 0):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 Training")
    parser.add_argument("--model", nargs="+",
                        choices=["convlstm", "mlp", "resnet50"],
                        default=["convlstm", "mlp", "resnet50"])
    parser.add_argument("--task", nargs="+",
                        choices=["style", "artist", "genre"],
                        default=CFG.data.tasks)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Log locally (CSV + plots) instead of W&B")
    args = parser.parse_args()
    main(models=args.model, tasks=args.task, use_wandb=not args.no_wandb)
