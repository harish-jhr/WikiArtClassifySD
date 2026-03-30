import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from logger import Logger
from metrics import compute_metrics


class EarlyStopping:

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        run_name: str,
        epochs: int = None,
        lr: float = None,
        unfreeze_epoch: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        class_weights: Optional[torch.Tensor] = None,
        use_wandb: bool = True,
    ):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.num_classes  = num_classes
        self.run_name     = run_name
        self.epochs       = epochs or CFG.train.epochs
        self.unfreeze_epoch = unfreeze_epoch
        self.class_names  = class_names or {}
        self.use_wandb    = use_wandb and CFG.wandb.enabled

        # Class-weighted loss addresses imbalance :
        weight = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=CFG.train.label_smoothing,
            weight=weight,
        )
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr or CFG.train.lr,
            weight_decay=CFG.train.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        self.scaler = torch.cuda.amp.GradScaler()

        self.early_stop = EarlyStopping(patience=CFG.train.patience)
        self.cpt_dir   = Path(CFG.output_dir) / "checkpoints" / run_name
        self.cpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.logger: Optional[Logger] = None

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        step = epoch * len(self.train_loader)

        for batch_idx, batch in enumerate(tqdm(
            self.train_loader, desc=f"Train {epoch+1}", leave=False
        )):
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                logits = self.model(inputs)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1)
            total_correct  += (preds == labels).sum().item()
            total_samples  += labels.size(0)
            total_loss     += loss.item() * labels.size(0)

            if (batch_idx + 1) % CFG.wandb.log_every_n_steps == 0:
                self.logger.log({
                    "train/loss_step": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "global_step": step + batch_idx,
                })

        return {
            "loss": total_loss / total_samples,
            "acc":  total_correct / total_samples,
        }

    # Validation :

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_preds, all_labels = [], []

        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

            with torch.cuda.amp.autocast():
                logits = self.model(inputs)
                loss   = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total_loss    += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        metrics = compute_metrics(all_preds, all_labels, self.num_classes)
        metrics["loss"] = total_loss / total_samples
        return metrics

    # Save / load checpoint:

    def _save_checkpoint(self, epoch: int, val_acc: float):
        path = self.cpt_dir / f"best.pt"
        torch.save({
            "epoch":      epoch,
            "model_state": self.model.state_dict(),
            "opt_state":  self.optimizer.state_dict(),
            "val_acc":    val_acc,
        }, path)

    def fit(self) -> Dict:
        run_cfg = {
            "model":            self.run_name,
            "num_classes":      self.num_classes,
            "epochs":           self.epochs,
            "lr":               CFG.train.lr,
            "batch_size":       CFG.train.batch_size,
            "images_per_class": CFG.data.images_per_class,
            "timestep":         CFG.diffusion.timestep,
            "hook_layers":      CFG.diffusion.hook_layers,
        }
        self.logger = Logger(
            project   = CFG.wandb.project,
            name      = self.run_name,
            config    = run_cfg,
            use_wandb = self.use_wandb,
        )
        self.logger.watch(self.model, log="gradients",
                          log_freq=CFG.wandb.log_every_n_steps)

        best_metrics = {}
        t0 = time.time()

        for epoch in range(self.epochs):
            if self.unfreeze_epoch and epoch == self.unfreeze_epoch:
                if hasattr(self.model, "unfreeze_all"):
                    self.model.unfreeze_all()
                    for g in self.optimizer.param_groups:
                        g["lr"] = CFG.train.baseline_lr / 10
                    print(f"[epoch {epoch}] Unfroze full bacbone.")

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch()
            self.scheduler.step()

            log_dict = {
                "epoch":        epoch + 1,
                "train/loss":   train_metrics["loss"],
                "train/acc":    train_metrics["acc"],
                "val/loss":     val_metrics["loss"],
                "val/acc":      val_metrics["acc"],
                "val/macro_f1": val_metrics["macro_f1"],
                "val/top5_acc": val_metrics.get("top5_acc", 0),
            }
            self.logger.log(log_dict)

            elapsed = (time.time() - t0) / 60
            print(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"TrainLoss={train_metrics['loss']:.4f} TrainAcc={train_metrics['acc']:.4f} | "
                f"ValLoss={val_metrics['loss']:.4f} ValAcc={val_metrics['acc']:.4f} | "
                f"F1={val_metrics['macro_f1']:.4f} | {elapsed:.1f}min"
            )

            if val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                best_metrics      = val_metrics
                self._save_checkpoint(epoch, val_metrics["acc"])
                self.logger.summary("best_val_acc",  val_metrics["acc"])
                self.logger.summary("best_macro_f1", val_metrics["macro_f1"])
                self.logger.summary("best_epoch",    epoch + 1)

            if self.early_stop.step(val_metrics["loss"]):
                print(f" No improvement for {CFG.train.patience} epochs, stopping training")
                break

        self.logger.finish()
        return best_metrics
