import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import CFG


def get_transform(train: bool, image_size: int = 512) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # SD convention
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_csv_split(
    csv_path: Path,
    data_root: Path,
    images_per_class: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(csv_path, header=None, names=["rel_path", "label"])
    df["abs_path"] = df["rel_path"].apply(lambda p: str(data_root / Path(p).name))
    exists_mask = df["abs_path"].apply(os.path.exists)
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f" {missing} paths not found")
    df = df[exists_mask].reset_index(drop=True)

    if images_per_class is not None:
        rng = np.random.default_rng(seed)
        groups = []
        for label, group in df.groupby("label"):
            n = min(len(group), images_per_class)
            groups.append(group.sample(n=n, random_state=int(rng.integers(1 << 31))))
        df = pd.concat(groups).reset_index(drop=True)
        print(f"  Subsampled to {len(df)} images ({images_per_class}/class max).")

    return df["abs_path"].tolist(), df["label"].tolist()


def load_class_names(class_txt: Path) -> Dict[int, str]:
    """Parse tas_class.txt to (index: name) mapping."""
    mapping = {}
    with open(class_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if parts[-1].isdigit():
                mapping[int(parts[-1])] = parts[0]
            else:
                parts = line.split(" ", 1)
                mapping[int(parts[0])] = parts[1]
    return mapping


# Raw img dataset for baseline training (or) actuvation extraction 

class WikiArtDataset(Dataset):
    #Loads raw paintings for baseline CNN training or activation extraction.

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: transforms.Compose,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Return a black image on corrupt file — logged by trainer
            img = Image.new("RGB", (CFG.data.image_size, CFG.data.image_size))
        return self.transform(img), self.labels[idx], path


# to process activatiions produced by extract_activations.py

class ActivationDataset(Dataset):

    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self._file   = h5py.File(h5_path, "r")
        self.features = self._file["features"]   # lazy — (N, feat_dim)
        self.labels   = self._file["labels"][:]
        self.paths    = self._file["paths"][:]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        feat = torch.from_numpy(self.features[idx].astype(np.float32))
        return feat, int(self.labels[idx])

    def __del__(self):
        if hasattr(self, "_file"):
            self._file.close()


# for Conv-LSTM
class SpatialActivationDataset(Dataset):

    def __init__(self, h5_path: Path):
        self._file   = h5py.File(h5_path, "r")
        self.spatial = self._file["spatial"]     
        self.labels  = self._file["labels"][:]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq = torch.from_numpy(self.spatial[idx].astype(np.float32))
        return seq, int(self.labels[idx])

    def __del__(self):
        if hasattr(self, "_file"):
            self._file.close()


# builds train and val loaders

def build_raw_loaders(task: str) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    csv_dir  = Path(CFG.data.csv_root) / task
    data_root = Path(CFG.data.data_root)

    train_paths, train_labels = load_csv_split(
        csv_dir / f"{task}_train.csv", data_root,
        images_per_class=None, seed=CFG.train.seed,
    )
    val_paths, val_labels = load_csv_split(
        csv_dir / f"{task}_val.csv", data_root,
        images_per_class=None, seed=CFG.train.seed,
    )
    class_names = load_class_names(csv_dir / f"{task}_class.txt")

    print(f"[{task}] train={len(train_labels)} | val={len(val_labels)} | "
          f"classes={len(class_names)}")

    train_ds = WikiArtDataset(train_paths, train_labels,
                               get_transform(train=True, image_size=CFG.data.image_size))
    val_ds   = WikiArtDataset(val_paths, val_labels,
                               get_transform(train=False, image_size=CFG.data.image_size))

    train_loader = DataLoader(train_ds, batch_size=CFG.train.batch_size,
                              shuffle=True, num_workers=CFG.data.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG.train.batch_size,
                              shuffle=False, num_workers=CFG.data.num_workers,
                              pin_memory=True)
    return train_loader, val_loader, class_names


def build_activation_loaders(task: str, spatial: bool = False
                              ) -> Tuple[DataLoader, DataLoader]:
    cache_dir = Path(CFG.data.activation_cache_dir)
    Cls       = SpatialActivationDataset if spatial else ActivationDataset
    train_ds = Cls(cache_dir / f"{task}_train.h5")
    val_ds   = Cls(cache_dir / f"{task}_val.h5")

    train_loader = DataLoader(train_ds, batch_size=CFG.train.batch_size,
                              shuffle=True, num_workers=CFG.data.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG.train.batch_size,
                              shuffle=False, num_workers=CFG.data.num_workers, pin_memory=True)
    return train_loader, val_loader

