import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset

DATA_DIR = Path("data")
IMG_DIR  = DATA_DIR / "wikiart"
CSV_DIR  = DATA_DIR / "csvs"
TASKS    = ["style", "artist", "genre"]
VAL_FRAC = 0.15
SEED     = 42


def stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls].copy()
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_frac))
        val.extend(cls_idx[:n_val])
        train.extend(cls_idx[n_val:])
    return np.array(train), np.array(val)


def subsample_per_class(
    indices: np.ndarray,
    labels: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kept = []
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls].copy()
        rng.shuffle(cls_idx)
        kept.extend(cls_idx[:n])
    return np.array(kept)


# ── CSV / txt writers ─────────────────────────────────────────────────────────

def write_csv(rows: List[Tuple[str, int]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def write_class_txt(class_names: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for idx, name in enumerate(class_names):
            f.write(f"{name} {idx}\n")


def main(images_per_class: Optional[int]):
    print("HF :  Loading label metadata from huggan/wikiart")
    ds = load_dataset(
        "huggan/wikiart",
        split="train",
        trust_remote_code=True,
    ).remove_columns(["image"])
    n_total = len(ds)
    print(f"[hf] {n_total} records loaded.")

    print("Scanning for existing images")
    on_disk = np.array(
        [i for i in range(n_total) if (IMG_DIR / f"{i}.jpg").exists()],
        dtype=np.int64,
    )
    print(f"[disk] {len(on_disk)} / {n_total} images found on disk")
    if len(on_disk) == 0:
        raise RuntimeError(
            f"No images found in {IMG_DIR}. "
        )
    for task in TASKS:
        print(f"\n[{task}] Building splits ...")
        csv_dir = CSV_DIR / task

        # Class names come from the HF ClassLabel feature schema
        class_names: List[str] = ds.features[task].names
        write_class_txt(class_names, csv_dir / f"{task}_class.txt")

        # Fetch all labels as a flat int array : fast columnar access
        all_labels = np.array(ds[task], dtype=np.int32)

        # Restrict to images that exist on disk
        labels_on_disk = all_labels[on_disk]

        # Subsample
        if images_per_class is not None:
            selected = subsample_per_class(
                on_disk, labels_on_disk, images_per_class, SEED
            )
        else:
            selected = on_disk

        selected_labels = all_labels[selected]
        train_idx, val_idx = stratified_split(
            selected, selected_labels, VAL_FRAC, SEED
        )
        print(f"  train={len(train_idx)}  val={len(val_idx)}  "
              f"classes={len(class_names)}")

        write_csv(
            [(f"{i}.jpg", int(all_labels[i])) for i in train_idx],
            csv_dir / f"{task}_train.csv",
        )
        write_csv(
            [(f"{i}.jpg", int(all_labels[i])) for i in val_idx],
            csv_dir / f"{task}_val.csv",
        )
        print(f"  Written to {csv_dir}/")

    print("All CSVs ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build WikiArt train/val CSV splits from downloaded images."
    )
    parser.add_argument(
        "--images-per-class",
        type=lambda x: None if x == "None" else int(x),
        default=150,
    )
    args = parser.parse_args()
    main(images_per_class=args.images_per_class)