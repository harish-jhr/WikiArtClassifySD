"""
Central Config
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    data_root: str = "data/wikiart"
    csv_root: str = "data/csvs"
    activation_cache_dir: str = "activations"
    tasks: List[str] = field(default_factory=lambda: ["style", "artist", "genre"])
    # 150 imgs/class × 27 style classes
    images_per_class: Optional[int] = 150
    image_size: int = 512   # resizing for SD native resolution
    num_workers: int = 8


@dataclass
class DiffusionConfig:
    model_id: str = "valhalla/sd-wikiart-v2"

    # Noise timestep for activation extraction.
    # t=200 (out of 1000)(shown by multiple papers to be appropriate for extracting semantically rich features : loo at https://github.com/harish-jhr/DiffuseSeg for instance)
    timestep: int = 200

    # SD v1 U-Net channels: down(320,640,1280,1280) | mid(1280) | up(1280,1280,640,320)
    #
    # I choose 3 layers to hoo for feature extraction:
    #   "down_blocs.2"  → 1280-ch, 16×16 spatial  — coarse semantic (style/artist)
    #   "mid_bloc"      → 1280-ch,  8×8 spatial    — bottlenec, most abstract
    #   "up_blocs.1"    → 1280-ch, 16×16 spatial   — decoder early, good for genre
    #
    # After global-average-pooling each gives a 1280-d vector.
    # Concatenated feature dim = 3 × 1280 = 3840.
    hook_layers: List[str] = field(default_factory=lambda: [
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
    ])

    feature_dim: int = 3840  # 3 layers × 1280 channels


@dataclass
class TrainConfig:
    # ConvLSTM classifier : 
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 7            # early stopping patience

    # Spatial grid size fed to LSTM (spatial toens from UNet feature maps)
    # down_blocs.2 gives 16×16 → 256 spatial toens of dim 1280
    spatial_tokens: int = 256    # 16×16
    lstm_hidden: int = 512
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    classifier_dropout: float = 0.4

    #CNN (ResNet50 fine-tune) :
    baseline_epochs: int = 20
    baseline_lr: float = 1e-4

    seed: int = 42


@dataclass
class WandbConfig:
    project: str = "artextract-task1"
    entity: Optional[str] = None
    log_every_n_steps: int = 50
    enabled: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "results_2"

CFG = Config()
