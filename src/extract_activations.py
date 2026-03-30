import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CFG
from dataset import WikiArtDataset, get_transform, load_class_names, load_csv_split
from logger import Logger


class ActivationHooks:

    def __init__(self, unet: UNet2DConditionModel, layer_names: List[str]):
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks = []
        for name in layer_names:
            module = self._get_submodule(unet, name)
            self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    @staticmethod
    def _get_submodule(model, name: str):
        m = model
        for part in name.split("."):
            m = getattr(m, part)
        return m

    def _make_hook(self, name: str):
        def hook(module, input, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.activations[name] = tensor.detach().cpu()
        return hook

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()


@torch.no_grad()
def extract_batch(
    images: torch.Tensor,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    hooks: ActivationHooks,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    images  = images.to(device, dtype=torch.float16)
    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    noise   = torch.randn_like(latents)
    t       = torch.full((images.shape[0],), CFG.diffusion.timestep,
                         device=device, dtype=torch.long)
    noisy   = scheduler.add_noise(latents, noise, t)
    enc_hs  = torch.zeros(images.shape[0], 1, unet.config.cross_attention_dim,
                          device=device, dtype=torch.float16)
    hooks.clear()
    unet(noisy, t, encoder_hidden_states=enc_hs)
    return {k: v.clone() for k, v in hooks.activations.items()}



def pool_activations(act_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    pooled = [act_dict[name].mean(dim=(-2, -1)) for name in CFG.diffusion.hook_layers]
    return torch.cat(pooled, dim=1).numpy().astype(np.float16)


def spatial_activations(act_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    feat = act_dict["down_blocks.2"]
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(B, H * W, C).numpy().astype(np.float16)


def activation_stats(act_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    stats = {}
    for name, feat in act_dict.items():
        f32 = feat.float()
        tag = name.replace(".", "_")
        stats[f"act/{tag}/mean"] = f32.mean().item()
        stats[f"act/{tag}/std"]  = f32.std().item()
        stats[f"act/{tag}/norm"] = f32.norm(dim=1).mean().item()
    return stats



def _feat_to_heatmap(feat: torch.Tensor, img_h: int, img_w: int) -> np.ndarray:
    from PIL import Image as PILImage

    C, H, W = feat.shape
    f32 = feat.float()

    # PCA: (C, H*W) → first principal component → (H*W,) → (H, W)
    flat = f32.reshape(C, H * W)                  # (C, N)
    flat = flat - flat.mean(dim=1, keepdim=True)  # centre channels
    # SVD on (C, N) avoids forming the huge (N, N) spatial covariance matrix
    try:
        _, _, Vt = torch.linalg.svd(flat, full_matrices=False)
        heatmap = Vt[0].reshape(H, W).numpy()     # (H, W) first right singular vec
    except Exception:
        heatmap = f32.abs().max(dim=0).values.numpy()

    heatmap = np.abs(heatmap)                     # both sign polarities matter

    hmap_pil = PILImage.fromarray(heatmap.astype(np.float32))
    hmap_resized = np.array(
        hmap_pil.resize((img_w, img_h), PILImage.BICUBIC)
    ).astype(np.float32)

    lo, hi = hmap_resized.min(), hmap_resized.max()
    if hi > lo:
        hmap_resized = (hmap_resized - lo) / (hi - lo)

    return hmap_resized


def make_activation_panel(
    images: torch.Tensor,
    act_dict: Dict[str, torch.Tensor],
    class_names: Dict[int, str],
    labels: torch.Tensor,
    n_show: int = 4,
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show   = min(n_show, images.shape[0])
    n_layers = len(CFG.diffusion.hook_layers)
    n_rows   = 1 + n_layers
    n_cols   = n_show

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 3.2),
                             dpi=100)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Un-normalise: SD normalisation is [-1, 1] bac to [0, 1]
    imgs_np = (images[:n_show].cpu().float() * 0.5 + 0.5).clamp(0, 1)
    imgs_np = imgs_np.permute(0, 2, 3, 1).numpy()    # (N, H, W, 3)
    img_h, img_w = imgs_np.shape[1], imgs_np.shape[2]

    for col in range(n_show):
        # Row 0: original painting
        axes[0, col].imshow(imgs_np[col])
        lbl   = labels[col].item() if hasattr(labels[col], "item") else int(labels[col])
        title = class_names.get(int(lbl), str(lbl))
        axes[0, col].set_title(title, fontsize=7, pad=3)
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=7)

        # Rows 1+: PCA heatmap per layer, overlaid on painting
        for row, layer_name in enumerate(CFG.diffusion.hook_layers, start=1):
            feat    = act_dict[layer_name][col]       # (C, H_feat, W_feat)
            C, Hf, Wf = feat.shape
            heatmap = _feat_to_heatmap(feat, img_h, img_w)

            axes[row, col].imshow(imgs_np[col])
            im = axes[row, col].imshow(
                heatmap, cmap="inferno", alpha=0.6, vmin=0, vmax=1
            )
            axes[row, col].axis("off")
            if col == 0:
                # Show layer name AND its native spatial resolution
                axes[row, col].set_ylabel(
                    f"{layer_name}_({Hf}×{Wf})", fontsize=6
                )

    # Shared colorbar on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, cax=cbar_ax, label="Activation (PCA-1, normalised)")

    plt.suptitle("Paintings  :  UNet Activation Maps (PCA first component)",
                 fontsize=9, y=1.01)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    panel_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8
                             ).reshape(h, w, 3).copy()
    plt.close(fig)
    return panel_np

@torch.no_grad()
def probe_spatial_shape(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    hooks: ActivationHooks,
    device: torch.device,
) -> Tuple[int, int]:
    dummy   = torch.zeros(1, 3, CFG.data.image_size, CFG.data.image_size,
                          device=device, dtype=torch.float16)
    latents = vae.encode(dummy).latent_dist.sample() * vae.config.scaling_factor
    noise   = torch.randn_like(latents)
    t       = torch.full((1,), CFG.diffusion.timestep, device=device, dtype=torch.long)
    noisy   = scheduler.add_noise(latents, noise, t)
    enc_hs  = torch.zeros(1, 1, unet.config.cross_attention_dim,
                          device=device, dtype=torch.float16)
    hooks.clear()
    unet(noisy, t, encoder_hidden_states=enc_hs)

    feat     = hooks.activations["down_blocks.2"]
    seq_len  = feat.shape[2] * feat.shape[3]
    channels = feat.shape[1]
    print(f" down_blocs.2 shape={tuple(feat.shape[1:])}  "
          f"seq_len={seq_len}  channels={channels}")
    return seq_len, channels


class HDF5Writer:

    def __init__(self, path: Path, n_total: int,
                 feat_dim: int, seq_len: int, spatial_channels: int):
        self.f       = h5py.File(path, "w")
        self.ds_feat = self.f.create_dataset(
            "features", shape=(n_total, feat_dim),
            dtype="float16", chunks=(min(256, n_total), feat_dim),
        )
        self.ds_spat = self.f.create_dataset(
            "spatial", shape=(n_total, seq_len, spatial_channels),
            dtype="float16", chunks=(min(64, n_total), seq_len, spatial_channels),
        )
        self.ds_lbl  = self.f.create_dataset("labels", shape=(n_total,), dtype="int32")
        self.ds_path = self.f.create_dataset(
            "paths", shape=(n_total,), dtype=h5py.string_dtype()
        )
        self.idx = 0

    def write(self, pooled: np.ndarray, spatial: np.ndarray,
              labels: np.ndarray, paths: List[str]):
        s, e = self.idx, self.idx + len(labels)
        self.ds_feat[s:e] = pooled
        self.ds_spat[s:e] = spatial
        self.ds_lbl[s:e]  = labels
        self.ds_path[s:e] = [p.encode() for p in paths]
        self.idx += len(labels)

    def close(self):
        self.f.close()


def extract_split(
    task: str,
    split: str,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    hooks: ActivationHooks,
    device: torch.device,
    out_dir: Path,
    seq_len: int,
    spatial_channels: int,
    class_names: Dict[int, str],
    viz_every_n_batches: int = 50,
    use_wandb: bool = True,
):
    csv_path  = Path(CFG.data.csv_root) / task / f"{task}_{split}.csv"
    data_root = Path(CFG.data.data_root)

    paths, labels = load_csv_split(csv_path, data_root,
                                   images_per_class=None, seed=CFG.train.seed)
    n_total = len(paths)
    print(f"[{task}/{split}] {n_total} images")

    logger = Logger(
        project   = CFG.wandb.project,
        name      = f"extract_{task}_{split}",
        config    = {
            "task": task, "split": split,
            "model_id": CFG.diffusion.model_id,
            "timestep": CFG.diffusion.timestep,
            "n_images": n_total, "seq_len": seq_len,
            "feat_dim": CFG.diffusion.feature_dim,
        },
        use_wandb = use_wandb and CFG.wandb.enabled,
    )

    loader = DataLoader(
        WikiArtDataset(paths, labels,
                       get_transform(train=False, image_size=CFG.data.image_size)),
        batch_size=16,
        num_workers=CFG.data.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    out_path = out_dir / f"{task}_{split}.h5"
    writer   = HDF5Writer(out_path, n_total=n_total,
                          feat_dim=CFG.diffusion.feature_dim,
                          seq_len=seq_len, spatial_channels=spatial_channels)

    all_norms: List[float] = []
    t0 = time.time()
    images_done = 0

    for batch_idx, (batch_imgs, batch_labels, batch_paths) in enumerate(
        tqdm(loader, desc=f"extract {task}/{split}", unit="batch")
    ):
        batch_t0 = time.time()
        act_dict = extract_batch(batch_imgs, vae, unet, scheduler, hooks, device)
        pooled   = pool_activations(act_dict)
        spatial  = spatial_activations(act_dict)
        writer.write(pooled, spatial, np.array(batch_labels), list(batch_paths))

        batch_size    = len(batch_labels)
        images_done  += batch_size
        elapsed       = time.time() - t0
        imgs_per_sec  = images_done / elapsed
        batch_ms      = (time.time() - batch_t0) * 1000

        stats = activation_stats(act_dict)

        # trac pooled norms for histogram at end
        norms = np.linalg.norm(pooled.astype(np.float32), axis=1)
        all_norms.extend(norms.tolist())

        # GPU memory
        gpu_mem_gb = (torch.cuda.memory_allocated(device) / 1e9
                      if torch.cuda.is_available() else 0.0)

        log_dict = {
            "extraction/images_done":    images_done,
            "extraction/images_per_sec": imgs_per_sec,
            "extraction/batch_ms":       batch_ms,
            "extraction/progress_pct":   100 * images_done / n_total,
            "extraction/gpu_mem_gb":     gpu_mem_gb,
            **stats,
            "batch": batch_idx,
        }

        if batch_idx % viz_every_n_batches == 0:
            panel_np = make_activation_panel(
                batch_imgs, act_dict, class_names, batch_labels, n_show=4
            )
            logger.log_image(
                "activation_panels/batch", panel_np,
                caption="Row0=image | Row1=down_blocks.2 | Row2=mid_block | Row3=up_blocks.1",
            )

        logger.log(log_dict)

    writer.close()

    file_size_gb = out_path.stat().st_size / 1e9
    total_mins   = (time.time() - t0) / 60

    final_panel = make_activation_panel(
        batch_imgs, act_dict, class_names, batch_labels, n_show=4
    )
    logger.log_image("activation_panels/final", final_panel,
                     caption="Final batch activation panel")
    logger.log_histogram("summary/feat_norm_hist", all_norms)
    logger.log({
        "summary/total_images":     n_total,
        "summary/file_size_gb":     file_size_gb,
        "summary/total_mins":       total_mins,
        "summary/imgs_per_sec_avg": n_total / (total_mins * 60),
        "summary/feat_norm_mean":   float(np.mean(all_norms)),
        "summary/feat_norm_std":    float(np.std(all_norms)),
    })

    logger.finish()
    print(f"  -> {out_path}  ({file_size_gb:.2f} GB)  [{total_mins:.1f} min]")


def load_sd_components(device: torch.device):
    model_id = CFG.diffusion.model_id
    print(f"Loading {model_id} in fp16 ")
    vae  = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return vae, unet, scheduler


def main(tasks: Optional[List[str]] = None, use_wandb: bool = True):
    tasks   = tasks or CFG.data.tasks
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(CFG.data.activation_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, unet, scheduler = load_sd_components(device)
    hooks = ActivationHooks(unet, CFG.diffusion.hook_layers)

    seq_len, spatial_channels = probe_spatial_shape(vae, unet, scheduler, hooks, device)

    for task in tasks:
        for split in ("train", "val"):
            out_path = out_dir / f"{task}_{split}.h5"
            if out_path.exists():
                print(f" {out_path} already exists")
                continue
            class_names = load_class_names(
                Path(CFG.data.csv_root) / task / f"{task}_class.txt"
            )
            extract_split(task, split, vae, unet, scheduler, hooks,
                          device, out_dir, seq_len, spatial_channels,
                          class_names, use_wandb=use_wandb)

    hooks.remove()
    print("Activation extraction complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SD UNet activations")
    parser.add_argument("--task", nargs="+", choices=["style", "artist", "genre"])
    parser.add_argument("--no-wandb", action="store_true",
                        help="Log locally instead of W&B")
    args = parser.parse_args()
    main(tasks=args.task, use_wandb=not args.no_wandb)