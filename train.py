from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from config.config import FALAPSConfig
from FALVPSAe import FeatureAlignedLViTPixelShuffleAutoencoder

from losses.losses_fixed import CompositeReconstructionLoss
from datasets.dataset import (
    SelectedVideoDataset,
    split_video_directories,
    create_dataloader,
    seed_everything,
)


class TrainConfig:
    # ---------------- Dataset ----------------
    OFFICIAL_TRAIN_DIR = Path(r"C:\Users\USER\Desktop\MGST_All\eidetic_vad-main\eidetic_vad-main_Shanghaitech\data\Shanghai_train")
    OFFICIAL_TEST_DIR = Path(r"./data/test")  # safety check only; never loaded

    IMAGE_SIZE = (256, 256)
    TRAIN_RATIO = 0.90
    FRAME_STRIDE = 1

    # ---------------- Loader -----------------
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # --------------- Training ---------------
    EPOCHS = 100
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4

    LAMBDA_MSE = 0.5
    LAMBDA_SSIM = 1.0
    LAMBDA_GRAD = 0.2

    # Disabled by default because the current LViT/FASC/PSFE forward path
    # produces non-finite activations under FP16 autocast.
    # Re-enable only after the model's AMP-sensitive block is identified/fixed.
    USE_AMP = False
    GRAD_CLIP_NORM = 1.0

    USE_COSINE_SCHEDULER = True
    MIN_LR = 1e-6

    EARLY_STOPPING = True
    PATIENCE = 15
    MIN_DELTA = 1e-5

    CHECKPOINT_DIR = Path("./checkpoints_FALAPS")
    BEST_MODEL_NAME = "best_model.pth"
    LAST_MODEL_NAME = "last_model.pth"

    RESUME_TRAINING = False
    RESUME_PATH = CHECKPOINT_DIR / LAST_MODEL_NAME

    SEED = 42

    MODEL = FALAPSConfig(
        img_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        reduction_ratio=4,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.0,
        kernel="exp",
        skip_indices=(1, 3, 5, 7),
        decoder_channels=(256, 128, 64, 32, 16),
        num_fasc=4,
        conv_norm="batch",
    )


def resolve_path(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def verify_dataset_separation(cfg: TrainConfig) -> Tuple[Path, Path]:
    """Guard against accidental use/mixing of the official test partition."""
    train_root = resolve_path(cfg.OFFICIAL_TRAIN_DIR)
    test_root = resolve_path(cfg.OFFICIAL_TEST_DIR)

    if not train_root.exists():
        raise FileNotFoundError(f"Official training directory does not exist: {train_root}")

    if train_root == test_root:
        raise RuntimeError("OFFICIAL_TRAIN_DIR and OFFICIAL_TEST_DIR are identical.")

    if is_relative_to(train_root, test_root):
        raise RuntimeError("Official training directory is nested inside official test directory.")

    if is_relative_to(test_root, train_root):
        raise RuntimeError("Official test directory is nested inside official training directory.")

    return train_root, test_root


def build_train_val_datasets(cfg: TrainConfig):
    """
    Split ONLY official training video folders into train/validation.
    Official TEST data is never passed here.
    """
    train_root, _ = verify_dataset_separation(cfg)

    train_video_dirs, val_video_dirs = split_video_directories(
        root_dir=train_root,
        train_ratio=cfg.TRAIN_RATIO,
        seed=cfg.SEED,
    )

    if len(train_video_dirs) == 0 or len(val_video_dirs) == 0:
        raise RuntimeError(
            "Video-level split produced an empty train or validation set. "
            "Check the number of official training video folders."
        )

    train_paths = {resolve_path(p) for p in train_video_dirs}
    val_paths = {resolve_path(p) for p in val_video_dirs}
    overlap = train_paths & val_paths
    if overlap:
        raise RuntimeError(f"Train/validation leakage detected: {overlap}")

    train_dataset = SelectedVideoDataset(
        video_dirs=train_video_dirs,
        image_size=cfg.IMAGE_SIZE,
        stride=cfg.FRAME_STRIDE,
    )

    val_dataset = SelectedVideoDataset(
        video_dirs=val_video_dirs,
        image_size=cfg.IMAGE_SIZE,
        stride=cfg.FRAME_STRIDE,
    )

    print("\n==============================================")
    print("Official benchmark partition policy")
    print("==============================================")
    print(f"Official TRAIN root : {train_root}")
    print("Official TEST root  : NOT USED DURING TRAINING")
    print(f"Training videos     : {len(train_video_dirs)}")
    print(f"Validation videos   : {len(val_video_dirs)}")
    print(f"Training frames     : {len(train_dataset)}")
    print(f"Validation frames   : {len(val_dataset)}")
    print("==============================================\n")

    return train_dataset, val_dataset, train_video_dirs, val_video_dirs


def build_dataloaders(cfg: TrainConfig):
    train_dataset, val_dataset, train_video_dirs, val_video_dirs = build_train_val_datasets(cfg)

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset, val_dataset, train_video_dirs, val_video_dirs


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return 0.0 if self.count == 0 else self.sum / self.count

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n


def new_loss_meters():
    return {k: AverageMeter() for k in ("total", "mse", "ssim", "grad")}


def _tensor_finite(name: str, x: torch.Tensor, batch_idx: int | None = None) -> None:
    """Raise a useful error as soon as NaN/Inf enters the training pipeline."""
    if torch.isfinite(x).all():
        return

    finite = x[torch.isfinite(x)]
    location = f" at batch {batch_idx}" if batch_idx is not None else ""

    if finite.numel() > 0:
        details = (
            f"finite_min={finite.min().item():.6e}, "
            f"finite_max={finite.max().item():.6e}"
        )
    else:
        details = "tensor contains no finite values"

    raise FloatingPointError(
        f"Non-finite values detected in {name}{location}; {details}"
    )


def _check_model_state_finite(model: nn.Module, where: str) -> None:
    """Check parameters and persistent floating-point buffers (e.g. BatchNorm stats)."""
    bad = []

    for name, param in model.named_parameters():
        if param is not None and not torch.isfinite(param).all():
            bad.append(f"parameter: {name}")

    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and not torch.isfinite(buf).all():
            bad.append(f"buffer: {name}")

    if bad:
        preview = "\n".join(bad[:20])
        extra = "" if len(bad) <= 20 else f"\n... and {len(bad) - 20} more"
        raise FloatingPointError(
            f"Model state became non-finite {where}:\n{preview}{extra}"
        )


def _finite_grad_norm(model: nn.Module) -> bool:
    """Return False if any existing gradient contains NaN/Inf."""
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip_norm: float,
):
    model.train()
    meters = new_loss_meters()
    skipped_batches = 0

    for batch_idx, batch in enumerate(loader):
        inputs = batch["input"].to(device, non_blocking=True).float()
        targets = batch["target"].to(device, non_blocking=True).float()

        _tensor_finite("training input", inputs, batch_idx)
        _tensor_finite("training target", targets, batch_idx)

        optimizer.zero_grad(set_to_none=True)

        # AMP is controlled by cfg.USE_AMP. It is disabled by default because
        # the current model has shown FP16 forward instability.
        with autocast(enabled=use_amp and device.type == "cuda"):
            reconstruction = model(inputs)

        # Detect the failure at the model boundary rather than inside the loss.
        if not torch.isfinite(reconstruction).all():
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            print(
                f"[WARNING] Skipping training batch {batch_idx}: "
                "model output contains NaN/Inf."
            )
            continue

        # The loss implementation performs its sensitive calculations in FP32.
        loss_dict = criterion(reconstruction, targets)
        total_loss = loss_dict["total_loss"]

        if not torch.isfinite(total_loss):
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            print(
                f"[WARNING] Skipping training batch {batch_idx}: "
                f"non-finite loss "
                f"(total={loss_dict['total_loss'].item()}, "
                f"mse={loss_dict['mse_loss'].item()}, "
                f"ssim={loss_dict['ssim_loss'].item()}, "
                f"grad={loss_dict['gradient_loss'].item()})."
            )
            continue

        scaler.scale(total_loss).backward()

        # IMPORTANT: gradients must be unscaled before checking/clipping.
        scaler.unscale_(optimizer)

        if not _finite_grad_norm(model):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            skipped_batches += 1
            print(
                f"[WARNING] Skipping training batch {batch_idx}: "
                "non-finite gradients detected."
            )
            continue

        if grad_clip_norm is not None and grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_norm,
            )

            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                skipped_batches += 1
                print(
                    f"[WARNING] Skipping training batch {batch_idx}: "
                    "gradient norm is non-finite."
                )
                continue

        scaler.step(optimizer)
        scaler.update()

        # Catch optimizer corruption immediately.
        _check_model_state_finite(
            model,
            where=f"after optimizer step at training batch {batch_idx}",
        )

        bs = inputs.size(0)
        meters["total"].update(loss_dict["total_loss"].item(), bs)
        meters["mse"].update(loss_dict["mse_loss"].item(), bs)
        meters["ssim"].update(loss_dict["ssim_loss"].item(), bs)
        meters["grad"].update(loss_dict["gradient_loss"].item(), bs)

    if meters["total"].count == 0:
        raise FloatingPointError(
            "Every training batch was skipped because of non-finite "
            "model outputs/losses/gradients. Disable AMP (default in this file) "
            "and inspect the model forward implementation."
        )

    if skipped_batches:
        print(
            f"[Numerical safety] Skipped {skipped_batches} unstable "
            "training batch(es) this epoch."
        )

    return {
        "total_loss": meters["total"].avg,
        "mse_loss": meters["mse"].avg,
        "ssim_loss": meters["ssim"].avg,
        "gradient_loss": meters["grad"].avg,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
):
    model.eval()
    meters = new_loss_meters()

    # Parameters + BatchNorm running statistics are checked before eval forward.
    _check_model_state_finite(model, where="before validation")

    for batch_idx, batch in enumerate(loader):
        inputs = batch["input"].to(device, non_blocking=True).float()
        targets = batch["target"].to(device, non_blocking=True).float()

        _tensor_finite("validation input", inputs, batch_idx)
        _tensor_finite("validation target", targets, batch_idx)

        # Validation intentionally runs in full FP32 even if AMP is enabled
        # for training. This removes the observed FP16/autocast failure path.
        with autocast(enabled=False):
            reconstruction = model(inputs)

        _tensor_finite("validation model output", reconstruction, batch_idx)

        loss_dict = criterion(reconstruction, targets)

        for key, value in loss_dict.items():
            if not torch.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite validation {key} at batch {batch_idx}: "
                    f"{value.item()}"
                )

        bs = inputs.size(0)
        meters["total"].update(loss_dict["total_loss"].item(), bs)
        meters["mse"].update(loss_dict["mse_loss"].item(), bs)
        meters["ssim"].update(loss_dict["ssim_loss"].item(), bs)
        meters["grad"].update(loss_dict["gradient_loss"].item(), bs)

    return {
        "total_loss": meters["total"].avg,
        "mse_loss": meters["mse"].avg,
        "ssim_loss": meters["ssim"].avg,
        "gradient_loss": meters["grad"].avg,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    best_val_loss: float,
    patience_counter: int,
    cfg: TrainConfig,
    train_video_dirs: List[Path],
    val_video_dirs: List[Path],
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "model_config": asdict(cfg.MODEL),
            "train_video_dirs": [str(p) for p in train_video_dirs],
            "val_video_dirs": [str(p) for p in val_video_dirs],
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_val_loss", float("inf"))),
        int(checkpoint.get("patience_counter", 0)),
        checkpoint,
    )


def initialize_csv(csv_path: Path):
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "epoch", "learning_rate",
                "train_total", "train_mse", "train_ssim", "train_gradient",
                "val_total", "val_mse", "val_ssim", "val_gradient",
                "epoch_time_sec",
            ]
        )


def append_csv(csv_path, epoch, lr, train_stats, val_stats, epoch_time):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                epoch, lr,
                train_stats["total_loss"], train_stats["mse_loss"],
                train_stats["ssim_loss"], train_stats["gradient_loss"],
                val_stats["total_loss"], val_stats["mse_loss"],
                val_stats["ssim_loss"], val_stats["gradient_loss"],
                epoch_time,
            ]
        )


def save_split_manifest(path, train_video_dirs, val_video_dirs, cfg):
    manifest = {
        "official_train_root": str(resolve_path(cfg.OFFICIAL_TRAIN_DIR)),
        "official_test_used_during_training": False,
        "train_ratio": cfg.TRAIN_RATIO,
        "seed": cfg.SEED,
        "training_videos": [str(resolve_path(p)) for p in train_video_dirs],
        "validation_videos": [str(resolve_path(p)) for p in val_video_dirs],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def train():
    cfg = TrainConfig()
    seed_everything(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==============================================")
    print("FALAPS Training")
    print("==============================================")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.CHECKPOINT_DIR / "training_history.csv"
    split_manifest_path = cfg.CHECKPOINT_DIR / "data_split_manifest.json"
    initialize_csv(csv_path)

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        train_video_dirs,
        val_video_dirs,
    ) = build_dataloaders(cfg)

    save_split_manifest(split_manifest_path, train_video_dirs, val_video_dirs, cfg)

    model = FeatureAlignedLViTPixelShuffleAutoencoder(cfg.MODEL).to(device)
    #print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = CompositeReconstructionLoss(
        lambda_mse=cfg.LAMBDA_MSE,
        lambda_ssim=cfg.LAMBDA_SSIM,
        lambda_grad=cfg.LAMBDA_GRAD,
        ssim_window_size=11,
        ssim_sigma=1.5,
        data_range=1.0,
        gradient_loss_type="l1",
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = (
        CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
        if cfg.USE_COSINE_SCHEDULER else None
    )

    use_amp = cfg.USE_AMP and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(f"AMP enabled: {use_amp}")

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    if cfg.RESUME_TRAINING:
        if not cfg.RESUME_PATH.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {cfg.RESUME_PATH}")

        start_epoch, best_val_loss, patience_counter, checkpoint = load_checkpoint(
            cfg.RESUME_PATH,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

        # Ensure exactly the same split is being reused.
        old_train = set(checkpoint.get("train_video_dirs", []))
        old_val = set(checkpoint.get("val_video_dirs", []))
        cur_train = {str(p) for p in train_video_dirs}
        cur_val = {str(p) for p in val_video_dirs}

        if old_train and old_train != cur_train:
            raise RuntimeError("Current training-video split differs from resume checkpoint.")
        if old_val and old_val != cur_val:
            raise RuntimeError("Current validation-video split differs from resume checkpoint.")

    print("\n==============================================")
    print("Training started")
    print("==============================================")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip_norm=cfg.GRAD_CLIP_NORM,
        )

        _check_model_state_finite(model, where="after training epoch / before validation")

        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        improved = val_stats["total_loss"] < (best_val_loss - cfg.MIN_DELTA)

        if improved:
            best_val_loss = val_stats["total_loss"]
            patience_counter = 0
            save_checkpoint(
                cfg.CHECKPOINT_DIR / cfg.BEST_MODEL_NAME,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_loss,
                patience_counter,
                cfg,
                train_video_dirs,
                val_video_dirs,
            )
            marker = " <-- BEST"
        else:
            patience_counter += 1
            marker = ""

        save_checkpoint(
            cfg.CHECKPOINT_DIR / cfg.LAST_MODEL_NAME,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val_loss,
            patience_counter,
            cfg,
            train_video_dirs,
            val_video_dirs,
        )

        append_csv(
            csv_path,
            epoch,
            current_lr,
            train_stats,
            val_stats,
            epoch_time,
        )

        print(
            f"\nEpoch [{epoch:03d}/{cfg.EPOCHS:03d}] "
            f"LR={current_lr:.6e} Time={epoch_time:.1f}s"
        )
        print(
            "  Train | "
            f"Total={train_stats['total_loss']:.6f} | "
            f"MSE={train_stats['mse_loss']:.6f} | "
            f"SSIM={train_stats['ssim_loss']:.6f} | "
            f"Grad={train_stats['gradient_loss']:.6f}"
        )
        print(
            "  Val   | "
            f"Total={val_stats['total_loss']:.6f} | "
            f"MSE={val_stats['mse_loss']:.6f} | "
            f"SSIM={val_stats['ssim_loss']:.6f} | "
            f"Grad={val_stats['gradient_loss']:.6f}{marker}"
        )
        print(f"  Best validation loss = {best_val_loss:.6f}")

        if cfg.EARLY_STOPPING and patience_counter >= cfg.PATIENCE:
            print(
                f"\nEarly stopping: no validation improvement for "
                f"{cfg.PATIENCE} epochs."
            )
            break

    print("\n==============================================")
    print("Training complete")
    print("==============================================")
    print(f"Best model : {cfg.CHECKPOINT_DIR / cfg.BEST_MODEL_NAME}")
    print(f"Last model : {cfg.CHECKPOINT_DIR / cfg.LAST_MODEL_NAME}")
    print(f"History    : {csv_path}")
    print(f"Split log  : {split_manifest_path}")
    print("Official benchmark TEST data was never used during training/validation.")


if __name__ == "__main__":
    train()
