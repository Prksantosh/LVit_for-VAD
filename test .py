from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.transforms import functional as TF

from config.config import FALAPSConfig
from FALVPSAe import FeatureAlignedLViTPixelShuffleAutoencoder


@dataclass
class TestConfig:
    TEST_DIR: Path = Path("./data/test")
    GT_DIR: Path = Path("./data/test/test_labels")

    CHECKPOINT_PATH: Path = Path("./checkpoints_FALAPS/best_model.pth")
    OUTPUT_DIR: Path = Path("./test_results_FALAPS")

    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    IMAGE_EXTENSIONS: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    )

    EPS: float = 1e-10
    SAVE_EVERY: int = 1
    ERROR_VIS_PERCENTILE: float = 99.0

    MODEL: FALAPSConfig = FALAPSConfig(
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


def ensure_finite(name: str, x: torch.Tensor) -> None:
    if torch.isfinite(x).all():
        return

    finite = x[torch.isfinite(x)]
    if finite.numel() > 0:
        details = (
            f"finite_min={finite.min().item():.6e}, "
            f"finite_max={finite.max().item():.6e}"
        )
    else:
        details = "no finite values"

    raise FloatingPointError(f"{name} contains NaN/Inf; {details}")


def list_video_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Test directory does not exist: {root}")

    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        raise RuntimeError(
            f"No test video folders found under {root}. "
            "Expected one directory per test video."
        )
    return dirs


def list_frames(video_dir: Path, extensions: Sequence[str]) -> List[Path]:
    ext_set = {e.lower() for e in extensions}
    frames = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ext_set
    )
    if not frames:
        raise RuntimeError(f"No image frames found in {video_dir}")
    return frames


def load_frame(path: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    """Load RGB image as [3,H,W] float tensor in [0,1]."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(
            (image_size[1], image_size[0]),
            resample=Image.BILINEAR,
        )
        x = TF.to_tensor(img)
    return x.float().clamp_(0.0, 1.0)


# ============================================================
# Ground truth
# ============================================================

def _load_vector_label_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        y = np.load(path)
    elif path.suffix.lower() == ".txt":
        y = np.loadtxt(path)
    elif path.suffix.lower() == ".csv":
        try:
            y = np.loadtxt(path, delimiter=",")
        except ValueError:
            arr = np.genfromtxt(path, delimiter=",", names=True)
            if arr.dtype.names is None:
                raise
            names = list(arr.dtype.names)
            candidate = next(
                (n for n in names if n.lower() in {"label", "gt", "anomaly", "target"}),
                names[-1],
            )
            y = arr[candidate]
    else:
        raise ValueError(f"Unsupported label file: {path}")

    y = np.asarray(y).reshape(-1)
    return (y > 0).astype(np.uint8)


def _labels_from_mask_folder(mask_dir: Path, n_frames: int) -> np.ndarray:
    masks = sorted(
        p for p in mask_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    if len(masks) != n_frames:
        raise ValueError(
            f"Ground-truth mask count mismatch for {mask_dir.name}: "
            f"{len(masks)} masks vs {n_frames} test frames."
        )

    labels = []
    for p in masks:
        with Image.open(p) as img:
            arr = np.asarray(img)
        labels.append(int(np.any(arr > 0)))

    return np.asarray(labels, dtype=np.uint8)


def load_video_labels(gt_root: Path, video_name: str, n_frames: int) -> np.ndarray:
    for ext in (".npy", ".txt", ".csv"):
        candidate = gt_root / f"{video_name}{ext}"
        if candidate.exists():
            labels = _load_vector_label_file(candidate)
            if len(labels) != n_frames:
                raise ValueError(
                    f"Label count mismatch for {video_name}: "
                    f"{len(labels)} labels vs {n_frames} frames."
                )
            return labels

    mask_dir = gt_root / video_name
    if mask_dir.is_dir():
        return _labels_from_mask_folder(mask_dir, n_frames)

    raise FileNotFoundError(
        f"No ground truth found for '{video_name}'. Expected one of:\n"
        f"  {gt_root / (video_name + '.npy')}\n"
        f"  {gt_root / (video_name + '.txt')}\n"
        f"  {gt_root / (video_name + '.csv')}\n"
        f"  {gt_root / video_name}/<mask frames>"
    )


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def frame_mse_psnr(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    eps: float = 1e-10,
) -> Tuple[float, float]:
    """
    Inputs are normalized [B,C,H,W] tensors in [0,1].

    PSNR_t = 10 * log10(1 / (MSE_t + eps))
    """
    target = target.float().clamp(0.0, 1.0)
    reconstruction = reconstruction.float().clamp(0.0, 1.0)

    mse = torch.mean((target - reconstruction) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))

    return float(mse[0].item()), float(psnr[0].item())


def psnr_to_normalized_anomaly_score(
    psnr: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-video normalization:
        normalized_psnr = (PSNR - min) / (max - min + eps)
        anomaly_score   = 1 - normalized_psnr
    """
    psnr = np.asarray(psnr, dtype=np.float64)
    p_min = float(np.min(psnr))
    p_max = float(np.max(psnr))

    if abs(p_max - p_min) <= eps:
        return np.zeros_like(psnr, dtype=np.float64)

    normalized_psnr = (psnr - p_min) / (p_max - p_min + eps)
    return np.clip(1.0 - normalized_psnr, 0.0, 1.0)


def calculate_auc_eer(
    labels: np.ndarray,
    anomaly_scores: np.ndarray,
) -> Tuple[float, float, float]:
    labels = np.asarray(labels).astype(np.uint8)
    scores = np.asarray(anomaly_scores, dtype=np.float64)

    if np.unique(labels).size < 2:
        raise ValueError(
            "ROC-AUC/EER require both normal (0) and anomalous (1) frames."
        )

    auc = float(roc_auc_score(labels, scores))

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])

    return auc, eer, threshold


# ============================================================
# Visualization
# ============================================================

def tensor_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    x = x.permute(1, 2, 0).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def build_error_map(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> np.ndarray:
    target = target.detach().float().cpu().clamp(0.0, 1.0)
    reconstruction = reconstruction.detach().float().cpu().clamp(0.0, 1.0)

    if target.ndim == 4:
        target = target[0]
    if reconstruction.ndim == 4:
        reconstruction = reconstruction[0]

    error = torch.mean(torch.abs(target - reconstruction), dim=0)
    return error.numpy().astype(np.float32)


def normalize_error_for_display(
    error_map: np.ndarray,
    percentile: float = 99.0,
) -> np.ndarray:
    upper = float(np.percentile(error_map, percentile))
    if upper <= 1e-12:
        upper = float(np.max(error_map))
    if upper <= 1e-12:
        return np.zeros_like(error_map, dtype=np.float32)
    return np.clip(error_map / upper, 0.0, 1.0).astype(np.float32)


def heatmap_rgb(norm_error: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    rgba = cmap(np.clip(norm_error, 0.0, 1.0))
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_gray(path: Path, image01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = (np.clip(image01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(x, mode="L").save(path)


def save_frame_visualizations(
    video_out: Path,
    stem: str,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    error_map: np.ndarray,
    error_vis_percentile: float,
) -> None:
    target_rgb = tensor_to_rgb_uint8(target)
    recon_rgb = tensor_to_rgb_uint8(reconstruction)

    error_vis = normalize_error_for_display(
        error_map,
        percentile=error_vis_percentile,
    )
    heat_rgb = heatmap_rgb(error_vis)

    overlay = (
        0.55 * target_rgb.astype(np.float32)
        + 0.45 * heat_rgb.astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    save_rgb(video_out / "target" / f"{stem}.png", target_rgb)
    save_rgb(video_out / "reconstruction" / f"{stem}.png", recon_rgb)
    save_gray(video_out / "error" / f"{stem}.png", error_vis)
    save_rgb(video_out / "heatmap" / f"{stem}.png", heat_rgb)
    save_rgb(video_out / "overlay" / f"{stem}.png", overlay)


def save_anomaly_plot(
    out_path: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    video_name: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(scores))
    plt.figure(figsize=(14, 4.8))
    plt.plot(x, scores, linewidth=1.5, label="Normalized anomaly score")
    plt.fill_between(
        x,
        0,
        labels.astype(float),
        alpha=0.20,
        label="Ground truth anomaly",
    )
    plt.ylim(-0.03, 1.03)
    plt.xlabel("Frame index")
    plt.ylabel("Anomaly score")
    plt.title(f"{video_name}: normalized PSNR-based anomaly score")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_combined_panel(
    out_path: Path,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    error_map: np.ndarray,
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    frame_idx: int,
    psnr: float,
    mse: float,
    percentile: float,
) -> None:
    target_rgb = tensor_to_rgb_uint8(target)
    recon_rgb = tensor_to_rgb_uint8(reconstruction)
    error_vis = normalize_error_for_display(error_map, percentile)
    heat_rgb = heatmap_rgb(error_vis)
    overlay = np.clip(
        0.55 * target_rgb.astype(np.float32)
        + 0.45 * heat_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(target_rgb)
    ax1.set_title("Target")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(recon_rgb)
    ax2.set_title(f"Reconstruction\nPSNR={psnr:.3f} dB")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(error_vis, cmap="gray", vmin=0, vmax=1)
    ax3.set_title(f"Error map\nMSE={mse:.6e}")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(heat_rgb)
    ax4.set_title("Error heatmap")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(overlay)
    ax5.set_title("Heatmap overlay")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    x = np.arange(len(anomaly_scores))
    ax6.plot(x, anomaly_scores, linewidth=1.3, label="Anomaly score")
    ax6.fill_between(
        x,
        0,
        labels.astype(float),
        alpha=0.20,
        label="GT",
    )
    ax6.axvline(frame_idx, linestyle="--", linewidth=1.2)
    ax6.scatter(
        [frame_idx],
        [anomaly_scores[frame_idx]],
        s=35,
        zorder=5,
    )
    ax6.set_ylim(-0.03, 1.03)
    ax6.set_xlabel("Frame index")
    ax6.set_ylabel("Normalized score")
    ax6.set_title(
        f"Score={anomaly_scores[frame_idx]:.4f}, "
        f"GT={int(labels[frame_idx])}"
    )
    ax6.grid(alpha=0.25)
    ax6.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ============================================================
# Model/checkpoint
# ============================================================

def load_model(cfg: TestConfig, device: torch.device) -> nn.Module:
    model = FeatureAlignedLViTPixelShuffleAutoencoder(cfg.MODEL).to(device)

    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    cleaned = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    model.load_state_dict(cleaned, strict=True)
    model.eval()

    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise FloatingPointError(
                f"Checkpoint contains NaN/Inf parameter: {name}"
            )

    for name, b in model.named_buffers():
        if torch.is_floating_point(b) and not torch.isfinite(b).all():
            raise FloatingPointError(
                f"Checkpoint contains NaN/Inf buffer: {name}"
            )

    return model


# ============================================================
# CSV
# ============================================================

def save_video_csv(
    path: Path,
    frame_paths: Sequence[Path],
    labels: np.ndarray,
    mse_values: np.ndarray,
    psnr_values: np.ndarray,
    anomaly_scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_index", "frame_name", "label", "mse", "psnr_db", "anomaly_score"]
        )
        for i, frame_path in enumerate(frame_paths):
            writer.writerow(
                [
                    i,
                    frame_path.name,
                    int(labels[i]),
                    float(mse_values[i]),
                    float(psnr_values[i]),
                    float(anomaly_scores[i]),
                ]
            )


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate_video(
    model: nn.Module,
    video_dir: Path,
    cfg: TestConfig,
    device: torch.device,
) -> Dict:
    frames = list_frames(video_dir, cfg.IMAGE_EXTENSIONS)
    labels = load_video_labels(cfg.GT_DIR, video_dir.name, len(frames))

    mse_values: List[float] = []
    psnr_values: List[float] = []

    saved_targets: List[torch.Tensor] = []
    saved_recons: List[torch.Tensor] = []
    saved_errors: List[np.ndarray] = []

    video_out = cfg.OUTPUT_DIR / video_dir.name

    for idx, frame_path in enumerate(frames):
        target = load_frame(frame_path, cfg.IMAGE_SIZE).unsqueeze(0).to(device)
        ensure_finite(f"{video_dir.name}/{frame_path.name} input", target)

        # FP32 inference to avoid the FP16 instability seen during validation.
        reconstruction = model(target.float())
        ensure_finite(
            f"{video_dir.name}/{frame_path.name} reconstruction",
            reconstruction,
        )

        if reconstruction.shape != target.shape:
            raise ValueError(
                f"Model output shape {tuple(reconstruction.shape)} does not "
                f"match target shape {tuple(target.shape)} for {frame_path}."
            )

        # PSNR must be computed from normalized image intensities in [0,1].
        target_n = target.float().clamp(0.0, 1.0)
        recon_n = reconstruction.float().clamp(0.0, 1.0)

        mse, psnr = frame_mse_psnr(
            target=target_n,
            reconstruction=recon_n,
            eps=cfg.EPS,
        )

        error_map = build_error_map(target_n, recon_n)

        mse_values.append(mse)
        psnr_values.append(psnr)
        saved_targets.append(target_n.detach().cpu())
        saved_recons.append(recon_n.detach().cpu())
        saved_errors.append(error_map)

        if idx % cfg.SAVE_EVERY == 0:
            save_frame_visualizations(
                video_out=video_out,
                stem=f"{idx:06d}_{frame_path.stem}",
                target=target_n,
                reconstruction=recon_n,
                error_map=error_map,
                error_vis_percentile=cfg.ERROR_VIS_PERCENTILE,
            )

    mse_arr = np.asarray(mse_values, dtype=np.float64)
    psnr_arr = np.asarray(psnr_values, dtype=np.float64)

    anomaly_scores = psnr_to_normalized_anomaly_score(
        psnr_arr,
        eps=cfg.EPS,
    )

    save_anomaly_plot(
        video_out / "anomaly_score_plot.png",
        anomaly_scores,
        labels,
        video_dir.name,
    )

    save_video_csv(
        video_out / "frame_metrics.csv",
        frames,
        labels,
        mse_arr,
        psnr_arr,
        anomaly_scores,
    )

    for idx, frame_path in enumerate(frames):
        if idx % cfg.SAVE_EVERY != 0:
            continue

        save_combined_panel(
            out_path=video_out / "combined" / f"{idx:06d}_{frame_path.stem}.png",
            target=saved_targets[idx],
            reconstruction=saved_recons[idx],
            error_map=saved_errors[idx],
            anomaly_scores=anomaly_scores,
            labels=labels,
            frame_idx=idx,
            psnr=float(psnr_arr[idx]),
            mse=float(mse_arr[idx]),
            percentile=cfg.ERROR_VIS_PERCENTILE,
        )

    return {
        "video": video_dir.name,
        "num_frames": len(frames),
        "labels": labels,
        "mse": mse_arr,
        "psnr": psnr_arr,
        "scores": anomaly_scores,
        "average_psnr": float(np.mean(psnr_arr)),
    }


def evaluate() -> None:
    cfg = TestConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n============================================================")
    print("FALAPS PSNR-based frame-level evaluation")
    print("============================================================")
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint      : {cfg.CHECKPOINT_PATH.resolve()}")
    print(f"Test root       : {cfg.TEST_DIR.resolve()}")
    print(f"Ground truth    : {cfg.GT_DIR.resolve()}")
    print(f"Output          : {cfg.OUTPUT_DIR.resolve()}")
    print("Inference dtype : FP32")
    print("============================================================\n")

    if not cfg.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.CHECKPOINT_PATH}")

    if not cfg.GT_DIR.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {cfg.GT_DIR}")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(cfg, device)
    video_dirs = list_video_dirs(cfg.TEST_DIR)

    results = []

    for i, video_dir in enumerate(video_dirs, start=1):
        print(
            f"[{i:03d}/{len(video_dirs):03d}] "
            f"Evaluating {video_dir.name} ..."
        )

        result = evaluate_video(
            model=model,
            video_dir=video_dir,
            cfg=cfg,
            device=device,
        )
        results.append(result)

        try:
            v_auc, v_eer, _ = calculate_auc_eer(
                result["labels"],
                result["scores"],
            )
            extra = f"AUC={v_auc:.6f} | EER={v_eer:.6f}"
        except ValueError:
            extra = "AUC/EER=N/A (single-class video)"

        print(
            f"    frames={result['num_frames']} | "
            f"Avg PSNR={result['average_psnr']:.4f} dB | "
            f"{extra}"
        )

    all_labels = np.concatenate([r["labels"] for r in results])
    all_scores = np.concatenate([r["scores"] for r in results])
    all_psnr = np.concatenate([r["psnr"] for r in results])
    all_mse = np.concatenate([r["mse"] for r in results])

    average_psnr = float(np.mean(all_psnr))
    average_mse = float(np.mean(all_mse))
    auc, eer, eer_threshold = calculate_auc_eer(all_labels, all_scores)

    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    plt.figure(figsize=(6.5, 6.0))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Frame-level ROC curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / "roc_curve.png", dpi=180)
    plt.close()

    summary = {
        "checkpoint": str(cfg.CHECKPOINT_PATH.resolve()),
        "num_videos": len(results),
        "num_frames": int(len(all_labels)),
        "normal_frames": int(np.sum(all_labels == 0)),
        "anomalous_frames": int(np.sum(all_labels == 1)),
        "average_mse": average_mse,
        "average_psnr_db": average_psnr,
        "frame_auc": auc,
        "frame_eer": eer,
        "eer_threshold": eer_threshold,
        "anomaly_score_definition": (
            "per-video 1 - minmax(PSNR), where "
            "PSNR=10*log10(1/(MSE+eps)) on [0,1] frames"
        ),
    }

    with (cfg.OUTPUT_DIR / "evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    with (cfg.OUTPUT_DIR / "video_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frames", "average_psnr_db"])
        for r in results:
            writer.writerow(
                [r["video"], r["num_frames"], r["average_psnr"]]
            )

    print("\n============================================================")
    print("FINAL FRAME-LEVEL RESULTS")
    print("============================================================")
    print(f"Videos             : {len(results)}")
    print(f"Frames             : {len(all_labels)}")
    print(f"Normal frames      : {np.sum(all_labels == 0)}")
    print(f"Anomalous frames   : {np.sum(all_labels == 1)}")
    print(f"Average MSE        : {average_mse:.8f}")
    print(f"Average PSNR       : {average_psnr:.6f} dB")
    print(f"Frame-level AUC    : {auc:.6f}")
    print(f"Frame-level EER    : {eer:.6f}")
    print(f"EER threshold      : {eer_threshold:.6f}")
    print("============================================================")
    print(f"Results saved to   : {cfg.OUTPUT_DIR.resolve()}")
    print("============================================================")


if __name__ == "__main__":
    evaluate()
