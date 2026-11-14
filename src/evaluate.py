"""
Exact Evaluation Script for Reproducible Segmentation Experiment.
- Computes Test Set Per-Class IoU, mIoU, and Pixel Accuracy
- Measures Parameter Count and Inference Latency (ms) under standardized conditions
- Generates side-by-side qualitative comparisons (Input, Ground Truth, Baseline, Dilated)
- Identifies and visualizes explicit Failure Cases
- Plots Training and Validation Loss/mIoU curves
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders, DATA_CONTRACT
from src.models import get_model, count_parameters
from src.metrics import SegmentationMetrics, measure_inference_latency


# Color map for trimap visualization:
# 0: Foreground (Pet) -> Red / Coral [220, 50, 50]
# 1: Background -> Forest Green [50, 160, 50]
# 2: Boundary -> Yellow / Amber [240, 200, 30]
COLOR_PALETTE = np.array([
    [220, 50, 50],    # Class 0: Pet
    [50, 160, 50],    # Class 1: Background
    [240, 200, 30]    # Class 2: Boundary
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Converts a (H, W) label mask with values in {0, 1, 2} to an RGB image."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id in range(3):
        colored[mask == c_id] = COLOR_PALETTE[c_id]
    return colored


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Inverts ImageNet normalization for visualization."""
    mean = np.array(DATA_CONTRACT["normalization"]["mean"])
    std = np.array(DATA_CONTRACT["normalization"]["std"])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def evaluate_model_on_test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    model.eval()
    metrics = SegmentationMetrics(DATA_CONTRACT["num_classes"], DATA_CONTRACT["class_names"])
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            metrics.update(outputs, masks)

    return metrics.compute()


def run_full_evaluation(
    output_dir: str = "./outputs",
    data_dir: str = "./data",
    seed: int = 42,
    base_channels: int = 32
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Full Evaluation on: {device}")

    # 1. Load Data
    _, _, test_loader, meta = get_dataloaders(
        data_dir=data_dir,
        batch_size=16,
        num_workers=0,
        seed=seed
    )

    # 2. Instantiate and Load Models
    baseline_model = get_model("baseline", num_classes=3, base_channels=base_channels).to(device)
    dilated_model = get_model("dilated", num_classes=3, base_channels=base_channels).to(device)

    baseline_ckpt_path = os.path.join(output_dir, "baseline", "best_model.pth")
    dilated_ckpt_path = os.path.join(output_dir, "dilated", "best_model.pth")

    if not os.path.exists(baseline_ckpt_path) or not os.path.exists(dilated_ckpt_path):
        raise FileNotFoundError(
            f"Checkpoints not found! Ensure both baseline ({baseline_ckpt_path}) "
            f"and dilated ({dilated_ckpt_path}) have been trained."
        )

    baseline_ckpt = torch.load(baseline_ckpt_path, map_location=device)
    dilated_ckpt = torch.load(dilated_ckpt_path, map_location=device)

    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])
    dilated_model.load_state_dict(dilated_ckpt["model_state_dict"])

    # 3. Model Parameters
    baseline_params = count_parameters(baseline_model)
    dilated_params = count_parameters(dilated_model)

    # 4. Latency Measurements (Standardized: batch_size=1, warmup=30, runs=100)
    print("\nMeasuring inference latency...")
    baseline_latency = measure_inference_latency(baseline_model, device)
    dilated_latency = measure_inference_latency(dilated_model, device)

    # 5. Evaluate on Test Split
    print("\nEvaluating on Test Set (15% split)...")
    baseline_test_results = evaluate_model_on_test(baseline_model, test_loader, device)
    dilated_test_results = evaluate_model_on_test(dilated_model, test_loader, device)

    # 6. Format Comparison Table
    results_summary = {
        "Baseline U-Net": {
            "parameter_count": baseline_params,
            "mean_latency_ms": round(baseline_latency["mean_latency_ms"], 2),
            "std_latency_ms": round(baseline_latency["std_latency_ms"], 2),
            "fps": round(baseline_latency["fps"], 1),
            "test_miou": round(baseline_test_results["miou"], 4),
            "test_pixel_acc": round(baseline_test_results["pixel_accuracy"], 4),
            "per_class_iou": {k: round(v, 4) for k, v in baseline_test_results["per_class_iou"].items()}
        },
        "Dilated U-Net (Ablation)": {
            "parameter_count": dilated_params,
            "mean_latency_ms": round(dilated_latency["mean_latency_ms"], 2),
            "std_latency_ms": round(dilated_latency["std_latency_ms"], 2),
            "fps": round(dilated_latency["fps"], 1),
            "test_miou": round(dilated_test_results["miou"], 4),
            "test_pixel_acc": round(dilated_test_results["pixel_accuracy"], 4),
            "per_class_iou": {k: round(v, 4) for k, v in dilated_test_results["per_class_iou"].items()}
        }
    }

    print("\n" + "=" * 78)
    print("                REPRODUCIBLE SEGMENTATION EVALUATION RESULTS")
    print("=" * 78)
    print(f"{'Metric':<28} | {'Baseline U-Net':<22} | {'Dilated U-Net (Ours)':<22}")
    print("-" * 78)
    print(f"{'Parameters':<28} | {baseline_params:<22,d} | {dilated_params:<22,d}")
    print(f"{'Latency (ms/img)':<28} | {baseline_latency['mean_latency_ms']:.2f} ± {baseline_latency['std_latency_ms']:.2f} ms     | {dilated_latency['mean_latency_ms']:.2f} ± {dilated_latency['std_latency_ms']:.2f} ms")
    print(f"{'Throughput (FPS)':<28} | {baseline_latency['fps']:.1f} FPS               | {dilated_latency['fps']:.1f} FPS")
    print(f"{'Test Mean IoU (mIoU)':<28} | {baseline_test_results['miou']:.4f}                 | {dilated_test_results['miou']:.4f}")
    print(f"{'Test Pixel Accuracy':<28} | {baseline_test_results['pixel_accuracy']:.4f}                 | {dilated_test_results['pixel_accuracy']:.4f}")
    print("-" * 78)
    for c_name in DATA_CONTRACT["class_names"]:
        b_iou = baseline_test_results["per_class_iou"][c_name]
        d_iou = dilated_test_results["per_class_iou"][c_name]
        print(f"{c_name + ' IoU':<28} | {b_iou:.4f}                 | {d_iou:.4f}")
    print("=" * 78)

    # 7. Generate Visual Comparisons & Identify Failures
    sample_images = []
    sample_masks = []
    baseline_preds = []
    dilated_preds = []
    ious_per_sample = []

    baseline_model.eval()
    dilated_model.eval()

    with torch.no_grad():
        for img_batch, mask_batch in test_loader:
            img_b = img_batch.to(device)
            out_b = baseline_model(img_b)
            out_d = dilated_model(img_b)

            pred_b = torch.argmax(out_b, dim=1).cpu().numpy()
            pred_d = torch.argmax(out_d, dim=1).cpu().numpy()
            masks_np = mask_batch.numpy()

            for i in range(len(img_batch)):
                sample_images.append(img_batch[i])
                sample_masks.append(masks_np[i])
                baseline_preds.append(pred_b[i])
                dilated_preds.append(pred_d[i])

                # Compute sample-level IoU to find failure cases
                inter = np.sum((pred_b[i] == masks_np[i]) & (masks_np[i] != 1))  # non-background overlap
                union = np.sum((pred_b[i] != 1) | (masks_np[i] != 1)) + 1e-6
                ious_per_sample.append(inter / union)

            if len(sample_images) >= 50:
                break

    # 7a. Plot Qualitative Comparison (Successes)
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    cols = ["Input Image", "Ground Truth Mask", "Baseline U-Net", "Dilated U-Net"]
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=14, fontweight="bold", pad=10)

    selected_indices = [0, 2, 4, 6]
    for row_idx, sample_i in enumerate(selected_indices):
        raw_img = denormalize_image(sample_images[sample_i])
        gt_mask = colorize_mask(sample_masks[sample_i])
        p_base = colorize_mask(baseline_preds[sample_i])
        p_dila = colorize_mask(dilated_preds[sample_i])

        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 1].imshow(gt_mask)
        axes[row_idx, 2].imshow(p_base)
        axes[row_idx, 3].imshow(p_dila)

        for col_idx in range(4):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    comparison_img_path = os.path.join(output_dir, "qualitative_comparisons.png")
    plt.savefig(comparison_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative predictions to: {comparison_img_path}")

    # 7b. Plot Explicit Failure Cases (samples with lowest IoU)
    failure_indices = np.argsort(ious_per_sample)[:4]
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=14, fontweight="bold", pad=10)

    for row_idx, sample_i in enumerate(failure_indices):
        raw_img = denormalize_image(sample_images[sample_i])
        gt_mask = colorize_mask(sample_masks[sample_i])
        p_base = colorize_mask(baseline_preds[sample_i])
        p_dila = colorize_mask(dilated_preds[sample_i])

        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 1].imshow(gt_mask)
        axes[row_idx, 2].imshow(p_base)
        axes[row_idx, 3].imshow(p_dila)

        for col_idx in range(4):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    failure_img_path = os.path.join(output_dir, "failure_cases.png")
    plt.savefig(failure_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved failure case analysis to: {failure_img_path}")

    # 8. Plot Training Curves
    b_log_path = os.path.join(output_dir, "baseline", "training_log.json")
    d_log_path = os.path.join(output_dir, "dilated", "training_log.json")

    if os.path.exists(b_log_path) and os.path.exists(d_log_path):
        with open(b_log_path) as f:
            b_log = json.load(f)["history"]
        with open(d_log_path) as f:
            d_log = json.load(f)["history"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = b_log["epoch"]

        # Loss Curve
        ax1.plot(epochs, b_log["train_loss"], "b--", label="Baseline Train Loss")
        ax1.plot(epochs, b_log["val_loss"], "b-", label="Baseline Val Loss")
        ax1.plot(epochs, d_log["train_loss"], "r--", label="Dilated Train Loss")
        ax1.plot(epochs, d_log["val_loss"], "r-", label="Dilated Val Loss")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
        ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

        # mIoU Curve
        ax2.plot(epochs, b_log["train_miou"], "b--", label="Baseline Train mIoU")
        ax2.plot(epochs, b_log["val_miou"], "b-", label="Baseline Val mIoU")
        ax2.plot(epochs, d_log["train_miou"], "r--", label="Dilated Train mIoU")
        ax2.plot(epochs, d_log["val_miou"], "r-", label="Dilated Val mIoU")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Mean IoU (mIoU)", fontsize=12)
        ax2.set_title("Training & Validation mIoU", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        curve_img_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(curve_img_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved training curves to: {curve_img_path}")

    # Save complete JSON summary
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved evaluation JSON to: {summary_path}")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs Dilated U-Net.")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_full_evaluation(output_dir=args.output_dir, data_dir=args.data_dir, seed=args.seed)
