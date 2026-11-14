"""
Reproducible Training Pipeline for Semantic Segmentation Experiment.
Records:
- Environment & library versions
- Architecture configuration
- Seed, resolution, augmentation, optimizer, learning rate schedule, and stopping rule
- Training & validation loss and mIoU per epoch
- Best checkpoint based on validation mIoU
"""

import os
import sys
import json
import time
import random
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.dataset import get_dataloaders, DATA_CONTRACT
from src.models import get_model, count_parameters
from src.metrics import SegmentationMetrics


def set_seed(seed: int = 42):
    """Ensures deterministic execution across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_environment_info() -> Dict[str, Any]:
    """Captures exact execution environment versions for reproducibility."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "None",
        "platform": sys.platform
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metrics: SegmentationMetrics
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    metrics.reset()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        metrics.update(outputs.detach(), masks)

    avg_loss = total_loss / len(loader.dataset)
    results = metrics.compute()
    return avg_loss, results["miou"]


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: SegmentationMetrics
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    metrics.reset()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item() * images.size(0)
            metrics.update(outputs, masks)

    avg_loss = total_loss / len(loader.dataset)
    results = metrics.compute()
    return avg_loss, results


def run_training(
    model_name: str = "baseline",
    data_dir: str = "./data",
    output_dir: str = "./outputs",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    base_channels: int = 32
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Training Model: {model_name.upper()} on {device}")
    print(f"=======================================================")

    save_dir = os.path.join(output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Prepare Data Loaders
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2 if sys.platform != "win32" else 0,  # avoid windows multiprocessing worker overhead
        seed=seed
    )

    # 2. Build Model
    model = get_model(model_name, num_classes=DATA_CONTRACT["num_classes"], base_channels=base_channels)
    model.to(device)
    param_count = count_parameters(model)
    print(f"Total Trainable Parameters: {param_count:,}")

    # 3. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Metrics Tracker
    train_metrics = SegmentationMetrics(DATA_CONTRACT["num_classes"], DATA_CONTRACT["class_names"])
    val_metrics = SegmentationMetrics(DATA_CONTRACT["num_classes"], DATA_CONTRACT["class_names"])

    # 5. Experiment Metadata & Contract Logging
    experiment_record = {
        "model_name": model_name,
        "parameter_count": param_count,
        "seed": seed,
        "resolution": list(DATA_CONTRACT["resolution"]),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "lr_scheduler": "CosineAnnealingLR",
        "stopping_rule": f"Fixed {epochs} epochs with best validation mIoU checkpointing",
        "environment": get_environment_info(),
        "data_contract": DATA_CONTRACT,
        "history": {
            "epoch": [],
            "train_loss": [],
            "train_miou": [],
            "val_loss": [],
            "val_miou": [],
            "lr": []
        }
    }

    best_val_miou = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, device, train_metrics)
        val_loss, val_results = evaluate(model, val_loader, criterion, device, val_metrics)
        val_miou = val_results["miou"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        ep_duration = time.time() - ep_start

        # Record history
        experiment_record["history"]["epoch"].append(epoch)
        experiment_record["history"]["train_loss"].append(round(train_loss, 4))
        experiment_record["history"]["train_miou"].append(round(train_miou, 4))
        experiment_record["history"]["val_loss"].append(round(val_loss, 4))
        experiment_record["history"]["val_miou"].append(round(val_miou, 4))
        experiment_record["history"]["lr"].append(current_lr)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f} (LR: {current_lr:.6f})")

        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "val_results": val_results,
                "experiment_record": experiment_record
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"  --> Saved new best checkpoint (Val mIoU: {best_val_miou:.4f})")

    total_time = time.time() - start_time
    experiment_record["total_training_time_seconds"] = round(total_time, 2)
    experiment_record["best_val_miou"] = round(best_val_miou, 4)

    # Save final log
    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(experiment_record, f, indent=2)

    print(f"\nTraining completed in {total_time/60:.2f} mins. Log saved to {log_path}")
    return experiment_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline or Dilated U-Net.")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "dilated"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
