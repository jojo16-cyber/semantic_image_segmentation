"""
Evaluation Metrics and Latency Measurement Suite.
Computes:
1. Per-class IoU (Intersection-over-Union)
2. Mean IoU (mIoU)
3. Overall Pixel Accuracy
4. Inference Latency (ms) and Throughput (FPS) under controlled warmup conditions.
"""

import time
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np


class SegmentationMetrics:
    """
    Accumulates confusion matrix across batches to compute exact per-class IoU,
    mIoU, and overall pixel accuracy without batch-averaging bias.
    """
    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: Tensor of predicted class indices (B, H, W) or logits (B, C, H, W)
        targets: Ground truth class indices (B, H, W)
        """
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        preds_flat = preds.view(-1).cpu().numpy()
        targets_flat = targets.view(-1).cpu().numpy()

        # Filter out invalid / ignore index if any
        mask = (targets_flat >= 0) & (targets_flat < self.num_classes)
        preds_valid = preds_flat[mask]
        targets_valid = targets_flat[mask]

        # Compute 2D histogram (confusion matrix)
        bins = self.num_classes
        hist, _, _ = np.histogram2d(
            targets_valid,
            preds_valid,
            bins=bins,
            range=[[0, bins], [0, bins]]
        )
        self.confusion_matrix += hist.astype(np.int64)

    def compute(self) -> Dict[str, Any]:
        """
        Computes per-class IoU and mean IoU.
        IoU = TP / (TP + FP + FN) = diag / (row_sum + col_sum - diag)
        """
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        denominator = tp + fp + fn
        # Avoid division by zero
        ious = np.where(denominator > 0, tp / denominator, 0.0)
        miou = float(np.mean(ious))

        pixel_accuracy = float(np.sum(tp) / (np.sum(cm) + 1e-12))

        per_class = {}
        for idx, (name, iou) in enumerate(zip(self.class_names, ious)):
            per_class[name] = float(iou)

        return {
            "per_class_iou": per_class,
            "miou": miou,
            "pixel_accuracy": pixel_accuracy,
            "confusion_matrix": cm.tolist()
        }


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    warmup_runs: int = 30,
    timed_runs: int = 100
) -> Dict[str, float]:
    """
    Measures inference latency under standardized conditions:
    - Input batch size: 1 (single-frame real-time simulation)
    - Warm-up iterations: 30
    - Timed iterations: 100
    - Synchronized GPU execution with CUDA events if available.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape, device=device)

    is_cuda = device.type == "cuda"

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        if is_cuda:
            torch.cuda.synchronize()

        # Timed runs
        timings = []
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(timed_runs):
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))  # returns milliseconds
        else:
            for _ in range(timed_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)  # ms

    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "mean_latency_ms": mean_ms,
        "std_latency_ms": std_ms,
        "fps": fps,
        "device": str(device),
        "input_shape": list(input_shape),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs
    }
