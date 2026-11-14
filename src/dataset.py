"""
Dataset Contract and Data Pipeline for Reproducible Semantic Segmentation Experiment.

================================================================================
DATA CONTRACT SPECIFICATION
================================================================================
1. Dataset: Oxford-IIIT Pet Dataset (Parkhi et al., 2012)
   - Domain: 37 breeds of dogs and cats.
   - Ground Truth: Pixel-level Trimap annotations.

2. Semantic Class IDs & Semantics:
   - Raw trimap values in Oxford-IIIT Pet:
       1 = Foreground (Pet)
       2 = Background
       3 = Boundary / Outline Contour
   - Canonical 0-indexed remapped labels for Cross-Entropy Loss:
       Class 0: Foreground (Pet)
       Class 1: Background
       Class 2: Boundary / Outline
   - Total Classes (C): 3
   - Class Names: ["Foreground (Pet)", "Background", "Boundary"]

3. Ignored Labels:
   - None (ignore_index = -100). All three classes (including fine boundary)
     are evaluated to assess receptive-field and edge preservation effects.

4. Splits & Reproducibility:
   - Random Seed: 42
   - Partition:
       * Training Set: 70%
       * Validation Set: 15%
       * Test Set: 15%
   - Split Method: Stratified/deterministic random split with fixed generator.

5. Image / Mask Transforms & Preprocessing:
   - Target Spatial Resolution: 256 x 256 pixels.
   - Image Preprocessing:
       * Resized to (256, 256) using BILINEAR interpolation.
       * Converted to FloatTensor in range [0.0, 1.0].
       * Normalized with ImageNet statistics:
         Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225].
   - Mask Preprocessing:
       * Resized to (256, 256) using NEAREST-NEIGHBOR interpolation
         (crucial: prevents fractional label contamination).
       * Remapped from {1, 2, 3} -> {0, 1, 2} LongTensor of shape (H, W).
   - Training Augmentations (Synchronized):
       * Random Horizontal Flip (p = 0.5) applied identically to image and mask.
       * Validation and Test: Deterministic resizing and normalization only.
================================================================================
"""

import os
import random
from typing import Tuple, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


DATA_CONTRACT: Dict[str, Any] = {
    "dataset_name": "Oxford-IIIT Pet",
    "num_classes": 3,
    "class_names": ["Foreground (Pet)", "Background", "Boundary"],
    "class_ids": {0: "Foreground (Pet)", 1: "Background", 2: "Boundary"},
    "raw_to_mapped": {1: 0, 2: 1, 3: 2},
    "ignored_label": None,
    "resolution": (256, 256),
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "seed": 42
}


class SegmentationTransform:
    """
    Synchronized transformation pipeline ensuring joint spatial transformations
    between the RGB image and its corresponding segmentation mask.
    """
    def __init__(self, size: Tuple[int, int] = (256, 256), is_train: bool = True):
        self.size = size
        self.is_train = is_train
        self.normalize = transforms.Normalize(
            mean=DATA_CONTRACT["normalization"]["mean"],
            std=DATA_CONTRACT["normalization"]["std"]
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Resize image with Bilinear, mask with Nearest Neighbor
        image = TF.resize(image, self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)

        # 2. Synchronized data augmentation (Horizontal Flip) for training
        if self.is_train and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 3. Convert image to Tensor & normalize
        image = TF.to_tensor(image)  # Converts [0, 255] PIL Image to FloatTensor [0.0, 1.0]
        image = self.normalize(image)

        # 4. Convert mask to numpy and remap classes: {1, 2, 3} -> {0, 1, 2}
        mask_np = np.array(mask, dtype=np.int64)
        # Oxford trimap: 1=foreground, 2=background, 3=boundary
        # Remap: 1 -> 0, 2 -> 1, 3 -> 2
        remapped_mask = np.zeros_like(mask_np)
        remapped_mask[mask_np == 1] = 0
        remapped_mask[mask_np == 2] = 1
        remapped_mask[mask_np == 3] = 2

        mask_tensor = torch.from_numpy(remapped_mask).long()
        return image, mask_tensor


class OxfordPetSegmentationDataset(Dataset):
    """
    Wrapper around torchvision OxfordIIITPet applying the strict Data Contract.
    """
    def __init__(self, root: str, split: str = "trainval", is_train: bool = True, download: bool = True):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download
        )
        self.transform = SegmentationTransform(size=DATA_CONTRACT["resolution"], is_train=is_train)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[idx]
        image = image.convert("RGB")
        return self.transform(image, mask)


class TransformedSubset(Dataset):
    """
    Helper to apply different transforms (e.g. train augment vs eval deterministic)
    to subsets created via random_split.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, mask = self.subset.dataset.dataset[self.subset.indices[idx]]
        image = image.convert("RGB")
        return self.transform(image, mask)


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Prepares reproducible Train, Validation, and Test DataLoaders matching the Data Contract.
    Combines trainval and test to partition strictly into 70% Train, 15% Val, 15% Test.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Load both partitions to create the complete fixed dataset
    full_trainval = OxfordPetSegmentationDataset(root=data_dir, split="trainval", is_train=False, download=True)
    full_test = OxfordPetSegmentationDataset(root=data_dir, split="test", is_train=False, download=True)

    # Combine into a single indexable dataset
    combined_dataset = torch.utils.data.ConcatDataset([full_trainval, full_test])
    total_samples = len(combined_dataset)

    train_size = int(DATA_CONTRACT["split_ratios"]["train"] * total_samples)
    val_size = int(DATA_CONTRACT["split_ratios"]["val"] * total_samples)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        combined_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Wrap each split with its appropriate transform
    train_transform = SegmentationTransform(size=DATA_CONTRACT["resolution"], is_train=True)
    eval_transform = SegmentationTransform(size=DATA_CONTRACT["resolution"], is_train=False)

    train_data = _SubsetWithTransform(train_sub, train_transform)
    val_data = _SubsetWithTransform(val_sub, eval_transform)
    test_data = _SubsetWithTransform(test_sub, eval_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    meta = {
        "total_samples": total_samples,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "data_contract": DATA_CONTRACT
    }
    return train_loader, val_loader, test_loader, meta


class _SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Resolve through ConcatDataset and Subset
        actual_idx = self.subset.indices[idx]
        concat_dataset = self.subset.dataset
        # concat_dataset has cumulative_sizes
        dataset_idx = 0
        while actual_idx >= concat_dataset.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        if dataset_idx == 0:
            sample_idx = actual_idx
        else:
            sample_idx = actual_idx - concat_dataset.cumulative_sizes[dataset_idx - 1]

        underlying_pet_ds = concat_dataset.datasets[dataset_idx].dataset
        image, mask = underlying_pet_ds[sample_idx]
        image = image.convert("RGB")
        return self.transform(image, mask)
