"""
Unit and integration tests for real Chest X-ray dataset loader and validation pipeline.
Verifies:
1. Patient-level split isolation (strictly zero patient leakage).
2. Ground-truth bounding box rasterization into 2D ROI masks.
3. PyTorch Dataset and DataLoader tensor dimensions, types, and normalizations.
4. Dataset manifest and split consistency.
"""

import os
import json
import csv
import pytest
import numpy as np
import torch
from src.dataset_real import (
    RealCXRDataset,
    rasterize_bounding_boxes,
    validate_dataset_pipeline,
    get_real_cxr_dataloaders
)


def test_patient_split_leakage_zero():
    """Ensure zero patient leakage across train, val, and test splits."""
    val_res = validate_dataset_pipeline("data/real_cxr")
    assert val_res["valid"] is True
    assert val_res["leakage_train_val"] == 0, "Patient leakage between train and val!"
    assert val_res["leakage_train_test"] == 0, "Patient leakage between train and test!"
    assert val_res["leakage_val_test"] == 0, "Patient leakage between val and test!"
    assert val_res["corrupt_images"] == 0
    assert val_res["images_train"] > 0
    assert val_res["images_val"] > 0
    assert val_res["images_test"] > 0


def test_bounding_box_rasterization():
    """Verify that COCO format [x, y, w, h] correctly scales and rasterizes to target size."""
    boxes = [
        {"bbox": [100, 100, 200, 200], "category_id": 1, "finding": ["COVID-19"]}
    ]
    orig_w, orig_h = 1000, 1000
    target_size = (224, 224)
    
    mask = rasterize_bounding_boxes(boxes, orig_w, orig_h, target_size)
    assert mask.shape == (224, 224)
    assert mask.dtype == np.float32
    # Check that mask values are strictly binary {0.0, 1.0}
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0.0, 1.0})
    
    # Target box should be ~[22, 22, 45, 45] (100 * 224 / 1000 = 22.4)
    assert mask[25, 25] == 1.0
    assert mask[10, 10] == 0.0
    assert mask[100, 100] == 0.0


def test_dataset_item_shapes_and_types():
    """Verify that RealCXRDataset returns correct tensor shapes and types."""
    ds = RealCXRDataset(data_dir="data/real_cxr", split="train", target_size=(224, 224), in_channels=1)
    assert len(ds) > 0
    
    img_tensor, label_tensor, mask_tensor, meta = ds[0]
    
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(label_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert isinstance(meta, dict)
    
    # Check dimensions
    assert img_tensor.shape == (1, 224, 224)
    assert mask_tensor.shape == (1, 224, 224)
    assert label_tensor.ndim == 0  # scalar
    assert label_tensor.item() in (0, 1)
    
    # Check mask is non-negative and bounded by 1.0
    assert mask_tensor.min() >= 0.0
    assert mask_tensor.max() <= 1.0
    
    # Metadata assertions
    assert "image_id" in meta
    assert "patient_id" in meta
    assert "local_path" in meta
    assert os.path.exists(meta["local_path"])


def test_dataloaders_batching():
    """Verify batch iteration across train, val, and test loaders."""
    train_loader, val_loader, test_loader = get_real_cxr_dataloaders(
        data_dir="data/real_cxr",
        batch_size=4,
        target_size=(224, 224),
        in_channels=1
    )
    
    batch = next(iter(train_loader))
    imgs, labels, masks, metas = batch
    
    assert imgs.shape == (4, 1, 224, 224)
    assert labels.shape == (4,)
    assert masks.shape == (4, 1, 224, 224)
    assert len(metas["image_id"]) == 4
