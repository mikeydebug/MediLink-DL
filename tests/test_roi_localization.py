"""test_roi_localization.py: Unit tests for Grad-CAM saliency extraction."""

import numpy as np
import pytest
import torch

from src.roi_localization import (
    GradCAM,
    build_demonstration_cxr_model,
    generate_roi_mask_from_heatmap,
    create_cam_overlay
)


def test_gradcam_execution_and_dimensions():
    model, target_layer, meta = build_demonstration_cxr_model()
    gradcam = GradCAM(model, target_layer)
    
    # Input tensor: 1 sample, 1 channel, 128x128
    input_tensor = torch.randn(1, 1, 128, 128, requires_grad=True)
    heatmap = gradcam.generate_heatmap(input_tensor, target_class_index=1)
    
    assert heatmap.shape == (128, 128)
    assert 0.0 <= np.min(heatmap)
    assert np.max(heatmap) <= 1.0
    gradcam.remove_hooks()


def test_generate_roi_mask_from_heatmap():
    heatmap = np.zeros((64, 64), dtype=np.float64)
    heatmap[20:40, 20:40] = 0.8  # Above threshold
    heatmap[0:10, 0:10] = 0.3    # Below threshold
    
    mask = generate_roi_mask_from_heatmap(heatmap, threshold=0.5)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert mask[25, 25] == 255
    assert mask[5, 5] == 0
    assert set(np.unique(mask)).issubset({0, 255})


def test_create_cam_overlay():
    image = np.full((64, 64), 100, dtype=np.uint8)
    heatmap = np.random.uniform(0.0, 1.0, size=(64, 64))
    
    overlay = create_cam_overlay(image, heatmap, alpha=0.5)
    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8
