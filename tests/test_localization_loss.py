"""test_localization_loss.py: Unit tests for differentiable Grad-CAM and localization loss."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.roi_localization import build_demonstration_cxr_model
from src.localization_loss import (
    DifferentiableGradCAM,
    alignment_loss,
    dice_loss,
    LocalizationAwareLoss,
    AdaptiveLocalizationAwareLoss,
    evaluate_saliency_metrics
)


def test_differentiable_gradcam_gradient_flow():
    """Verify that gradients propagate from Grad-CAM back to model parameters."""
    model, target_layer, _ = build_demonstration_cxr_model()
    model.train()
    
    gradcam = DifferentiableGradCAM(model, target_layer)
    x = torch.randn(2, 1, 64, 64, requires_grad=False)
    targets = torch.tensor([1, 0])
    
    logits, heatmaps = gradcam.compute_heatmap(x, target_classes=targets, create_graph=True)
    
    assert heatmaps.shape == (2, 1, 64, 64)
    assert not torch.isnan(heatmaps).any()
    assert not torch.isinf(heatmaps).any()
    assert heatmaps.min().item() >= 0.0
    assert heatmaps.max().item() <= 1.0 + 1e-5
    
    # Check backpropagation into the first convolutional layer
    loss = heatmaps.sum()
    loss.backward()
    
    first_conv_weight = model.features[0].weight
    assert first_conv_weight.grad is not None
    assert first_conv_weight.grad.norm().item() > 0.0
    
    gradcam.remove_hook()


def test_training_step_changes_parameters():
    """Verify that an optimizer step using composite loss modifies model parameters."""
    model, target_layer, _ = build_demonstration_cxr_model()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    gradcam = DifferentiableGradCAM(model, target_layer)
    criterion = LocalizationAwareLoss(mode="combined", lambda_loc=1.0)
    
    x = torch.randn(2, 1, 64, 64)
    targets = torch.tensor([1, 1])
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 20:40, 20:40] = 1.0  # Synthetic box mask
    
    # Store initial parameter snapshot
    initial_params = [p.clone() for p in model.parameters()]
    
    optimizer.zero_grad()
    logits, heatmaps = gradcam.compute_heatmap(x, targets, create_graph=True)
    loss_dict = criterion(logits, targets, heatmaps, masks)
    loss_dict["loss_total"].backward()
    optimizer.step()
    
    # Verify at least one parameter changed
    changed = any(not torch.equal(p0, p1) for p0, p1 in zip(initial_params, model.parameters()))
    assert changed is True
    
    gradcam.remove_hook()


def test_alignment_and_dice_loss_penalties():
    """Verify alignment and dice loss return higher penalties for misaligned heatmaps."""
    B, C, H, W = 1, 1, 32, 32
    
    # Create mask at top-left
    mask = torch.zeros(B, C, H, W)
    mask[:, :, 0:10, 0:10] = 1.0
    
    # Perfectly aligned heatmap
    h_aligned = mask.clone()
    
    # Completely misaligned heatmap at bottom-right
    h_misaligned = torch.zeros(B, C, H, W)
    h_misaligned[:, :, 20:30, 20:30] = 1.0
    
    l_align_good = alignment_loss(h_aligned, mask).item()
    l_align_bad = alignment_loss(h_misaligned, mask).item()
    assert l_align_good < l_align_bad
    assert l_align_good < 0.05
    assert l_align_bad > 0.95
    
    l_dice_good = dice_loss(h_aligned, mask).item()
    l_dice_bad = dice_loss(h_misaligned, mask).item()
    assert l_dice_good < l_dice_bad


def test_empty_mask_handling():
    """Verify that batches with empty masks (normal cases) do not produce NaNs."""
    B, C, H, W = 2, 1, 32, 32
    heatmaps = torch.rand(B, C, H, W)
    empty_masks = torch.zeros(B, C, H, W)
    
    l_align = alignment_loss(heatmaps, empty_masks)
    l_dice = dice_loss(heatmaps, empty_masks)
    
    assert not torch.isnan(l_align)
    assert not torch.isnan(l_dice)
    assert l_align.item() == 0.0
    assert l_dice.item() == 0.0


def test_evaluate_saliency_metrics():
    """Verify calculation of Dice, IoU, pointing game hit, and energy metrics."""
    H, W = 64, 64
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    gt_mask[16:32, 16:32] = 255  # 16x16 square ROI
    
    # Perfect heatmap with peak inside ROI
    heatmap_hit = np.zeros((H, W), dtype=np.float64)
    heatmap_hit[16:32, 16:32] = 0.9
    heatmap_hit[20, 20] = 1.0  # Peak inside
    
    metrics_hit = evaluate_saliency_metrics(heatmap_hit, gt_mask, threshold=0.5)
    assert metrics_hit["dice"] > 0.95
    assert metrics_hit["iou"] > 0.95
    assert metrics_hit["pointing_game_hit"] == 1.0
    assert metrics_hit["energy_inside_roi"] > 0.95
    
    # Heatmap with peak outside ROI
    heatmap_miss = np.zeros((H, W), dtype=np.float64)
    heatmap_miss[48:60, 48:60] = 0.9
    heatmap_miss[50, 50] = 1.0  # Peak outside
    
    metrics_miss = evaluate_saliency_metrics(heatmap_miss, gt_mask, threshold=0.5)
    assert metrics_miss["dice"] < 0.05
    assert metrics_miss["pointing_game_hit"] == 0.0
    assert metrics_miss["energy_inside_roi"] < 0.05


def test_adaptive_uncertainty_loss_optimization():
    """Verify that mode='adaptive' dynamically optimizes log_var uncertainty parameters."""
    model, target_layer, _ = build_demonstration_cxr_model()
    model.train()
    
    gradcam = DifferentiableGradCAM(model, target_layer)
    criterion = LocalizationAwareLoss(mode="adaptive")
    
    # Optimizer updates both neural weights AND loss uncertainty parameters
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=0.05
    )
    
    x = torch.randn(2, 1, 64, 64)
    targets = torch.tensor([1, 1])
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 20:40, 20:40] = 1.0
    
    initial_log_var_cls = criterion.log_var_cls.item()
    initial_log_var_align = criterion.log_var_align.item()
    initial_log_var_dice = criterion.log_var_dice.item()
    
    assert initial_log_var_cls == 0.0
    assert initial_log_var_align == 0.0
    assert initial_log_var_dice == 0.0
    
    optimizer.zero_grad()
    logits, heatmaps = gradcam.compute_heatmap(x, targets, create_graph=True)
    loss_dict = criterion(logits, targets, heatmaps, masks)
    
    assert "loss_total" in loss_dict
    assert "effective_lambda" in loss_dict
    assert "effective_beta" in loss_dict
    
    loss_dict["loss_total"].backward()
    
    # Check that uncertainty parameters have received non-zero gradients
    assert criterion.log_var_cls.grad is not None
    assert criterion.log_var_align.grad is not None
    assert criterion.log_var_dice.grad is not None
    
    optimizer.step()
    
    # Verify that at least one uncertainty parameter was adjusted by the optimizer
    weights = criterion.get_effective_weights()
    assert "w_cls" in weights
    assert "w_align" in weights
    assert "w_dice" in weights
    assert "effective_lambda_loc" in weights
    assert "effective_beta_dice" in weights
    
    params_changed = (
        criterion.log_var_cls.item() != initial_log_var_cls or
        criterion.log_var_align.item() != initial_log_var_align or
        criterion.log_var_dice.item() != initial_log_var_dice
    )
    assert params_changed is True
    
    gradcam.remove_hook()


def test_adaptive_localization_aware_loss_subclass():
    """Verify AdaptiveLocalizationAwareLoss convenience wrapper subclass."""
    criterion = AdaptiveLocalizationAwareLoss()
    assert criterion.mode == "adaptive"
    assert hasattr(criterion, "log_var_cls")
    assert hasattr(criterion, "log_var_align")
    assert hasattr(criterion, "log_var_dice")

