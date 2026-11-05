"""localization_loss.py: Differentiable Grad-CAM and Localization-Aware Loss.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Implements differentiable Grad-CAM backpropagation, alignment loss, soft-Dice loss,
and combined localization-penalized training objectives to guide neural attention
toward diagnostically certified reference ROIs.

THEORETICAL AND ACADEMIC CONTEXT:
Standard classification CNNs are trained solely on global category labels (L_cls).
Grad-CAM heatmaps derived from such models frequently suffer from "saliency drift",
where high activation falls on irrelevant artifacts (e.g. text labels, clavicles)
rather than the pathology. By backpropagating a differentiable localization penalty:
    L_total = L_cls + lambda_loc * (L_align + beta * L_dice)
through second-order autograd (create_graph=True), model gradients directly penalize
divergence between the gradient-weighted feature activations and the reference ROI mask.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGradCAM:
    """Differentiable Grad-CAM engine preserving autograd computational graph for training."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize with model and target convolutional layer.
        
        Args:
            model: PyTorch model (nn.Module).
            target_layer: Target convolutional module (e.g. final conv layer).
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self) -> None:
        def hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            # Preserve graph: do NOT call .detach()
            self.activations = out
            
        self._hook_handle = self.target_layer.register_forward_hook(hook)

    def remove_hook(self) -> None:
        """Remove forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def compute_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_classes: Optional[torch.Tensor] = None,
        create_graph: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and differentiable Grad-CAM heatmap.
        
        Args:
            input_tensor: Batch of images of shape (B, 1, H, W).
            target_classes: Tensor of target class indices of shape (B,). If None, argmax.
            create_graph: If True, keeps autograd graph for second-order backprop.
            
        Returns:
            Tuple of:
                - logits: Model outputs of shape (B, num_classes).
                - heatmaps: Differentiable normalized heatmaps of shape (B, 1, H, W) in [0.0, 1.0].
        """
        with torch.enable_grad():
            self.activations = None
            logits = self.model(input_tensor)
        
        if self.activations is None:
            raise RuntimeError("DifferentiableGradCAM failed to capture layer activations.")
            
        A = self.activations  # Shape (B, C, H_feat, W_feat)
        B, C, H_feat, W_feat = A.shape
        H_in, W_in = input_tensor.shape[2], input_tensor.shape[3]
        
        if target_classes is None:
            target_classes = torch.argmax(logits, dim=1)
            
        # Select target class scores for each sample in batch
        target_scores = logits.gather(1, target_classes.view(-1, 1)).squeeze(1)  # (B,)
        
        # Compute first-order gradients of target scores with respect to feature maps A
        grads = torch.autograd.grad(
            outputs=target_scores.sum(),
            inputs=A,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True
        )[0]  # Shape (B, C, H_feat, W_feat)
        
        # Global Average Pooling (GAP) of gradients: alpha_k = (1 / Z) * sum_{i,j} dY / dA_k,i,j
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Linear combination of weighted feature maps followed by ReLU
        cam = torch.sum(weights * A, dim=1, keepdim=True)  # (B, 1, H_feat, W_feat)
        cam = F.relu(cam)
        
        # Bilinear interpolation to input spatial dimensions
        cam_resized = F.interpolate(
            cam,
            size=(H_in, W_in),
            mode="bilinear",
            align_corners=False
        )  # (B, 1, H_in, W_in)
        
        # Differentiable min-max normalization to [0.0, 1.0] per image
        c_min = torch.amin(cam_resized, dim=(2, 3), keepdim=True)
        c_max = torch.amax(cam_resized, dim=(2, 3), keepdim=True)
        denom = c_max - c_min + 1e-7
        heatmaps = (cam_resized - c_min) / denom
        
        return logits, heatmaps


def alignment_loss(
    heatmaps: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """Compute normalized Saliency Alignment Loss.
    
    L_align = 1.0 - sum(H * M) / (sum(H) + eps)
    
    Penalizes Grad-CAM energy that spreads outside the reference ROI mask.
    Only active for samples that contain a positive reference mask.
    
    Args:
        heatmaps: Tensor of shape (B, 1, H, W) in [0.0, 1.0].
        masks: Binary reference mask tensor of shape (B, 1, H, W) with values in {0.0, 1.0}.
        eps: Small numerical stabilizer.
        
    Returns:
        torch.Tensor: Scalar loss averaged across positive-mask samples.
    """
    intersection = torch.sum(heatmaps * masks, dim=(1, 2, 3))  # (B,)
    sum_H = torch.sum(heatmaps, dim=(1, 2, 3))                  # (B,)
    sum_M = torch.sum(masks, dim=(1, 2, 3))                     # (B,)
    
    # Calculate per-sample alignment penalty
    per_sample_align = 1.0 - (intersection / (sum_H + eps))
    
    # Active only on samples containing non-empty reference mask
    has_mask = (sum_M > 0.5).float()
    num_positive = torch.sum(has_mask)
    
    if num_positive > 0:
        return torch.sum(per_sample_align * has_mask) / (num_positive + eps)
    else:
        # If all samples in batch have empty masks (pure normal batch), return zero loss
        return torch.tensor(0.0, device=heatmaps.device, dtype=heatmaps.dtype, requires_grad=True)


def dice_loss(
    heatmaps: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """Compute soft continuous Dice Loss between Grad-CAM heatmap and reference ROI mask.
    
    L_dice = 1.0 - (2.0 * sum(H * M) + eps) / (sum(H) + sum(M) + eps)
    
    Args:
        heatmaps: Tensor of shape (B, 1, H, W) in [0.0, 1.0].
        masks: Binary reference mask tensor of shape (B, 1, H, W) with values in {0.0, 1.0}.
        eps: Small numerical stabilizer.
        
    Returns:
        torch.Tensor: Scalar soft Dice loss.
    """
    intersection = torch.sum(heatmaps * masks, dim=(1, 2, 3))
    sum_H = torch.sum(heatmaps, dim=(1, 2, 3))
    sum_M = torch.sum(masks, dim=(1, 2, 3))
    
    per_sample_dice = 1.0 - ((2.0 * intersection + eps) / (sum_H + sum_M + eps))
    
    has_mask = (sum_M > 0.5).float()
    num_positive = torch.sum(has_mask)
    
    if num_positive > 0:
        return torch.sum(per_sample_dice * has_mask) / (num_positive + eps)
    else:
        return torch.tensor(0.0, device=heatmaps.device, dtype=heatmaps.dtype, requires_grad=True)


class LocalizationAwareLoss(nn.Module):
    """Composite loss function combining classification and localization objectives."""
    
    def __init__(
        self,
        mode: str = "combined",
        lambda_loc: float = 1.0,
        beta_dice: float = 0.5,
        eps: float = 1e-7
    ):
        """Initialize composite loss.
        
        Args:
            mode: One of 'cls_only', 'align_only', 'dice_only', 'combined', 'adaptive'.
                  'adaptive' uses Kendall & Gal (CVPR 2018) Homoscedastic Uncertainty Weighting,
                  where optimal task penalties (lambda_loc, beta_dice) are learned automatically
                  via gradient descent, removing the need for manual hyperparameter guessing.
            lambda_loc: Scaling factor for localization penalty (used in fixed modes).
            beta_dice: Weight of Dice penalty relative to alignment penalty (used in fixed modes).
            eps: Numerical stabilization constant.
        """
        super().__init__()
        valid_modes = ("cls_only", "align_only", "dice_only", "combined", "adaptive")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
            
        self.mode = mode
        self.lambda_loc = float(lambda_loc)
        self.beta_dice = float(beta_dice)
        self.eps = float(eps)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Homoscedastic Uncertainty Parameters (Kendall & Gal, CVPR 2018)
        # s_i = log(sigma_i^2). Initial s = 0.0 => sigma = 1.0 => exp(-s) = 1.0 (equal initial weight).
        # When mode == 'adaptive', these parameters are updated automatically during backpropagation.
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_align = nn.Parameter(torch.zeros(1))
        self.log_var_dice = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        heatmaps: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute composite training loss.
        
        Args:
            logits: (B, num_classes) classification predictions.
            targets: (B,) class indices.
            heatmaps: (B, 1, H, W) Grad-CAM heatmaps.
            masks: (B, 1, H, W) reference ROI binary masks.
            
        Returns:
            Dict containing 'loss_total', 'loss_cls', 'loss_loc', 'loss_align', 'loss_dice',
            and when in adaptive mode: 'effective_lambda', 'effective_beta', and log-variance values.
        """
        l_cls = self.ce_loss(logits, targets)
        zero_tensor = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        if self.mode == "cls_only" or heatmaps is None or masks is None or (self.lambda_loc <= 0.0 and self.mode != "adaptive"):
            return {
                "loss_total": l_cls,
                "loss_cls": l_cls,
                "loss_loc": zero_tensor,
                "loss_align": zero_tensor,
                "loss_dice": zero_tensor
            }
            
        l_align = alignment_loss(heatmaps, masks, eps=self.eps)
        l_dice = dice_loss(heatmaps, masks, eps=self.eps)
        
        if self.mode == "adaptive":
            # Homoscedastic Uncertainty Weighting (Kendall et al., CVPR 2018)
            # L_i_weighted = 0.5 * exp(-s_i) * L_i + 0.5 * s_i
            loss_cls_term = 0.5 * torch.exp(-self.log_var_cls) * l_cls + 0.5 * self.log_var_cls
            loss_align_term = 0.5 * torch.exp(-self.log_var_align) * l_align + 0.5 * self.log_var_align
            loss_dice_term = 0.5 * torch.exp(-self.log_var_dice) * l_dice + 0.5 * self.log_var_dice
            
            l_total = loss_cls_term + loss_align_term + loss_dice_term
            l_loc = l_align + l_dice
            
            # Derive effective dynamically learned penalty factors
            w_cls = torch.exp(-self.log_var_cls).item()
            w_align = torch.exp(-self.log_var_align).item()
            w_dice = torch.exp(-self.log_var_dice).item()
            effective_lambda = w_align / (w_cls + self.eps)
            effective_beta = w_dice / (w_align + self.eps)
            
            return {
                "loss_total": l_total,
                "loss_cls": l_cls,
                "loss_loc": l_loc,
                "loss_align": l_align,
                "loss_dice": l_dice,
                "effective_lambda": effective_lambda,
                "effective_beta": effective_beta,
                "log_var_cls": self.log_var_cls.item(),
                "log_var_align": self.log_var_align.item(),
                "log_var_dice": self.log_var_dice.item()
            }
        
        if self.mode == "align_only":
            l_loc = l_align
        elif self.mode == "dice_only":
            l_loc = l_dice
        elif self.mode == "combined":
            l_loc = l_align + self.beta_dice * l_dice
        else:
            l_loc = zero_tensor
            
        l_total = l_cls + self.lambda_loc * l_loc
        
        return {
            "loss_total": l_total,
            "loss_cls": l_cls,
            "loss_loc": l_loc,
            "loss_align": l_align,
            "loss_dice": l_dice
        }

    def get_effective_weights(self) -> Dict[str, float]:
        """Return effective loss weights w_i = exp(-s_i) and derived lambda/beta factors."""
        w_cls = float(torch.exp(-self.log_var_cls).item())
        w_align = float(torch.exp(-self.log_var_align).item())
        w_dice = float(torch.exp(-self.log_var_dice).item())
        return {
            "w_cls": w_cls,
            "w_align": w_align,
            "w_dice": w_dice,
            "sigma_cls": float(torch.exp(0.5 * self.log_var_cls).item()),
            "sigma_align": float(torch.exp(0.5 * self.log_var_align).item()),
            "sigma_dice": float(torch.exp(0.5 * self.log_var_dice).item()),
            "effective_lambda_loc": w_align / (w_cls + 1e-9),
            "effective_beta_dice": w_dice / (w_align + 1e-9),
        }


class AdaptiveLocalizationAwareLoss(LocalizationAwareLoss):
    """Convenience subclass for LocalizationAwareLoss(mode='adaptive').
    
    Implements Kendall & Gal (CVPR 2018) Multi-Task Homoscedastic Uncertainty Weighting.
    Automatically optimizes task penalties (lambda_loc, beta_dice) during training
    via second-order backpropagation, eliminating manual hyperparameter guesswork.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__(mode="adaptive", eps=eps)


def evaluate_saliency_metrics(
    heatmap: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    """Evaluate localization accuracy between continuous Grad-CAM heatmap and binary ground truth.
    
    Args:
        heatmap: 2D float array in [0.0, 1.0] of shape (H, W).
        ground_truth_mask: 2D uint8 array (255 or 1 inside ROI, 0 elsewhere).
        threshold: Binarization cutoff for heatmap.
        
    Returns:
        Dict containing:
            - dice: Binary Dice coefficient [0.0, 1.0].
            - iou: Intersection over Union (Jaccard index) [0.0, 1.0].
            - pointing_game_hit: 1.0 if peak heatmap pixel is inside ground truth ROI, else 0.0.
            - energy_inside_roi: Fraction of total heatmap energy falling inside reference ROI.
    """
    h_arr = np.clip(np.asarray(heatmap, dtype=np.float64), 0.0, 1.0)
    m_arr = (np.asarray(ground_truth_mask) > 0).astype(np.float64)
    
    # Check if ground truth mask is empty
    sum_m = np.sum(m_arr)
    if sum_m == 0:
        return {
            "dice": 1.0 if np.max(h_arr) < threshold else 0.0,
            "iou": 1.0 if np.max(h_arr) < threshold else 0.0,
            "pointing_game_hit": 0.0,
            "energy_inside_roi": 0.0
        }
        
    # Binarize heatmap
    h_bin = (h_arr >= threshold).astype(np.float64)
    
    intersection = np.sum(h_bin * m_arr)
    sum_h_bin = np.sum(h_bin)
    
    # Dice: 2 * |H & M| / (|H| + |M|)
    dice = (2.0 * intersection) / (sum_h_bin + sum_m + 1e-9)
    
    # IoU: |H & M| / |H | M|
    union = sum_h_bin + sum_m - intersection
    iou = intersection / (union + 1e-9)
    
    # Pointing Game Accuracy (Zhang et al. ECCV 2018):
    # Maximum saliency point argmax_{(y,x)} H(y, x)
    peak_y, peak_x = np.unravel_index(np.argmax(h_arr), h_arr.shape)
    pointing_hit = 1.0 if m_arr[peak_y, peak_x] > 0.5 else 0.0
    
    # Energy inside ROI: sum(H * M) / sum(H)
    total_energy = np.sum(h_arr)
    energy_in_roi = np.sum(h_arr * m_arr) / (total_energy + 1e-9)
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pointing_game_hit": float(pointing_hit),
        "energy_inside_roi": float(energy_in_roi)
    }
