"""roi_localization.py: Deep Learning Saliency and Grad-CAM ROI Extraction.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to generate
interpretability heatmaps and thresholded diagnostic Region-of-Interest (ROI) masks.

CRITICAL ACADEMIC & CLINICAL NOTICES:
1. Grad-CAM visualizes gradient-based feature importance learned by a neural network.
   It does NOT establish histological ground truth, anatomical boundaries, or clinical
   diagnostic safety.
2. Standard ImageNet-pretrained weights reflect consumer photographic features (dogs,
   cars) and MUST NOT be claimed as a clinically validated chest radiograph model.
3. Model loading is modularly isolated from the core image-processing pipeline so the
   prototype functions deterministically on CPU even without downloading external weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.image_utils import validate_image, save_image


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) implementation."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize Grad-CAM hook manager.
        
        Args:
            model: PyTorch neural network in eval mode.
            target_layer: The convolutional layer to inspect (usually the final conv layer).
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []
        
        self._register_hooks()
        
    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h1, h2])
        
    def remove_hooks(self) -> None:
        """Clean up forward and backward hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_index: Optional[int] = None
    ) -> np.ndarray:
        """Compute the 2D Grad-CAM saliency map for a given class.
        
        Args:
            input_tensor: Tensor of shape (1, C, H, W).
            target_class_index: Index of class to compute gradients for (default: argmax).
            
        Returns:
            np.ndarray: Normalized 2D heatmap in [0.0, 1.0] matching (H, W).
        """
        self.model.eval()
        self.activations = None
        self.gradients = None
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if target_class_index is None:
            target_class_index = int(torch.argmax(logits, dim=1).item())
            
        target_score = logits[0, target_class_index]
        
        # Zero gradients and backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=False)
        
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")
            
        # Global Average Pooling of gradients: alpha_k = (1/Z) * sum(grad_k)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Linear combination: L_cam = ReLU( sum(w_k * A_k) )
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)
        cam = F.relu(cam)
        
        # Interpolate to match original input spatial dimensions
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = F.interpolate(cam, size=(h_in, w_in), mode="bilinear", align_corners=False)
        cam_arr = cam_resized.squeeze().cpu().numpy()
        
        # Min-max normalize to [0.0, 1.0]
        c_min, c_max = np.min(cam_arr), np.max(cam_arr)
        if (c_max - c_min) > 1e-8:
            cam_norm = (cam_arr - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_arr)
            
        return cam_norm


def generate_roi_mask_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.50
) -> np.ndarray:
    """Threshold a continuous Grad-CAM heatmap into an operational binary ROI mask.
    
    Args:
        heatmap: Normalized float array in [0.0, 1.0] of shape (H, W).
        threshold: Saliency cutoff in [0.0, 1.0]. Pixels >= threshold are marked active.
        
    Returns:
        np.ndarray: Binary uint8 mask (H, W) where 255 = ROI, 0 = Background.
    """
    t = max(0.01, min(threshold, 0.99))
    mask = (heatmap >= t).astype(np.uint8) * 255
    return mask


def create_cam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:
    """Generate a pseudo-color heatmap overlay atop the grayscale radiograph.
    
    Args:
        image: Grayscale uint8 array (H, W).
        heatmap: Normalized float array (H, W) in [0.0, 1.0].
        alpha: Blend factor for the color overlay.
        
    Returns:
        np.ndarray: RGB uint8 array (H, W, 3).
    """
    validate_image(image)
    import matplotlib
    try:
        colormap = matplotlib.colormaps["jet"]
    except AttributeError:
        import matplotlib.cm as cm
        colormap = cm.get_cmap("jet")
    
    colored_heatmap = colormap(heatmap)[:, :, :3]  # drop alpha -> (H, W, 3) in [0, 1]
    colored_heatmap_uint8 = (colored_heatmap * 255.0).astype(np.uint8)
    
    # Expand grayscale image to 3 channels
    img_rgb = np.stack([image] * 3, axis=-1)
    
    # Alpha blend: overlay = (1 - alpha) * img + alpha * heatmap
    blended = (1.0 - alpha) * img_rgb.astype(np.float64) + alpha * colored_heatmap_uint8.astype(np.float64)
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def build_demonstration_cxr_model() -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    """Build a lightweight convolutional neural architecture for testable Grad-CAM execution.
    
    ACADEMIC VERIFICATION NOTE:
    This architecture serves as a verifiable, CPU-testable convolutional backbone
    (Conv -> ReLU -> MaxPool -> Conv -> GAP -> Linear) for demonstrating Grad-CAM
    gradient flow and ROI thresholding without requiring external multi-gigabyte
    checkpoints to be fetched at test time.
    """
    class SimpleCXRClassifier(nn.Module):
        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)  # Target layer for Grad-CAM
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, num_classes)
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.features(x)
            return self.classifier(feats)
            
    model = SimpleCXRClassifier(num_classes=2)
    model.eval()
    target_layer = model.features[8]  # Final Conv2d layer
    
    meta = {
        "architecture": "SimpleCXRClassifier (2-class prototype)",
        "target_layer": "features[8] (Conv2d 32->64)",
        "classes": ["Normal", "Focal Pathology"],
        "is_medically_certified": False,
        "disclaimer": "Prototype neural network for Grad-CAM algorithmic testing only."
    }
    return model, target_layer, meta
