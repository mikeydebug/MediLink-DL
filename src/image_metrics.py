"""image_metrics.py: Quantitative Medical Image Quality Evaluation.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Calculates global, ROI-specific, and background-specific PSNR and SSIM,
alongside compression payload metrics.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from skimage.metrics import structural_similarity

from src.image_utils import validate_image


class MetricComputationError(Exception):
    """Raised when metric inputs are incompatible or invalid."""
    pass


def calculate_mse(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Calculate Mean Squared Error (MSE), optionally localized by a binary mask.
    
    Args:
        original: Reference uint8 array (H, W).
        reconstructed: Test uint8 array (H, W).
        mask: Optional binary mask (H, W). If provided, MSE is computed only over mask > 0.
        
    Returns:
        float: Computed MSE, or float('nan') if mask has zero active pixels.
    """
    if original.shape != reconstructed.shape:
        raise MetricComputationError(
            f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
        )
    
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    sq_diff = diff ** 2
    
    if mask is not None:
        if mask.shape != original.shape:
            raise MetricComputationError(f"Mask shape {mask.shape} != image shape {original.shape}")
        active = mask > 0
        n_pixels = int(np.count_nonzero(active))
        if n_pixels == 0:
            return float("nan")
        return float(np.sum(sq_diff[active]) / n_pixels)
    
    return float(np.mean(sq_diff))


def calculate_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: float = 255.0
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    Args:
        original: Reference uint8 array (H, W).
        reconstructed: Test uint8 array (H, W).
        mask: Optional binary mask (H, W). If provided, calculates PSNR over mask > 0.
        data_range: Dynamic range of pixel intensities (255.0 for 8-bit).
        
    Returns:
        float: PSNR in dB, float('inf') for identical images, or float('nan') if mask is empty.
    """
    mse = calculate_mse(original, reconstructed, mask=mask)
    if math.isnan(mse):
        return float("nan")
    if mse == 0.0:
        return float("inf")
    
    return float(10.0 * math.log10((data_range ** 2) / mse))


def calculate_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: float = 255.0
) -> float:
    """Calculate Structural Similarity Index (SSIM), globally or partitioned by mask.
    
    Args:
        original: Reference uint8 array (H, W).
        reconstructed: Test uint8 array (H, W).
        mask: Optional binary mask (H, W). If provided, calculates average SSIM over mask > 0.
        data_range: Dynamic range (255.0 for 8-bit).
        
    Returns:
        float: SSIM value in [-1.0, 1.0], or float('nan') if mask is empty.
    """
    if original.shape != reconstructed.shape:
        raise MetricComputationError("Image shapes must match for SSIM calculation.")
    
    # Compute full per-pixel SSIM spatial error map
    global_ssim, ssim_map = structural_similarity(
        original,
        reconstructed,
        data_range=data_range,
        full=True
    )
    
    if mask is not None:
        if mask.shape != original.shape:
            raise MetricComputationError("Mask shape does not match image shape.")
        active = mask > 0
        if not np.any(active):
            return float("nan")
        return float(np.mean(ssim_map[active]))
    
    return float(global_ssim)


def evaluate_quality_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None,
    raw_bytes: int = 0,
    comp_bytes: int = 0,
    label: str = "baseline"
) -> Dict[str, Any]:
    """Perform comprehensive medical image quality evaluation.
    
    Computes global, ROI-specific, and background-specific PSNR and SSIM,
    along with file sizes, compression ratio, and size reduction percentage.
    
    ACADEMIC NOTE ON METRIC LIMITATIONS:
    - PSNR is based on pixel-wise L2 Euclidean distance. It penalizes uniform luminance
      shifts equally with structural boundary blurring, despite their vastly different
      radiological impact.
    - SSIM improves upon PSNR by evaluating local luminance, contrast, and structural
      covariance across sliding Gaussian windows. However, SSIM is agnostic to clinical
      semantics (e.g. subtle pneumothorax pleura lines or microcalcifications).
    - Partitioning metrics into ROI and background ensures that high background compression
      does not obscure diagnostic fidelity retention within critical pathological zones.
    """
    validate_image(original)
    validate_image(reconstructed)
    
    h, w = original.shape
    total_pixels = h * w
    
    if raw_bytes <= 0:
        raw_bytes = original.nbytes
    
    comp_ratio = raw_bytes / max(1, comp_bytes) if comp_bytes > 0 else 1.0
    size_red_pct = ((raw_bytes - comp_bytes) / raw_bytes * 100.0) if comp_bytes > 0 else 0.0
    
    # Global metrics
    psnr_global = calculate_psnr(original, reconstructed)
    ssim_global = calculate_ssim(original, reconstructed)
    
    # ROI and Background metrics
    psnr_roi = float("nan")
    ssim_roi = float("nan")
    psnr_bg = float("nan")
    ssim_bg = float("nan")
    roi_pixels = 0
    bg_pixels = total_pixels
    
    if mask is not None:
        if mask.shape != original.shape:
            raise MetricComputationError("Mask shape does not match image shape.")
        roi_bin = (mask > 0).astype(np.uint8)
        bg_bin = (mask == 0).astype(np.uint8)
        roi_pixels = int(np.count_nonzero(roi_bin))
        bg_pixels = int(np.count_nonzero(bg_bin))
        
        if roi_pixels > 0:
            psnr_roi = calculate_psnr(original, reconstructed, mask=roi_bin)
            ssim_roi = calculate_ssim(original, reconstructed, mask=roi_bin)
            
        if bg_pixels > 0:
            psnr_bg = calculate_psnr(original, reconstructed, mask=bg_bin)
            ssim_bg = calculate_ssim(original, reconstructed, mask=bg_bin)
            
    return {
        "label": label,
        "image_shape": f"{h}x{w}",
        "total_pixels": total_pixels,
        "roi_pixels": roi_pixels,
        "bg_pixels": bg_pixels,
        "roi_pixel_fraction": roi_pixels / max(1, total_pixels),
        "raw_bytes": raw_bytes,
        "compressed_bytes": comp_bytes,
        "compression_ratio": round(comp_ratio, 2),
        "size_reduction_pct": round(size_red_pct, 2),
        "psnr_global_db": round(psnr_global, 2) if not math.isnan(psnr_global) else "NaN",
        "ssim_global": round(ssim_global, 4) if not math.isnan(ssim_global) else "NaN",
        "psnr_roi_db": round(psnr_roi, 2) if not math.isnan(psnr_roi) else "NaN",
        "ssim_roi": round(ssim_roi, 4) if not math.isnan(ssim_roi) else "NaN",
        "psnr_bg_db": round(psnr_bg, 2) if not math.isnan(psnr_bg) else "NaN",
        "ssim_bg": round(ssim_bg, 4) if not math.isnan(ssim_bg) else "NaN",
    }


def export_metrics_to_csv(
    metrics_list: List[Dict[str, Any]],
    output_csv_path: str | Path
) -> Path:
    """Export evaluated quality metrics to a structured CSV file.
    
    Args:
        metrics_list: List of metric dictionaries from evaluate_quality_metrics.
        output_csv_path: Target CSV file destination (e.g. results/metrics.csv).
        
    Returns:
        Path: Resolved output CSV path.
    """
    if not metrics_list:
        raise MetricComputationError("Cannot export empty metrics list.")
        
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "label",
        "image_shape",
        "total_pixels",
        "roi_pixels",
        "bg_pixels",
        "roi_pixel_fraction",
        "raw_bytes",
        "compressed_bytes",
        "compression_ratio",
        "size_reduction_pct",
        "psnr_global_db",
        "ssim_global",
        "psnr_roi_db",
        "ssim_roi",
        "psnr_bg_db",
        "ssim_bg",
    ]
    
    with open(out, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics_list:
            # Filter keys to fieldnames
            row = {k: item.get(k, "") for k in fieldnames}
            writer.writerow(row)
            
    return out.resolve()
