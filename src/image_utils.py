"""image_utils.py: Medical Image I/O, Validation, and Synthetic CXR Phantom Generator.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ImageCorruptedError(Exception):
    """Raised when an image file exists but cannot be parsed/decoded."""
    pass


class InvalidImageFormatError(Exception):
    """Raised when an image does not meet dimensionality or type requirements."""
    pass


def validate_image(image: np.ndarray, min_dim: int = 4) -> None:
    """Validate that the image is a valid 2D grayscale NumPy array.
    
    Args:
        image: NumPy array to validate.
        min_dim: Minimum height and width allowed in pixels (default: 4).
        
    Raises:
        InvalidImageFormatError: If the array is not 2D uint8 or is smaller than min_dim.
    """
    if not isinstance(image, np.ndarray):
        raise InvalidImageFormatError(f"Expected np.ndarray, got {type(image)}")
    
    if image.ndim != 2:
        raise InvalidImageFormatError(
            f"Expected 2D grayscale image (H, W), got shape {image.shape} with {image.ndim} dimensions."
        )
    
    if image.dtype != np.uint8:
        raise InvalidImageFormatError(
            f"Expected image dtype np.uint8 (range 0-255), got {image.dtype}."
        )
        
    h, w = image.shape
    if h < min_dim or w < min_dim:
        raise InvalidImageFormatError(
            f"Image dimensions ({h}x{w}) are smaller than minimum allowed ({min_dim}x{min_dim})."
        )


def load_grayscale_image(
    image_path: str | Path,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Load an image file as a single-channel 8-bit grayscale array.
    
    Args:
        image_path: Path to the image file (PNG, JPEG, TIFF, etc.).
        target_size: Optional (height, width) tuple to resize image.
        
    Returns:
        np.ndarray: Grayscale image array with shape (H, W) and dtype np.uint8.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ImageCorruptedError: If the file cannot be decoded by PIL.
        InvalidImageFormatError: If dimensions are invalid.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path.resolve()}")
    
    try:
        with Image.open(path) as img:
            # Force conversion to 8-bit grayscale
            gray_img = img.convert("L")
            if target_size is not None:
                # PIL resize takes (width, height)
                gray_img = gray_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
            arr = np.array(gray_img, dtype=np.uint8)
    except Exception as exc:
        raise ImageCorruptedError(f"Failed to decode image at {path}: {exc}") from exc
    
    validate_image(arr)
    return arr


def save_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save a grayscale NumPy array to disk.
    
    Args:
        image: Grayscale image array of shape (H, W) and dtype uint8.
        output_path: Destination path.
        
    Returns:
        Path: Resolved output file path.
    """
    validate_image(image)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(image, mode="L")
    pil_img.save(out)
    return out.resolve()


# ==============================================================================
# [HISTORICAL / UNIT TEST FIXTURE ONLY - DEPRECATED FOR CLINICAL EVALUATION]
# NOTE: In earlier prototype phases, this synthetic phantom generator was used
# before the real Cohen et al. COVID-19 / Pneumonia CXR dataset was integrated.
# In the finalized system, production and clinical evaluations strictly use REAL CXRs
# loaded from `data/real_cxr/` and trained via `src/train_real_models.py`.
# This function is retained EXCLUSIVELY as a lightweight, deterministic offline fixture
# for fast unit testing (e.g. in `tests/test_image_utils.py` and `tests/test_transmission.py`)
# without requiring external dataset downloads on new developer machines.
# ==============================================================================
def generate_synthetic_cxr_phantom(
    width: int = 512,
    height: int = 512,
    seed: int = 42,
    add_lesion: bool = True,
    lesion_center: Optional[Tuple[int, int]] = None,  # (y, x), defaults to anatomical right middle lung field
    lesion_radius: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Synthesize an anatomically stylized grayscale chest X-ray phantom.
    
    CRITICAL RESEARCH NOTE:
    This function generates a synthetic phantom intended strictly for reproducible
    algorithmic testing when clinical X-ray datasets (e.g. NIH ChestX-ray14 or
    CheXpert) are not mounted locally. IT IS EXPLICITLY NOT A REAL MEDICAL X-RAY
    AND CANNOT BE USED FOR CLINICAL TRIAGE. A prominent synthetic disclaimer watermark
    is embedded directly into the pixel grid.
    
    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        seed: Random seed for reproducibility.
        add_lesion: Whether to inject a simulated focal consolidation (ROI).
        lesion_center: (row, col) coordinates of the lesion center.
        lesion_radius: Radius in pixels of the simulated focal pathology.
        
    Returns:
        tuple containing:
            - phantom (np.ndarray): Shape (H, W), dtype np.uint8.
            - roi_mask (np.ndarray): Shape (H, W), dtype np.uint8 (255 inside ROI, 0 outside).
            - metadata (dict): Descriptive metadata documenting synthetic generation parameters.
    """
    if lesion_center is None:
        lesion_center = (int(height * 0.62), int(width * 0.38))
    if lesion_radius is None:
        lesion_radius = max(8, int(min(height, width) * 0.065))
    rng = np.random.default_rng(seed)
    
    # Coordinate grids normalized to [-1, 1]
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    y_norm = (y_coords - height / 2.0) / (height / 2.0)
    x_norm = (x_coords - width / 2.0) / (width / 2.0)
    
    # Base background: soft gradient attenuation
    base = np.full((height, width), 25.0, dtype=np.float64)
    
    # Thoracic rib cage outer boundary (ellipse)
    thorax_mask = ((x_norm / 0.82) ** 2 + ((y_norm - 0.05) / 0.88) ** 2) <= 1.0
    base[thorax_mask] += 55.0
    
    # Bilateral Lung Fields (radiolucent: lower attenuation / darker regions)
    # Left lung cavity (anatomical right, patient left)
    left_lung_mask = (((x_norm - 0.36) / 0.28) ** 2 + ((y_norm - 0.05) / 0.62) ** 2) <= 1.0
    # Right lung cavity
    right_lung_mask = (((x_norm + 0.36) / 0.28) ** 2 + ((y_norm - 0.05) / 0.62) ** 2) <= 1.0
    
    base[left_lung_mask] -= 45.0
    base[right_lung_mask] -= 45.0
    
    # Mediastinum / Spine central column (radiopaque: brighter)
    spine_shadow = np.exp(-((x_norm) ** 2) / (2 * (0.12 ** 2))) * 80.0
    base += spine_shadow
    
    # Cardiac silhouette (prominent on patient left / viewer right)
    heart_mask = (((x_norm - 0.16) / 0.26) ** 2 + ((y_norm - 0.22) / 0.28) ** 2) <= 1.0
    base[heart_mask] += 65.0
    
    # Simulated rib cage horizontal attenuation bands
    rib_freq = 14.0
    rib_pattern = np.sin(y_norm * np.pi * rib_freq) ** 2
    rib_attenuation = rib_pattern * 22.0 * thorax_mask
    base += rib_attenuation
    
    # Simulated clavicles (horizontal radiopaque bands near top)
    clavicle_mask = (np.abs(y_norm - (-0.55)) < 0.04) & (np.abs(x_norm) < 0.7)
    base[clavicle_mask] += 40.0
    
    # Simulated vascular markings (low-amplitude Perlin/Gaussian texture)
    noise = rng.normal(loc=0.0, scale=4.5, size=(height, width))
    base += noise
    
    # Create Ground Truth ROI Mask
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    
    if add_lesion:
        ly, lx = lesion_center
        dist_sq = (y_coords - ly) ** 2 + (x_coords - lx) ** 2
        # Ground-truth binary mask within radius
        roi_mask[dist_sq <= (lesion_radius ** 2)] = 255
        
        # Inject smooth Gaussian focal consolidation into the image
        lesion_intensity = np.exp(-dist_sq / (2.0 * ((lesion_radius * 0.65) ** 2))) * 85.0
        base += lesion_intensity
    
    # Clamp base image to [0, 255] uint8
    phantom = np.clip(base, 0, 255).astype(np.uint8)
    
    # Permanently stamp visible disclaimer banner onto the image
    pil_phantom = Image.fromarray(phantom, mode="L")
    draw = ImageDraw.Draw(pil_phantom)
    disclaimer_text = "[SYNTHETIC PHANTOM - NOT A REAL MEDICAL X-RAY]"
    # Draw background box for banner
    banner_y = height - 24
    draw.rectangle([(0, banner_y), (width, height)], fill=0)
    draw.text((10, banner_y + 4), disclaimer_text, fill=255)
    
    phantom = np.array(pil_phantom, dtype=np.uint8)
    
    # Ensure ROI mask does not overlap banner
    roi_mask[banner_y:height, :] = 0
    
    metadata = {
        "is_synthetic": True,
        "width": width,
        "height": height,
        "seed": seed,
        "has_lesion": add_lesion,
        "lesion_center_yx": lesion_center,
        "lesion_radius": lesion_radius,
        "roi_pixel_count": int(np.count_nonzero(roi_mask)),
        "disclaimer": "SYNTHETIC PHANTOM ONLY - NOT REAL MEDICAL PATIENT DATA",
    }
    
    return phantom, roi_mask, metadata


def create_rectangular_roi_mask(
    image_shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """Create a binary rectangular ROI mask.
    
    Args:
        image_shape: (height, width) of the target image.
        bbox: Bounding box as (top, left, bottom, right).
        
    Returns:
        np.ndarray: Binary mask of shape (H, W), dtype uint8 with 255 inside bbox.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    top, left, bottom, right = bbox
    top = max(0, min(top, h))
    bottom = max(0, min(bottom, h))
    left = max(0, min(left, w))
    right = max(0, min(right, w))
    mask[top:bottom, left:right] = 255
    return mask
