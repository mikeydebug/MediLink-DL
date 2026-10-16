"""roi_compression.py: Region-of-Interest (ROI) Aware Dual-Quality Compression Baseline.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from src.image_utils import validate_image, save_image


class ROICompressionError(Exception):
    """Raised when ROI compression fails due to invalid parameters or formatting."""
    pass


def compress_jpeg_to_bytes(image: np.ndarray, quality: int) -> bytes:
    """Encode a 2D grayscale image array into standard JPEG byte stream.
    
    Args:
        image: Grayscale uint8 array of shape (H, W).
        quality: JPEG quality factor (1-95).
        
    Returns:
        bytes: Raw JPEG bitstream.
    """
    validate_image(image)
    q = max(1, min(int(quality), 95))
    pil_img = Image.fromarray(image, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=q, optimize=True)
    return buf.getvalue()


def decompress_jpeg_from_bytes(jpeg_bytes: bytes) -> np.ndarray:
    """Decode a JPEG byte stream into a 2D grayscale uint8 array.
    
    Args:
        jpeg_bytes: Encoded JPEG bytes.
        
    Returns:
        np.ndarray: Decoded uint8 array of shape (H, W).
    """
    buf = io.BytesIO(jpeg_bytes)
    with Image.open(buf) as pil_img:
        arr = np.array(pil_img.convert("L"), dtype=np.uint8)
    return arr


def serialize_dual_stream_payload(
    bg_jpeg_bytes: bytes,
    roi_crop_jpeg_bytes: bytes,
    roi_bbox: Tuple[int, int, int, int],  # (top, left, bottom, right)
    image_shape: Tuple[int, int],
    q_roi: int,
    q_bg: int
) -> bytes:
    """Package dual-stream ROI transmission data into a verifiable binary format.
    
    Format specification:
    - 4 bytes: Magic header b"MLDL"
    - 2 bytes: Version (uint16 = 1)
    - 2 bytes: Image Height (uint16)
    - 2 bytes: Image Width (uint16)
    - 1 byte:  q_roi (uint8)
    - 1 byte:  q_bg (uint8)
    - 2 bytes: bbox top (uint16)
    - 2 bytes: bbox left (uint16)
    - 2 bytes: bbox bottom (uint16)
    - 2 bytes: bbox right (uint16)
    - 4 bytes: bg_bytes length (uint32)
    - 4 bytes: roi_crop_bytes length (uint32)
    - N bytes: bg_jpeg_bytes
    - M bytes: roi_crop_jpeg_bytes
    """
    h, w = image_shape
    top, left, bottom, right = roi_bbox
    header = struct.pack(
        ">4sHHHBBHHHHII",
        b"MLDL",
        1,       # version
        h, w,
        q_roi, q_bg,
        top, left, bottom, right,
        len(bg_jpeg_bytes),
        len(roi_crop_jpeg_bytes)
    )
    return header + bg_jpeg_bytes + roi_crop_jpeg_bytes


def deserialize_dual_stream_payload(
    payload_bytes: bytes
) -> Dict[str, Any]:
    """Unpack a dual-stream ROI transmission binary payload."""
    header_size = struct.calcsize(">4sHHHBBHHHHII")
    if len(payload_bytes) < header_size:
        raise ROICompressionError(f"Payload size ({len(payload_bytes)}) smaller than header ({header_size}).")
    
    magic, ver, h, w, q_roi, q_bg, top, left, bottom, right, bg_len, roi_len = struct.unpack(
        ">4sHHHBBHHHHII", payload_bytes[:header_size]
    )
    if magic != b"MLDL":
        raise ROICompressionError(f"Invalid magic header: {magic}")
    
    offset = header_size
    bg_bytes = payload_bytes[offset: offset + bg_len]
    offset += bg_len
    roi_bytes = payload_bytes[offset: offset + roi_len]
    
    if len(bg_bytes) != bg_len or len(roi_bytes) != roi_len:
        raise ROICompressionError("Payload truncated or lengths mismatch.")
    
    return {
        "version": ver,
        "image_shape": (h, w),
        "q_roi": q_roi,
        "q_bg": q_bg,
        "roi_bbox": (top, left, bottom, right),
        "bg_jpeg_bytes": bg_bytes,
        "roi_crop_jpeg_bytes": roi_bytes
    }


def compute_mask_bounding_box(mask: np.ndarray, padding: int = 4) -> Tuple[int, int, int, int]:
    """Compute the bounding box (top, left, bottom, right) of a binary mask.
    
    If the mask is completely empty, returns (0, 0, 0, 0).
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)
    
    h, w = mask.shape
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    top = max(0, rmin - padding)
    bottom = min(h, rmax + 1 + padding)
    left = max(0, cmin - padding)
    right = min(w, cmax + 1 + padding)
    return (int(top), int(left), int(bottom), int(right))


def compress_roi_aware(
    image: np.ndarray,
    mask: np.ndarray,
    q_roi: int = 90,
    q_bg: int = 25
) -> Tuple[np.ndarray, bytes, Dict[str, Any]]:
    """Compress an image with high quality in the ROI and lower quality in the background.
    
    ACADEMIC & ALGORITHMIC NOTE:
    Standard baseline JPEG (ISO/IEC 10918-1) does not natively support non-uniform,
    spatially varying quantization matrices across arbitrary non-rectangular regions
    without format extensions (unlike JPEG2000 Part 2 MAXSHIFT). This implementation
    constructs an operational dual-stream transmission baseline:
    1. A low-quality full-frame background stream (q_bg).
    2. A high-quality ROI cropped patch stream (q_roi) localized to the diagnostic mask bbox.
    3. The client receives both streams and composites the high-fidelity ROI over the
       low-fidelity context image, preserving full spatial resolution and high local fidelity.
       
    CLINICAL DISCLAIMER:
    Fidelity retention in the ROI does not imply diagnostic safety or clinical equivalency.
    The background region contains lossy artifacts that may obscure incidental findings.
    
    Args:
        image: Original grayscale image (H, W), dtype uint8.
        mask: Binary or soft ROI mask (H, W), values in [0, 255] or [0.0, 1.0].
        q_roi: JPEG quality for the ROI (1-95).
        q_bg: JPEG quality for the background (1-95).
        
    Returns:
        tuple containing:
            - reconstructed (np.ndarray): Shape (H, W), dtype uint8.
            - payload_bytes (bytes): Complete serializable dual-stream payload.
            - info (dict): Detailed compression diagnostics and metadata.
    """
    validate_image(image)
    if mask.shape != image.shape:
        raise ROICompressionError(
            f"Mask shape {mask.shape} does not match image shape {image.shape}."
        )
    
    # Normalize mask to float [0.0, 1.0]
    if mask.dtype == np.uint8:
        mask_float = mask.astype(np.float64) / 255.0
    else:
        mask_float = np.clip(mask.astype(np.float64), 0.0, 1.0)
    
    # 1. Full-frame background compression
    bg_jpeg_bytes = compress_jpeg_to_bytes(image, quality=q_bg)
    bg_decompressed = decompress_jpeg_from_bytes(bg_jpeg_bytes)
    
    # 2. Check ROI bounding box
    binary_mask = (mask_float > 0.1).astype(np.uint8) * 255
    bbox = compute_mask_bounding_box(binary_mask)
    top, left, bottom, right = bbox
    
    if (bottom - top) > 0 and (right - left) > 0:
        # Crop ROI sub-image
        roi_crop = image[top:bottom, left:right]
        roi_crop_jpeg = compress_jpeg_to_bytes(roi_crop, quality=q_roi)
        roi_crop_decompressed = decompress_jpeg_from_bytes(roi_crop_jpeg)
        
        # Build composite reconstructed image
        reconstructed = bg_decompressed.copy()
        
        # Alpha-blend the crop using the sub-mask for seamless boundary transition
        sub_mask = mask_float[top:bottom, left:right]
        blended_roi = (
            sub_mask * roi_crop_decompressed.astype(np.float64) +
            (1.0 - sub_mask) * bg_decompressed[top:bottom, left:right].astype(np.float64)
        )
        reconstructed[top:bottom, left:right] = np.clip(np.round(blended_roi), 0, 255).astype(np.uint8)
    else:
        # Empty ROI fallback
        roi_crop_jpeg = b""
        reconstructed = bg_decompressed.copy()
    
    # 3. Serialize dual-stream payload
    payload_bytes = serialize_dual_stream_payload(
        bg_jpeg_bytes=bg_jpeg_bytes,
        roi_crop_jpeg_bytes=roi_crop_jpeg,
        roi_bbox=bbox,
        image_shape=image.shape,
        q_roi=q_roi,
        q_bg=q_bg
    )
    
    raw_uncompressed_bytes = image.nbytes
    compressed_bytes = len(payload_bytes)
    compression_ratio = raw_uncompressed_bytes / max(1, compressed_bytes)
    size_reduction_pct = (1.0 - (compressed_bytes / raw_uncompressed_bytes)) * 100.0
    
    info = {
        "mode": "roi_aware_dual_stream",
        "q_roi": q_roi,
        "q_bg": q_bg,
        "roi_bbox": bbox,
        "uncompressed_bytes": raw_uncompressed_bytes,
        "compressed_bytes": compressed_bytes,
        "bg_stream_bytes": len(bg_jpeg_bytes),
        "roi_stream_bytes": len(roi_crop_jpeg),
        "compression_ratio": compression_ratio,
        "size_reduction_pct": size_reduction_pct,
        "has_roi": bool((bottom - top) > 0 and (right - left) > 0),
        "disclaimer": "Experimental dual-stream JPEG approximation; not certified for clinical diagnostic use."
    }
    
    return reconstructed, payload_bytes, info


def reconstruct_from_payload(payload_bytes: bytes) -> np.ndarray:
    """Reconstruct an image from a received serialized dual-stream payload."""
    data = deserialize_dual_stream_payload(payload_bytes)
    bg_decompressed = decompress_jpeg_from_bytes(data["bg_jpeg_bytes"])
    
    top, left, bottom, right = data["roi_bbox"]
    if len(data["roi_crop_jpeg_bytes"]) > 0 and (bottom - top) > 0 and (right - left) > 0:
        roi_crop_decompressed = decompress_jpeg_from_bytes(data["roi_crop_jpeg_bytes"])
        reconstructed = bg_decompressed.copy()
        reconstructed[top:bottom, left:right] = roi_crop_decompressed
        return reconstructed
    return bg_decompressed


def compress_uniform(
    image: np.ndarray,
    quality: int
) -> Tuple[np.ndarray, bytes, Dict[str, Any]]:
    """Compress an image uniformly using standard JPEG.
    
    Args:
        image: Grayscale uint8 array (H, W).
        quality: JPEG quality factor (1-95).
        
    Returns:
        tuple containing:
            - reconstructed (np.ndarray): Decoded uint8 array (H, W).
            - jpeg_bytes (bytes): Encoded bitstream.
            - info (dict): Compression metadata.
    """
    validate_image(image)
    q = max(1, min(int(quality), 95))
    jpeg_bytes = compress_jpeg_to_bytes(image, quality=q)
    reconstructed = decompress_jpeg_from_bytes(jpeg_bytes)
    
    raw_bytes = image.nbytes
    comp_bytes = len(jpeg_bytes)
    compression_ratio = raw_bytes / max(1, comp_bytes)
    size_reduction_pct = (1.0 - (comp_bytes / raw_bytes)) * 100.0
    
    info = {
        "mode": "uniform_jpeg",
        "quality": q,
        "uncompressed_bytes": raw_bytes,
        "compressed_bytes": comp_bytes,
        "compression_ratio": compression_ratio,
        "size_reduction_pct": size_reduction_pct
    }
    
    return reconstructed, jpeg_bytes, info


def find_comparable_uniform_quality(
    image: np.ndarray,
    target_bytes: int
) -> Tuple[int, np.ndarray, bytes, Dict[str, Any]]:
    """Find a uniform JPEG quality level that produces a file size closest to target_bytes.
    
    Performs an exact evaluation across valid integer quality factors [1, 95] to ensure
    the fair rate-distortion comparison mandated by academic experimental methodology.
    
    Args:
        image: Grayscale uint8 array (H, W).
        target_bytes: Target byte size (e.g., from ROI-aware payload).
        
    Returns:
        tuple: (best_quality, reconstructed_image, jpeg_bytes, info)
    """
    best_q = 50
    best_diff = float("inf")
    best_bytes = None
    best_recon = None
    best_info = None
    
    # Binary search or bounded linear sweep (1 to 95 is only 95 steps, very fast on 512x512)
    for q in range(1, 96):
        recon, data, info = compress_uniform(image, quality=q)
        diff = abs(len(data) - target_bytes)
        if diff < best_diff:
            best_diff = diff
            best_q = q
            best_bytes = data
            best_recon = recon
            best_info = info
            
    best_info["target_bytes"] = target_bytes
    best_info["size_delta_bytes"] = len(best_bytes) - target_bytes
    best_info["size_delta_pct"] = (len(best_bytes) - target_bytes) / max(1, target_bytes) * 100.0
    
    return best_q, best_recon, best_bytes, best_info
