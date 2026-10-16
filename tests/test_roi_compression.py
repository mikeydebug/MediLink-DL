"""test_roi_compression.py: Unit tests for ROI compression and serialization."""

import numpy as np
import pytest

from src.roi_compression import (
    compress_jpeg_to_bytes,
    decompress_jpeg_from_bytes,
    compress_roi_aware,
    compress_uniform,
    find_comparable_uniform_quality,
    reconstruct_from_payload,
    deserialize_dual_stream_payload,
    ROICompressionError
)
from src.image_utils import generate_synthetic_cxr_phantom


def test_jpeg_bytes_roundtrip():
    img = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)
    jpeg_bytes = compress_jpeg_to_bytes(img, quality=80)
    assert len(jpeg_bytes) > 0
    recon = decompress_jpeg_from_bytes(jpeg_bytes)
    assert recon.shape == img.shape
    assert recon.dtype == np.uint8


def test_roi_aware_compression_basic():
    phantom, mask, _ = generate_synthetic_cxr_phantom(width=256, height=256, seed=123)
    recon, payload, info = compress_roi_aware(phantom, mask, q_roi=90, q_bg=20)
    
    assert recon.shape == phantom.shape
    assert recon.dtype == np.uint8
    assert len(payload) > 0
    assert info["compression_ratio"] > 1.0
    assert info["size_reduction_pct"] > 0.0
    assert info["has_roi"] is True


def test_roi_aware_empty_mask():
    phantom = np.full((128, 128), 120, dtype=np.uint8)
    empty_mask = np.zeros((128, 128), dtype=np.uint8)
    recon, payload, info = compress_roi_aware(phantom, empty_mask, q_roi=90, q_bg=25)
    
    assert recon.shape == phantom.shape
    assert info["has_roi"] is False
    assert len(payload) > 0


def test_payload_reconstruction():
    phantom, mask, _ = generate_synthetic_cxr_phantom(width=128, height=128, seed=42)
    _, payload, _ = compress_roi_aware(phantom, mask, q_roi=85, q_bg=25)
    
    reconstructed = reconstruct_from_payload(payload)
    assert reconstructed.shape == phantom.shape
    assert reconstructed.dtype == np.uint8


def test_find_comparable_uniform_quality():
    phantom, _, _ = generate_synthetic_cxr_phantom(width=128, height=128, seed=42)
    target_bytes = 3500
    best_q, recon, comp_bytes, info = find_comparable_uniform_quality(phantom, target_bytes)
    
    assert 1 <= best_q <= 95
    assert recon.shape == phantom.shape
    assert len(comp_bytes) > 0
    # Difference should be relatively small
    assert abs(len(comp_bytes) - target_bytes) < 2000


def test_mask_shape_mismatch_raises():
    img = np.zeros((64, 64), dtype=np.uint8)
    bad_mask = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ROICompressionError):
        compress_roi_aware(img, bad_mask)
