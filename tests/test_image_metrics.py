"""test_image_metrics.py: Unit tests for image quality metrics module."""

import math
import numpy as np
import pytest

from src.image_metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    evaluate_quality_metrics,
    export_metrics_to_csv,
    MetricComputationError
)


def test_calculate_mse_identical():
    img = np.full((128, 128), 100, dtype=np.uint8)
    assert calculate_mse(img, img) == 0.0


def test_calculate_mse_known_difference():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img2 = np.full((10, 10), 5, dtype=np.uint8)
    assert calculate_mse(img1, img2) == 25.0


def test_calculate_mse_masked():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img2 = np.full((10, 10), 10, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, 0:5] = 255  # 25 pixels active
    # Modify img2 outside mask to 20
    img2[5:10, 5:10] = 20
    # Over mask, difference is 10 -> squared diff is 100
    assert calculate_mse(img1, img2, mask=mask) == 100.0


def test_calculate_mse_empty_mask():
    img = np.full((10, 10), 100, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert math.isnan(calculate_mse(img, img, mask=mask))


def test_calculate_psnr_identical():
    img = np.full((64, 64), 120, dtype=np.uint8)
    assert calculate_psnr(img, img) == float("inf")


def test_calculate_psnr_empty_mask():
    img1 = np.zeros((32, 32), dtype=np.uint8)
    img2 = np.ones((32, 32), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    assert math.isnan(calculate_psnr(img1, img2, mask=mask))


def test_calculate_ssim_identical():
    img = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    ssim_val = calculate_ssim(img, img)
    assert pytest.approx(ssim_val, rel=1e-3) == 1.0


def test_calculate_ssim_degraded():
    img1 = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)
    img2 = np.clip(img1.astype(int) + 30, 0, 255).astype(np.uint8)
    ssim_val = calculate_ssim(img1, img2)
    assert 0.0 <= ssim_val < 1.0


def test_evaluate_quality_metrics_structure():
    img1 = np.full((64, 64), 100, dtype=np.uint8)
    img2 = np.full((64, 64), 105, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    
    results = evaluate_quality_metrics(
        img1, img2, mask=mask, raw_bytes=4096, comp_bytes=1024, label="test"
    )
    assert "psnr_global_db" in results
    assert "ssim_global" in results
    assert "psnr_roi_db" in results
    assert "psnr_bg_db" in results
    assert results["compression_ratio"] == 4.0
    assert results["size_reduction_pct"] == 75.0


def test_shape_mismatch_raises():
    img1 = np.zeros((64, 64), dtype=np.uint8)
    img2 = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(MetricComputationError):
        calculate_mse(img1, img2)


def test_csv_export(tmp_path):
    out_csv = tmp_path / "metrics.csv"
    data = [{
        "label": "demo",
        "image_shape": "64x64",
        "total_pixels": 4096,
        "roi_pixels": 400,
        "bg_pixels": 3696,
        "roi_pixel_fraction": 0.097,
        "raw_bytes": 4096,
        "compressed_bytes": 1024,
        "compression_ratio": 4.0,
        "size_reduction_pct": 75.0,
        "psnr_global_db": 34.2,
        "ssim_global": 0.95,
        "psnr_roi_db": 38.5,
        "ssim_roi": 0.98,
        "psnr_bg_db": 33.1,
        "ssim_bg": 0.94,
    }]
    export_metrics_to_csv(data, out_csv)
    assert out_csv.exists()
    assert out_csv.stat().st_size > 0
