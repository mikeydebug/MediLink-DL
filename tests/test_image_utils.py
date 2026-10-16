"""test_image_utils.py: Unit tests for image utilities, I/O, and synthetic generator."""

from pathlib import Path
import numpy as np
import pytest

from src.image_utils import (
    validate_image,
    load_grayscale_image,
    save_image,
    generate_synthetic_cxr_phantom,
    create_rectangular_roi_mask,
    InvalidImageFormatError,
    ImageCorruptedError
)


def test_validate_image_valid():
    img = np.zeros((128, 128), dtype=np.uint8)
    validate_image(img)  # Should not raise


def test_validate_image_invalid_dimensions():
    # 3D RGB array
    img_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageFormatError):
        validate_image(img_rgb)


def test_validate_image_invalid_dtype():
    img_float = np.zeros((128, 128), dtype=np.float32)
    with pytest.raises(InvalidImageFormatError):
        validate_image(img_float)


def test_validate_image_too_small():
    img_small = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(InvalidImageFormatError):
        validate_image(img_small, min_dim=32)


def test_synthetic_phantom_generation():
    phantom, mask, meta = generate_synthetic_cxr_phantom(
        width=256, height=256, seed=42, add_lesion=True
    )
    assert phantom.shape == (256, 256)
    assert phantom.dtype == np.uint8
    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert meta["is_synthetic"] is True
    assert meta["has_lesion"] is True
    assert meta["roi_pixel_count"] > 0
    # Check that mask has values 0 or 255
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})


def test_rectangular_roi_mask():
    mask = create_rectangular_roi_mask((100, 100), (10, 20, 50, 60))
    assert mask.shape == (100, 100)
    assert np.count_nonzero(mask) == (50 - 10) * (60 - 20)
    assert mask[25, 35] == 255
    assert mask[5, 5] == 0


def test_save_and_load_roundtrip(tmp_path):
    img = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)
    save_path = tmp_path / "test_cxr.png"
    save_image(img, save_path)
    assert save_path.exists()
    
    loaded = load_grayscale_image(save_path)
    assert loaded.shape == img.shape
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, img)


def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_grayscale_image("/nonexistent/path/to/image.png")


def test_load_corrupted_file_raises(tmp_path):
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"NOT_A_VALID_PNG_FILE_DATA")
    with pytest.raises(ImageCorruptedError):
        load_grayscale_image(corrupt_file)
