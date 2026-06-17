"""
test_metrics.py
---------------
Unit tests for the SSIM and PSNR evaluation metrics module.

Tests are fully self-contained and rely only on NumPy arrays — no disk I/O
or model loading required.  All edge cases (identical images, single-channel
arrays, shape mismatches, wrong dtypes) are covered.

Run from project root:
    python -m pytest tests/test_metrics.py -v
"""

import logging
import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml_core.evaluation.metrics import compute_ssim, compute_psnr  # noqa: E402

# ---------------------------------------------------------------------------
# Logging configuration for test runs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_H = 64
IMG_W = 64
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def identical_images() -> tuple[np.ndarray, np.ndarray]:
    """Return two identical uint8 RGB arrays for identity tests.

    Returns:
        Tuple of two identical ``(64, 64, 3)`` uint8 arrays.
    """
    rng = np.random.default_rng(RNG_SEED)
    img = rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
    return img, img.copy()


@pytest.fixture()
def different_images() -> tuple[np.ndarray, np.ndarray]:
    """Return two distinct uint8 RGB arrays for modification tests.

    The second image has uniform Gaussian noise added to simulate a mild
    adversarial perturbation.

    Returns:
        Tuple of two distinct ``(64, 64, 3)`` uint8 arrays.
    """
    rng = np.random.default_rng(RNG_SEED)
    img1 = rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
    # Add perturbation in the range ±20 (simulates ε ≈ 0.08 in [0,1] space)
    noise = rng.integers(-20, 21, (IMG_H, IMG_W, 3), dtype=np.int16)
    img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img1, img2


@pytest.fixture()
def small_perturbation_images() -> tuple[np.ndarray, np.ndarray]:
    """Return a pair where img2 has only tiny perturbations (ε ≈ 0.01).

    Returns:
        Tuple of ``(64, 64, 3)`` uint8 arrays with minimal difference.
    """
    rng = np.random.default_rng(RNG_SEED)
    img1 = rng.integers(10, 246, (IMG_H, IMG_W, 3), dtype=np.uint8)
    noise = rng.integers(-2, 3, (IMG_H, IMG_W, 3), dtype=np.int16)
    img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img1, img2


# ---------------------------------------------------------------------------
# SSIM Tests
# ---------------------------------------------------------------------------

class TestComputeSSIM:
    """Tests for ``compute_ssim()``."""

    def test_identical_images_return_one(
        self, identical_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """SSIM of identical images must be exactly 1.0."""
        img1, img2 = identical_images
        result = compute_ssim(img1, img2)
        assert result == pytest.approx(1.0, abs=1e-5), (
            f"Expected SSIM ≈ 1.0 for identical images, got {result:.6f}."
        )

    def test_different_images_return_less_than_one(
        self, different_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """SSIM of different images must be strictly less than 1.0."""
        img1, img2 = different_images
        result = compute_ssim(img1, img2)
        assert result < 1.0, (
            f"Expected SSIM < 1.0 for different images, got {result:.6f}."
        )

    def test_ssim_within_valid_range(
        self, different_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """SSIM must be in the valid range [-1.0, 1.0]."""
        img1, img2 = different_images
        result = compute_ssim(img1, img2)
        assert -1.0 <= result <= 1.0, (
            f"SSIM {result:.6f} is outside the valid range [-1.0, 1.0]."
        )

    def test_ssim_is_float(
        self, identical_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Return type must be Python float."""
        img1, img2 = identical_images
        result = compute_ssim(img1, img2)
        assert isinstance(result, float), (
            f"Expected float, got {type(result).__name__}."
        )

    def test_small_perturbation_yields_high_ssim(
        self, small_perturbation_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A tiny perturbation (±2 counts) should yield SSIM near 1.0."""
        img1, img2 = small_perturbation_images
        result = compute_ssim(img1, img2)
        assert result > 0.90, (
            f"Expected SSIM > 0.90 for small perturbation, got {result:.6f}."
        )

    # --- Validation / error handling ---

    def test_shape_mismatch_raises_value_error(self) -> None:
        """Mismatched shapes must raise ValueError."""
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="identical shapes"):
            compute_ssim(img1, img2)

    def test_single_channel_raises_value_error(self) -> None:
        """Grayscale (single-channel) input must raise ValueError."""
        img1 = np.zeros((64, 64, 1), dtype=np.uint8)
        img2 = np.zeros((64, 64, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            compute_ssim(img1, img2)

    def test_wrong_dtype_raises_value_error(self) -> None:
        """Float32 arrays (instead of uint8) must raise ValueError."""
        img1 = np.zeros((64, 64, 3), dtype=np.float32)
        img2 = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            compute_ssim(img1, img2)

    def test_non_array_raises_type_error(self) -> None:
        """Non-ndarray input must raise TypeError."""
        with pytest.raises(TypeError, match="NumPy ndarray"):
            compute_ssim([[1, 2, 3]], np.zeros((64, 64, 3), dtype=np.uint8))  # type: ignore[arg-type]

    def test_2d_array_raises_value_error(self) -> None:
        """A 2-D array (no channel axis) must raise ValueError."""
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            compute_ssim(img1, img2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PSNR Tests
# ---------------------------------------------------------------------------

class TestComputePSNR:
    """Tests for ``compute_psnr()``."""

    def test_identical_images_return_inf(
        self, identical_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PSNR of identical images must be float('inf')."""
        img1, img2 = identical_images
        result = compute_psnr(img1, img2)
        assert math.isinf(result) and result > 0, (
            f"Expected inf for identical images, got {result}."
        )

    def test_different_images_return_finite(
        self, different_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PSNR of different images must be a finite float."""
        img1, img2 = different_images
        result = compute_psnr(img1, img2)
        assert math.isfinite(result), (
            f"Expected finite PSNR for different images, got {result}."
        )

    def test_psnr_is_positive_for_different_images(
        self, different_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PSNR must be positive (MSE is always non-negative)."""
        img1, img2 = different_images
        result = compute_psnr(img1, img2)
        assert result > 0.0, (
            f"Expected positive PSNR, got {result:.4f}."
        )

    def test_psnr_is_float(
        self, different_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Return type must be Python float."""
        img1, img2 = different_images
        result = compute_psnr(img1, img2)
        assert isinstance(result, float), (
            f"Expected float, got {type(result).__name__}."
        )

    def test_small_perturbation_yields_high_psnr(
        self, small_perturbation_images: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A ±2-count perturbation should yield PSNR well above 40 dB."""
        img1, img2 = small_perturbation_images
        result = compute_psnr(img1, img2)
        assert result > 40.0, (
            f"Expected PSNR > 40 dB for small perturbation, got {result:.2f} dB."
        )

    def test_psnr_decreases_with_more_noise(self) -> None:
        """A heavily-noised image must yield lower PSNR than a slightly-noised one."""
        rng = np.random.default_rng(RNG_SEED)
        base = rng.integers(10, 246, (IMG_H, IMG_W, 3), dtype=np.uint8)

        small_noise = rng.integers(-2, 3, (IMG_H, IMG_W, 3), dtype=np.int16)
        large_noise = rng.integers(-50, 51, (IMG_H, IMG_W, 3), dtype=np.int16)

        img_small = np.clip(base.astype(np.int16) + small_noise, 0, 255).astype(np.uint8)
        img_large = np.clip(base.astype(np.int16) + large_noise, 0, 255).astype(np.uint8)

        psnr_small = compute_psnr(base, img_small)
        psnr_large = compute_psnr(base, img_large)

        assert psnr_small > psnr_large, (
            f"Expected higher PSNR for smaller noise "
            f"(small: {psnr_small:.2f} dB, large: {psnr_large:.2f} dB)."
        )

    # --- Validation / error handling ---

    def test_shape_mismatch_raises_value_error(self) -> None:
        """Mismatched shapes must raise ValueError."""
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="identical shapes"):
            compute_psnr(img1, img2)

    def test_single_channel_raises_value_error(self) -> None:
        """Grayscale (single-channel) input must raise ValueError."""
        img1 = np.zeros((64, 64, 1), dtype=np.uint8)
        img2 = np.zeros((64, 64, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            compute_psnr(img1, img2)

    def test_wrong_dtype_raises_value_error(self) -> None:
        """Float32 arrays must raise ValueError (expects uint8)."""
        img1 = np.zeros((64, 64, 3), dtype=np.float32)
        img2 = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            compute_psnr(img1, img2)

    def test_non_array_raises_type_error(self) -> None:
        """Non-ndarray input must raise TypeError."""
        with pytest.raises(TypeError, match="NumPy ndarray"):
            compute_psnr(None, np.zeros((64, 64, 3), dtype=np.uint8))  # type: ignore[arg-type]

    def test_2d_array_raises_value_error(self) -> None:
        """A 2-D array must raise ValueError."""
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            compute_psnr(img1, img2)  # type: ignore[arg-type]
