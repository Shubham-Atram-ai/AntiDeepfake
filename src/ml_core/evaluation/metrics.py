"""
metrics.py
----------
Image quality evaluation metrics for adversarial perturbation analysis.

Provides two standard metrics used to quantify the perceptual and
signal-level differences between an original image and its adversarially
perturbed counterpart:

* **SSIM** (Structural Similarity Index Measure): Perceptually motivated
  metric that compares luminance, contrast, and structure between two images.
  Range [−1, 1]; value of 1.0 indicates identical images.

* **PSNR** (Peak Signal-to-Noise Ratio): Signal-level metric expressed in
  decibels.  Higher values indicate less distortion; a value of ``inf``
  indicates perfectly identical images.

Both functions operate on RGB uint8 arrays (shape ``(H, W, 3)``) as produced
by the project's ``image_loader.load_image()`` utility.
"""

import logging
import math
from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal validation helper
# ---------------------------------------------------------------------------

def _validate_images(
    img1: np.ndarray,
    img2: np.ndarray,
    caller: str = "metric",
) -> None:
    """Validate that two image arrays are compatible for metric computation.

    Args:
        img1: First image array.
        img2: Second image array.
        caller: Name of the calling function, used in error messages.

    Raises:
        ValueError: If either array is not a 3-channel uint8 array, or if
            the arrays have different shapes.
        TypeError: If either argument is not a NumPy ndarray.
    """
    for name, img in (("img1", img1), ("img2", img2)):
        if not isinstance(img, np.ndarray):
            raise TypeError(
                f"[{caller}] '{name}' must be a NumPy ndarray, "
                f"got {type(img).__name__}."
            )
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"[{caller}] '{name}' must have shape (H, W, 3), "
                f"got {img.shape}."
            )
        if img.dtype != np.uint8:
            raise ValueError(
                f"[{caller}] '{name}' must have dtype uint8, "
                f"got {img.dtype}. "
                "Convert with: array.astype(np.uint8)"
            )

    if img1.shape != img2.shape:
        raise ValueError(
            f"[{caller}] img1 and img2 must have identical shapes. "
            f"Got img1={img1.shape}, img2={img2.shape}."
        )


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute the Structural Similarity Index (SSIM) between two RGB images.

    SSIM is a perceptually-motivated image quality metric that models the
    degradation of structural information.  It computes a weighted combination
    of luminance, contrast, and structural similarity across local windows.

    A value of ``1.0`` means the images are visually identical.  Values close
    to ``1.0`` (e.g. ≥ 0.90) indicate that the adversarial perturbation is
    likely imperceptible to the human visual system.

    Uses ``skimage.metrics.structural_similarity`` with:
    - ``channel_axis=2`` for RGB (not channel-first).
    - ``data_range=255`` for uint8 images.
    - ``win_size=7`` for the sliding window (default scikit-image value).

    Args:
        img1: Original image as a NumPy array of shape ``(H, W, 3)`` with
            dtype ``uint8`` in RGB colour space.
        img2: Comparison image (e.g. adversarially perturbed) with the same
            shape and dtype as ``img1``.

    Returns:
        SSIM score as a float in ``[-1.0, 1.0]``.  Values near ``1.0``
        indicate high visual similarity.

    Raises:
        TypeError: If either input is not a NumPy ndarray.
        ValueError: If inputs do not have shape ``(H, W, 3)``, dtype
            ``uint8``, or if they have different shapes.

    Example:
        >>> ssim = compute_ssim(original_rgb, adversarial_rgb)
        >>> print(f"SSIM: {ssim:.4f}")
    """
    _validate_images(img1, img2, caller="compute_ssim")

    ssim_value: float = float(
        _ssim(
            img1,
            img2,
            data_range=255,
            channel_axis=2,
            win_size=7,
        )
    )

    logger.info(
        "SSIM computed: %.6f  "
        "(1.0 = identical; ≥ 0.90 typically imperceptible to humans)",
        ssim_value,
    )
    return ssim_value


def compute_psnr(
    img1: np.ndarray,
    img2: np.ndarray,
) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two RGB images.

    PSNR is defined as:

    .. math::

        \\text{PSNR} = 10 \\cdot \\log_{10}\\!\\left(
            \\frac{\\text{MAX}^2}{\\text{MSE}}
        \\right)

    where ``MAX = 255`` for uint8 images and ``MSE`` is the mean squared
    error between the two images.

    Higher PSNR indicates better signal quality (less distortion).  Typical
    thresholds for adversarial images:

    * ``> 40 dB`` — practically indistinguishable from the original.
    * ``30 – 40 dB`` — minor visible noise in close inspection.
    * ``< 30 dB`` — noticeable distortion.

    Identical images produce infinite PSNR (MSE = 0); this case is handled
    explicitly and returns ``float('inf')``.

    Uses ``skimage.metrics.peak_signal_noise_ratio`` with
    ``data_range=255``.

    Args:
        img1: Original image as a NumPy array of shape ``(H, W, 3)`` with
            dtype ``uint8`` in RGB colour space.
        img2: Comparison image (e.g. adversarially perturbed) with the same
            shape and dtype as ``img1``.

    Returns:
        PSNR value in decibels (dB) as a float.  Returns ``float('inf')``
        if the images are identical.

    Raises:
        TypeError: If either input is not a NumPy ndarray.
        ValueError: If inputs do not have shape ``(H, W, 3)``, dtype
            ``uint8``, or if they have different shapes.

    Example:
        >>> psnr = compute_psnr(original_rgb, adversarial_rgb)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    _validate_images(img1, img2, caller="compute_psnr")

    # Detect identical images before delegating to skimage to avoid
    # division-by-zero warnings and return a meaningful value.
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0.0:
        logger.warning(
            "PSNR: Images are identical (MSE = 0). Returning inf dB. "
            "This is expected when comparing an image to itself."
        )
        return float("inf")

    psnr_value: float = float(
        _psnr(img1, img2, data_range=255)
    )

    logger.info(
        "PSNR computed: %.4f dB  "
        "(MSE: %.6f | > 40 dB ≈ imperceptible; 30–40 dB ≈ minor noise)",
        psnr_value,
        mse,
    )
    return psnr_value
