"""
metrics_service.py
------------------
Service adapter for image-quality metric computation.

Provides thin wrappers around the existing ``compute_ssim()`` and
``compute_psnr()`` functions from ``src/ml_core/evaluation/metrics.py``,
translating infinite PSNR (identical images) into ``None`` for clean JSON
serialisation.
"""

import logging
import math
from typing import Optional

import numpy as np
from fastapi import HTTPException, status

from src.ml_core.evaluation.metrics import compute_ssim, compute_psnr

logger = logging.getLogger(__name__)


def evaluate_metrics(
    original_rgb: np.ndarray,
    cloaked_rgb: np.ndarray,
) -> tuple[float, Optional[float]]:
    """Compute SSIM and PSNR between the original and cloaked images.

    Wraps ``compute_ssim()`` and ``compute_psnr()`` with HTTP error handling.
    Converts ``float('inf')`` PSNR (identical images) to ``None`` for clean
    JSON serialisation — ``inf`` is not valid JSON.

    Args:
        original_rgb: Full-resolution original image as ``(H, W, 3)`` uint8
            RGB NumPy array.
        cloaked_rgb: Full-resolution cloaked image with the same shape and
            dtype as ``original_rgb``.

    Returns:
        A two-element tuple ``(ssim_score, psnr_db)``:

        - ``ssim_score``: SSIM value in ``[-1.0, 1.0]``.
        - ``psnr_db``: PSNR in dB, or ``None`` if images are identical.

    Raises:
        HTTPException(500, PROCESSING_ERROR): If metric computation raises
            an unexpected exception.
    """
    logger.info("metrics_service — computing SSIM and PSNR …")

    try:
        ssim_score: float = compute_ssim(original_rgb, cloaked_rgb)
        raw_psnr: float = compute_psnr(original_rgb, cloaked_rgb)
    except (TypeError, ValueError) as exc:
        logger.error("metrics_service — metric computation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error": "Image quality metric computation failed.",
            },
        ) from exc

    # ``float('inf')`` is not valid JSON — return None for the identical-image
    # edge case.  Clients should interpret None as "infinite PSNR".
    psnr_db: Optional[float] = None if math.isinf(raw_psnr) else raw_psnr

    logger.info(
        "metrics_service — SSIM: %.6f | PSNR: %s dB",
        ssim_score,
        f"{psnr_db:.4f}" if psnr_db is not None else "inf",
    )
    return ssim_score, psnr_db
