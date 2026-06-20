"""
detector_service.py
-------------------
Service adapter for face detection.

Provides a thin, testable wrapper around ``FaceDetector.detect_and_crop()``
that translates its output into API-friendly types and raises structured
``HTTPException`` on failure instead of returning ``None`` values.

The underlying ``FaceDetector`` is not instantiated here — it is retrieved
from the application-level ``ModelRegistry`` to ensure models are loaded
only once at startup.
"""

import logging
from typing import Tuple, List

import numpy as np
from fastapi import HTTPException, status

from src.ml_core.models.mtcnn_detector import FaceDetector

logger = logging.getLogger(__name__)


def detect_face(
    detector: FaceDetector,
    image_rgb: np.ndarray,
) -> Tuple[np.ndarray, List[float]]:
    """Run MTCNN face detection and return the crop and bounding box.

    Wraps ``FaceDetector.detect_and_crop()`` with structured HTTP error
    handling so route handlers remain thin and exception-free.

    Args:
        detector: Initialised ``FaceDetector`` instance from the model
            registry.  Must not be ``None``.
        image_rgb: Full-resolution RGB image as a ``(H, W, 3)`` uint8
            NumPy array — exactly as produced by ``cv2.imdecode()`` after
            BGR→RGB conversion.

    Returns:
        A two-element tuple ``(face_crop, bounding_box)``:

        - ``face_crop``: Cropped face region as ``(h, w, 3)`` uint8 RGB
          array.
        - ``bounding_box``: ``[x1, y1, x2, y2]`` bounding box coordinates
          as floats, clamped to image boundaries.

    Raises:
        HTTPException(400, NO_FACE_DETECTED): If MTCNN detects no face in
            the provided image.
        HTTPException(500, PROCESSING_ERROR): If ``FaceDetector`` raises
            an unexpected ``ValueError``.
    """
    logger.info(
        "detector_service — running face detection on image of shape %s",
        image_rgb.shape,
    )

    try:
        face_crop, bounding_box = detector.detect_and_crop(image_rgb)
    except ValueError as exc:
        logger.error("detector_service — FaceDetector raised ValueError: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error": "Face detection encountered an unexpected error.",
            },
        ) from exc

    if face_crop is None or bounding_box is None:
        logger.warning("detector_service — no face detected in uploaded image.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error_code": "NO_FACE_DETECTED",
                "error": (
                    "No face detected in the uploaded image. "
                    "Please upload a clear, forward-facing face photo."
                ),
            },
        )

    logger.info(
        "detector_service — face detected. Bounding box: %s | Crop shape: %s",
        [f"{v:.1f}" for v in bounding_box],
        face_crop.shape,
    )
    return face_crop, bounding_box
