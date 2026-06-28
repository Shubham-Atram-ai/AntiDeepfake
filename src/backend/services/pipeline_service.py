"""
pipeline_service.py
-------------------
Orchestration layer for the AntiDeepfake PGD adversarial cloaking pipeline.

This service is the single authoritative entry point for processing an
uploaded image end-to-end.  It delegates to the three specialised service
modules and the ML model registry, keeping each concern isolated.

Pipeline flow (mirrors ``test_fgsm_pipeline.main()`` but runs entirely
in-memory — no disk I/O):

    Raw bytes (UploadFile)
          ↓
    cv2.imdecode()      — bytes → BGR array
          ↓
    BGR → RGB           — OpenCV default is BGR; ML models expect RGB
          ↓
    detect_face()       — RetinaFace → (face_crop, bounding_box)
          ↓
    prepare_face_tensor() — resize 160×160, normalise [0,1], CHW tensor
          ↓
    run_attack()        — PGDAttack.attack() → adversarial face uint8 RGB
          ↓
    _reconstruct_image() — paste adversarial face back into original canvas
          ↓
    evaluate_metrics()  — SSIM + PSNR (cloaked vs original)
          ↓
    cv2.imencode() + base64 — encode cloaked JPEG → Base64 string
          ↓
    CloakResponse dict  — returned to the route handler
"""

import base64
import logging
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import HTTPException, status

from src.backend.core.model_registry import registry
from src.backend.services.detector_service import detect_face
from src.backend.services.attack_service import prepare_face_tensor, run_attack
from src.backend.services.metrics_service import evaluate_metrics

logger = logging.getLogger(__name__)

# Supported MIME types / file extensions accepted by the /cloak endpoint.
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
}

# JPEG encoding quality for the Base64 output (1–100).
_JPEG_ENCODE_QUALITY: int = 95


def _decode_image_bytes(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Decode raw image bytes into a uint8 RGB NumPy array.

    Uses ``cv2.imdecode()`` for in-memory decoding (no temp files written).
    Converts the result from OpenCV's default BGR to RGB so that all
    downstream ML code receives the expected colour order.

    Args:
        raw_bytes: Raw bytes from the uploaded file.
        filename: Original filename string, used only for log messages.

    Returns:
        RGB uint8 NumPy array of shape ``(H, W, 3)``.

    Raises:
        HTTPException(400, INVALID_IMAGE): If ``cv2.imdecode()`` returns
            ``None`` (unsupported format, corrupt data, or non-image payload).
    """
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if bgr_image is None:
        logger.warning(
            "pipeline_service — cv2.imdecode returned None for '%s'. "
            "File may be corrupt or in an unsupported format.",
            filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error_code": "INVALID_IMAGE",
                "error": (
                    "Could not decode the uploaded file. "
                    "Supported formats: JPEG, PNG, BMP, WebP."
                ),
            },
        )

    rgb_image: np.ndarray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    logger.info(
        "pipeline_service — image decoded. File: '%s' | Shape: %s | dtype: %s",
        filename,
        rgb_image.shape,
        rgb_image.dtype,
    )
    return rgb_image


def _reconstruct_image(
    original_rgb: np.ndarray,
    adversarial_face_rgb: np.ndarray,
    bounding_box: List[float],
) -> np.ndarray:
    """Reinsert the adversarial face crop into the original image canvas.

    Mirrors ``test_fgsm_pipeline._reconstruct_image()`` exactly, including
    the double-clamping guard against out-of-bounds bounding box coordinates.

    Steps:
        1. Clamp bounding box to absolute canvas dimensions.
        2. Resize adversarial face to the original bounding box slot size
           using LANCZOS interpolation.
        3. Copy original image and paste the resized adversarial face in-place.

    Args:
        original_rgb: Full-resolution original RGB uint8 image.
        adversarial_face_rgb: Adversarial face as RGB uint8 array ``(H', W', 3)``.
        bounding_box: ``[x1, y1, x2, y2]`` from the detector (floats).

    Returns:
        Protected (cloaked) full-resolution RGB uint8 image with the same
        shape as ``original_rgb``.
    """
    logger.info("pipeline_service — reconstructing cloaked image …")

    img_h, img_w = original_rgb.shape[:2]

    # ── Defensive clamp (same guard as test_fgsm_pipeline._reconstruct_image) ─
    x1: int = max(0, min(int(bounding_box[0]), img_w - 1))
    y1: int = max(0, min(int(bounding_box[1]), img_h - 1))
    x2: int = max(x1 + 1, min(int(bounding_box[2]), img_w))
    y2: int = max(y1 + 1, min(int(bounding_box[3]), img_h))

    target_w = x2 - x1
    target_h = y2 - y1

    # Resize adversarial face back to original bounding box dimensions.
    adv_resized = cv2.resize(
        adversarial_face_rgb,
        (target_w, target_h),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # Copy original — never mutate the source array.
    cloaked_rgb = original_rgb.copy()
    cloaked_rgb[y1:y2, x1:x2] = adv_resized

    logger.info(
        "pipeline_service — reconstruction complete. "
        "Cloaked shape: %s | Face slot: [%d:%d, %d:%d]",
        cloaked_rgb.shape,
        y1, y2, x1, x2,
    )
    return cloaked_rgb


def _encode_image_base64(rgb_image: np.ndarray) -> str:
    """Encode a uint8 RGB image as a Base64 JPEG string.

    Converts RGB → BGR (required by ``cv2.imencode``), encodes as JPEG at
    95 % quality, then Base64-encodes the compressed bytes.

    Args:
        rgb_image: uint8 NumPy array of shape ``(H, W, 3)`` in RGB colour
            space.

    Returns:
        Base64-encoded JPEG string (UTF-8 decoded, no newlines).

    Raises:
        HTTPException(500, PROCESSING_ERROR): If ``cv2.imencode`` fails.
    """
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_ENCODE_QUALITY]
    success, buffer = cv2.imencode(".jpg", bgr_image, encode_params)

    if not success:
        logger.error("pipeline_service — cv2.imencode failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error": "Failed to encode the cloaked image.",
            },
        )

    b64_string = base64.b64encode(buffer.tobytes()).decode("utf-8")
    logger.info(
        "pipeline_service — image encoded to Base64 (JPEG %d%% quality). "
        "Payload size: %d chars",
        _JPEG_ENCODE_QUALITY,
        len(b64_string),
    )
    return b64_string


def validate_content_type(content_type: str, filename: str) -> None:
    """Raise HTTP 400 if the uploaded file's content type is not an image.

    Args:
        content_type: MIME type string from the ``UploadFile`` object
            (e.g. ``"image/jpeg"``).
        filename: Original filename for log messages.

    Raises:
        HTTPException(400, INVALID_IMAGE): If ``content_type`` is not in the
            set of allowed image MIME types.
    """
    normalised = (content_type or "").lower().split(";")[0].strip()
    if normalised not in _ALLOWED_CONTENT_TYPES:
        logger.warning(
            "pipeline_service — rejected file '%s' with content type '%s'.",
            filename,
            content_type,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error_code": "INVALID_IMAGE",
                "error": (
                    f"Unsupported file type '{content_type}'. "
                    "Please upload a JPEG, PNG, BMP, or WebP image."
                ),
            },
        )


def run_cloaking_pipeline(
    image_bytes: bytes,
    filename: str,
    epsilon: float,
) -> Dict[str, Any]:
    """Execute the full PGD cloaking pipeline and return a response dict.

    This is the single entry point called by the ``/api/v1/cloak`` route
    handler.  All business logic lives here; the route handler is kept thin.

    Pipeline sequence:
        1. Decode image bytes → RGB NumPy array.
        2. Run RetinaFace face detection → crop + bounding box.
        3. Prepare FaceNet tensor from the crop.
        4. Run PGD attack → adversarial face RGB array.
        5. Reconstruct adversarial face into the original canvas.
        6. Compute SSIM and PSNR.
        7. Encode cloaked image as Base64 JPEG.
        8. Build and return the response dictionary.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        filename: Original filename for logging (not used for disk I/O).
        epsilon: PGD perturbation budget for this request.

    Returns:
        A dictionary matching the ``CloakResponse`` schema:
        ``{success, processing_time_ms, metrics: {ssim, psnr}, cloaked_image_base64}``.

    Raises:
        HTTPException: Forwarded from any service layer call (400 or 500).
        HTTPException(503, SERVICE_UNAVAILABLE): If the model registry is not
            ready (models not loaded — indicates a startup failure).
    """
    if not registry.is_ready:
        logger.error("pipeline_service — model registry not ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "error_code": "SERVICE_UNAVAILABLE",
                "error": "ML models are not loaded. The service is not ready.",
            },
        )

    t_start = time.perf_counter()

    logger.info(
        "pipeline_service — starting cloaking pipeline | "
        "file='%s' | epsilon=%.4f",
        filename,
        epsilon,
    )

    # ── Step 1: Decode uploaded bytes → RGB array ─────────────────────────────
    original_rgb = _decode_image_bytes(image_bytes, filename)

    # ── Step 2: Face detection ────────────────────────────────────────────────
    face_crop, bounding_box = detect_face(registry.face_detector, original_rgb)

    # ── Step 3: Tensor preparation ────────────────────────────────────────────
    face_tensor = prepare_face_tensor(face_crop)

    # ── Step 4: PGD attack ────────────────────────────────────────────────────
    adversarial_face_rgb, _ = run_attack(registry.pgd_attack, face_tensor, epsilon)

    # ── Step 5: Reconstruct cloaked image ─────────────────────────────────────
    cloaked_rgb = _reconstruct_image(original_rgb, adversarial_face_rgb, bounding_box)

    # ── Step 6: Compute image quality metrics ─────────────────────────────────
    ssim_score, psnr_db = evaluate_metrics(original_rgb, cloaked_rgb)

    # ── Step 7: Encode cloaked image to Base64 ────────────────────────────────
    cloaked_b64 = _encode_image_base64(cloaked_rgb)

    # ── Step 8: Assemble response ─────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    logger.info(
        "pipeline_service — pipeline complete. "
        "SSIM=%.4f | PSNR=%s dB | Time=%.1f ms",
        ssim_score,
        f"{psnr_db:.4f}" if psnr_db is not None else "inf",
        elapsed_ms,
    )

    return {
        "success": True,
        "processing_time_ms": round(elapsed_ms, 2),
        "metrics": {
            "ssim": round(ssim_score, 6),
            "psnr": round(psnr_db, 4) if psnr_db is not None else None,
        },
        "cloaked_image_base64": cloaked_b64,
    }
