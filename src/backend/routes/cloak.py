"""
cloak.py
--------
FastAPI router for the adversarial cloaking endpoint.

Mounts under the ``/api/v1`` prefix (configured in ``main.py``).

All business logic is delegated to ``pipeline_service.run_cloaking_pipeline()``;
this module is intentionally kept thin — it handles only HTTP concerns
(parameter extraction, content-type validation, error forwarding).
"""

import logging

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from src.backend.schemas.response import CloakResponse, ErrorResponse
from src.backend.services.pipeline_service import (
    run_cloaking_pipeline,
    validate_content_type,
)
from src.backend.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/cloak",
    tags=["Cloaking"],
)


@router.post(
    "",
    response_model=CloakResponse,
    status_code=status.HTTP_200_OK,
    summary="Apply FGSM adversarial cloaking to a face image",
    description=(
        "Upload a face image to apply an FGSM adversarial perturbation that "
        "confuses automated face-recognition systems while remaining "
        "visually imperceptible to humans.  "
        "Returns the cloaked image as a Base64-encoded JPEG along with "
        "SSIM and PSNR quality metrics."
    ),
    responses={
        200: {
            "description": "Cloaking successful.",
            "model": CloakResponse,
        },
        400: {
            "description": "Invalid image or no face detected.",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "no_face": {
                            "summary": "No face detected",
                            "value": {
                                "success": False,
                                "error_code": "NO_FACE_DETECTED",
                                "error": (
                                    "No face detected in the uploaded image. "
                                    "Please upload a clear, forward-facing face photo."
                                ),
                            },
                        },
                        "invalid_image": {
                            "summary": "Invalid image format",
                            "value": {
                                "success": False,
                                "error_code": "INVALID_IMAGE",
                                "error": "Unsupported file type. Please upload JPEG, PNG, BMP, or WebP.",
                            },
                        },
                    }
                }
            },
        },
        500: {
            "description": "Internal processing error.",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error_code": "PROCESSING_ERROR",
                        "error": "FGSM attack failed during adversarial perturbation.",
                    }
                }
            },
        },
    },
)
async def cloak_image(
    file: UploadFile = File(
        ...,
        description="Face image to cloak. Accepted formats: JPEG, PNG, BMP, WebP.",
    ),
    epsilon: float = Form(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "FGSM L-infinity perturbation budget in (0.0, 1.0]. "
            f"Defaults to {settings.default_epsilon} when not provided."
        ),
    ),
) -> JSONResponse:
    """Apply FGSM adversarial cloaking to the uploaded face image.

    **Workflow**:
    1. Validate content type (JPEG / PNG / BMP / WebP).
    2. Read raw bytes from the upload.
    3. Delegate to ``pipeline_service.run_cloaking_pipeline()``.
    4. Return a JSON response containing:
       - ``success``: ``true``
       - ``processing_time_ms``: wall-clock time in milliseconds
       - ``metrics``: ``{ssim, psnr}`` image quality scores
       - ``cloaked_image_base64``: Base64-encoded JPEG of the cloaked image

    **epsilon** controls the strength of the adversarial perturbation:
    - ``0.02`` (default) — subtle, typically imperceptible noise
    - ``0.05`` — moderate, usually invisible but stronger protection
    - ``0.10`` — strong protection, may introduce barely-visible texture

    Args:
        file: Uploaded image file (multipart/form-data).
        epsilon: Optional FGSM perturbation strength override.

    Returns:
        ``JSONResponse`` with ``CloakResponse`` body on success, or an
        ``ErrorResponse`` body on validation / processing failure.
    """
    logger.info(
        "cloak — request received | filename='%s' | content_type='%s' | epsilon=%s",
        file.filename,
        file.content_type,
        epsilon,
    )

    # Use the configured default if epsilon was not supplied in the form.
    effective_epsilon: float = epsilon if epsilon is not None else settings.default_epsilon

    # ── Content-type guard ────────────────────────────────────────────────────
    validate_content_type(file.content_type or "", file.filename or "unknown")

    # ── Read upload bytes ─────────────────────────────────────────────────────
    image_bytes: bytes = await file.read()

    if not image_bytes:
        logger.warning("cloak — uploaded file is empty: '%s'", file.filename)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error_code": "INVALID_IMAGE",
                "error": "Uploaded file is empty.",
            },
        )

    logger.info(
        "cloak — image bytes read. Size: %d bytes | epsilon=%.4f",
        len(image_bytes),
        effective_epsilon,
    )

    # ── Run pipeline (HTTPException propagates automatically) ─────────────────
    result = run_cloaking_pipeline(
        image_bytes=image_bytes,
        filename=file.filename or "upload",
        epsilon=effective_epsilon,
    )

    logger.info(
        "cloak — response generated. Processing time: %.1f ms",
        result["processing_time_ms"],
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=result)
