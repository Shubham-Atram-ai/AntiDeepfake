"""
response.py
-----------
Pydantic response schemas for the AntiDeepfake backend API.

All JSON responses are validated against these models before being sent to
the client, ensuring a stable, documented contract for future frontend
(React) integration.
"""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for ``GET /health``.

    Attributes:
        status: Overall API health.  ``"healthy"`` when all models are loaded.
        version: Semantic version string of the running API.
        face_detector_loaded: ``True`` when RetinaFace is ready.
        pgd_engine_loaded: ``True`` when ResNet50 is ready.
    """

    status: str = Field(
        ...,
        description="Overall API health status.",
        examples=["healthy"],
    )
    version: str = Field(
        ...,
        description="API semantic version string.",
        examples=["1.0.0"],
    )
    face_detector_loaded: bool = Field(
        ...,
        description="Whether the RetinaFace face detector is loaded and ready.",
    )
    pgd_engine_loaded: bool = Field(
        ...,
        description="Whether the PGD ResNet50 engine is loaded and ready.",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "status": "healthy",
            "version": "1.0.0",
            "face_detector_loaded": True,
            "pgd_engine_loaded": True,
        }]
    }}


class MetricsResponse(BaseModel):
    """Nested image-quality and protection metrics returned inside ``CloakResponse``.

    Attributes:
        ssim: Structural Similarity Index in ``[-1.0, 1.0]``.  Values ≥ 0.90
            are typically imperceptible to humans.
        psnr: Peak Signal-to-Noise Ratio in decibels.  Values > 40 dB are
            practically indistinguishable from the original; ``null`` is
            returned for identical images (infinite PSNR).
        cosine_similarity: Cosine similarity between the original and
            adversarial FaceNet embeddings in ``[-1.0, 1.0]``.  Values near
            1.0 indicate the face identity is unchanged (weak protection);
            values near 0.0 or below indicate strong embedding divergence
            (strong protection).
        detection_probability: Estimated probability (0–100 %) that an AI
            face-recognition system will still correctly identify the cloaked
            face.  Derived linearly from ``cosine_similarity``.
    """

    ssim: float = Field(
        ...,
        description="SSIM score in [-1.0, 1.0]. Values ≥ 0.90 indicate imperceptible perturbation.",
        examples=[0.9412],
    )
    psnr: Optional[float] = Field(
        ...,
        description="PSNR in dB. > 40 dB ≈ imperceptible. null for identical images (inf).",
        examples=[36.14],
    )
    cosine_similarity: float = Field(
        ...,
        description=(
            "Cosine similarity between original and adversarial FaceNet embeddings. "
            "Range [-1.0, 1.0]. Near 1.0 = same identity (weak protection); ≤ 0.5 = strong protection."
        ),
        examples=[0.1823],
    )
    detection_probability: float = Field(
        ...,
        description=(
            "Estimated probability (0–100 %) that AI will still identify the face. "
            "0 % = fully protected; 100 % = unprotected."
        ),
        examples=[18.23],
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "ssim": 0.9412,
            "psnr": 36.14,
            "cosine_similarity": 0.1823,
            "detection_probability": 18.23,
        }]
    }}


class CloakResponse(BaseModel):
    """Success response model for ``POST /api/v1/cloak``.

    Attributes:
        success: Always ``True`` on a successful cloaking request.
        processing_time_ms: Wall-clock processing time in milliseconds,
            covering decode → detect → attack → reconstruct → encode.
        metrics: Nested ``MetricsResponse`` with SSIM, PSNR, cosine similarity,
            and detection probability.
        cloaked_image_base64: The cloaked full-resolution image encoded as a
            Base64 string (JPEG format).  Decode with ``base64.b64decode()``.
    """

    success: bool = Field(
        default=True,
        description="Indicates successful processing.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total server-side processing time in milliseconds.",
        examples=[742.1],
    )
    metrics: MetricsResponse = Field(
        ...,
        description="Image quality and protection metrics comparing original vs cloaked image.",
    )
    cloaked_image_base64: str = Field(
        ...,
        description=(
            "Base64-encoded JPEG of the cloaked image. "
            "Decode with base64.b64decode() or display via a data URI."
        ),
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "success": True,
            "processing_time_ms": 742.1,
            "metrics": {
                "ssim": 0.9412,
                "psnr": 36.14,
                "cosine_similarity": 0.1823,
                "detection_probability": 18.23,
            },
            "cloaked_image_base64": "<base64-encoded-jpeg>",
        }]
    }}


class ErrorResponse(BaseModel):
    """Structured error response model for all failure cases.

    Attributes:
        success: Always ``False`` on an error response.
        error_code: Machine-readable uppercase error code for programmatic
            handling by frontend clients.
        error: Human-readable description of the error.
    """

    success: bool = Field(
        default=False,
        description="Always False for error responses.",
    )
    error_code: str = Field(
        ...,
        description="Machine-readable uppercase error code.",
        examples=["NO_FACE_DETECTED", "INVALID_IMAGE", "PROCESSING_ERROR"],
    )
    error: str = Field(
        ...,
        description="Human-readable error description.",
        examples=["No face detected in the uploaded image."],
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "success": False,
            "error_code": "NO_FACE_DETECTED",
            "error": "No face detected in the uploaded image.",
        }]
    }}
