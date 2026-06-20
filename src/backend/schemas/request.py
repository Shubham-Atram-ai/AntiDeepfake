"""
request.py
----------
Pydantic request schemas for the AntiDeepfake backend API.

Defines the validated input models consumed by API route handlers.
FastAPI automatically enforces these constraints and surfaces validation
errors as ``422 Unprocessable Entity`` responses.
"""

from pydantic import BaseModel, Field


class CloakRequest(BaseModel):
    """Query parameters / form fields for the ``POST /api/v1/cloak`` endpoint.

    Note:
        The ``file`` field (the uploaded image) is handled separately as a
        FastAPI ``UploadFile`` parameter.  This schema covers only the
        optional numeric parameters.

    Attributes:
        epsilon: FGSM L-infinity perturbation budget.  Controls how much each
            pixel is allowed to change.  Must be in the range ``(0.0, 1.0]``.
            Lower values → more subtle changes, potentially less effective
            cloaking.  Higher values → stronger cloaking but visible noise.
            Defaults to ``0.02`` (2 % of the normalised pixel range).
    """

    epsilon: float = Field(
        default=0.02,
        gt=0.0,
        le=1.0,
        description=(
            "FGSM L-infinity perturbation budget in [0.0, 1.0]. "
            "Controls per-pixel change magnitude. Default: 0.02."
        ),
        examples=[0.02, 0.05, 0.1],
    )
