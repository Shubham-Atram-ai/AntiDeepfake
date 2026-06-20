"""
main.py
-------
FastAPI application entry point for the AntiDeepfake Backend V1.

This module:
    - Creates the ``FastAPI`` application instance with full metadata.
    - Configures the ``lifespan`` context manager for model loading/unloading.
    - Adds ``CORSMiddleware`` for cross-origin frontend access.
    - Registers root (``GET /``) and health (``GET /health``) endpoints.
    - Mounts the versioned ``/api/v1`` router from ``routes/cloak.py``.

Run the server:
    uvicorn src.backend.main:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs

ReDoc:
    http://127.0.0.1:8000/redoc
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.backend.core.config import settings
from src.backend.core.model_registry import registry
from src.backend.routes.cloak import router as cloak_router
from src.backend.schemas.response import HealthResponse

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — model loading and teardown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage the application startup and shutdown lifecycle.

    **Startup**:
        1. Call ``registry.load()`` which initialises ``FaceDetector`` and
           ``FGSMAttack``.  Both models are cached in the module-level
           ``registry`` singleton and reused across all requests.
        2. If either model fails to load (e.g. missing weights), the
           application exits immediately — a partially-initialised backend
           is not useful.

    **Shutdown**:
        1. Call ``registry.clear()`` to release model references so the
           garbage collector can reclaim memory cleanly.

    Args:
        app: The running ``FastAPI`` application instance (unused directly,
             but required by the lifespan signature).

    Yields:
        Control to the running application between startup and shutdown.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("AntiDeepfake API — starting up …")
    logger.info("Version : %s", settings.app_version)
    logger.info("=" * 60)

    try:
        registry.load()
    except Exception as exc:  # pragma: no cover
        logger.critical(
            "FATAL — model loading failed. The API cannot start. Error: %s", exc
        )
        sys.exit(1)

    logger.info("Startup complete — all models loaded. API is ready.")
    logger.info("Swagger UI: http://%s:%d/docs", settings.host, settings.port)

    yield  # ── Application runs here ─────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("AntiDeepfake API — shutting down …")
    registry.clear()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "System",
            "description": "Root and informational endpoints.",
        },
        {
            "name": "Health",
            "description": (
                "API and model health checks.  Use these to verify that "
                "all ML models loaded successfully before sending image requests."
            ),
        },
        {
            "name": "Cloaking",
            "description": (
                "Adversarial image cloaking endpoints.  Upload a face image "
                "to receive a perturbed version that confuses face-recognition "
                "systems while remaining visually imperceptible."
            ),
        },
    ],
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["System"],
    summary="API root",
    description="Returns a simple message confirming the API is running.",
    response_description="Basic API status message.",
)
async def root() -> JSONResponse:
    """Return a greeting confirming the API is reachable.

    Returns:
        JSON object: ``{"message": "AntiDeepfake API Running"}``.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "AntiDeepfake API Running"},
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Health"],
    response_model=HealthResponse,
    summary="API health check",
    description=(
        "Returns the current health status of the API and indicates whether "
        "both ML models (MTCNN face detector and FGSM engine) are loaded and ready."
    ),
    responses={
        200: {"description": "API is healthy and all models are loaded."},
        503: {"description": "API is running but models are not loaded yet."},
    },
)
async def health() -> JSONResponse:
    """Return the API and model health status.

    Returns:
        JSON object with:
        - ``status``: ``"healthy"`` or ``"degraded"``
        - ``version``: API version string
        - ``face_detector_loaded``: ``true`` / ``false``
        - ``fgsm_engine_loaded``: ``true`` / ``false``
    """
    face_loaded = registry.face_detector is not None
    fgsm_loaded = registry.fgsm_attack is not None
    overall = "healthy" if (face_loaded and fgsm_loaded) else "degraded"

    http_status = (
        status.HTTP_200_OK if overall == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=http_status,
        content={
            "status": overall,
            "version": settings.app_version,
            "face_detector_loaded": face_loaded,
            "fgsm_engine_loaded": fgsm_loaded,
        },
    )


# ---------------------------------------------------------------------------
# Versioned API router
# ---------------------------------------------------------------------------

app.include_router(
    cloak_router,
    prefix="/api/v1",
)
