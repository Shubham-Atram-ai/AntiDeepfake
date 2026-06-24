"""
model_registry.py
-----------------
Application-level model registry for the AntiDeepfake backend.

Stores fully-initialised ML model instances so they are loaded exactly
**once** at application startup and reused across every request, avoiding
the multi-second penalty of re-loading FaceNet weights per request.

Usage::

    from src.backend.core.model_registry import registry

    # During startup (lifespan):
    registry.load()

    # During a request:
    detector  = registry.face_detector
    attack    = registry.pgd_attack

    # During shutdown (lifespan):
    registry.clear()
"""

import logging
from typing import Optional

from src.ml_core.models.retinaface_detector import FaceDetector
from src.ml_core.attacks.pgd_attack import PGDAttack
from src.backend.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton container for initialised ML models.

    Attributes:
        face_detector: Loaded ``FaceDetector`` (RetinaFace) instance, or
            ``None`` before ``load()`` is called.
        pgd_attack: Loaded ``PGDAttack`` instance, or ``None`` before
            ``load()`` is called.
    """

    def __init__(self) -> None:
        """Initialise an empty registry (models are not yet loaded)."""
        self.face_detector: Optional[FaceDetector] = None
        self.pgd_attack: Optional[PGDAttack] = None

    def load(self) -> None:
        """Load and cache all ML models.

        Called once during FastAPI application startup.  Failure here is
        intentionally propagated — the application should not start without
        functional ML models.

        Raises:
            RuntimeError: If either model fails to initialise (e.g. missing
                weights, no internet connection for first-time download).
        """
        logger.info("ModelRegistry — loading FaceDetector (RetinaFace) …")
        self.face_detector = FaceDetector(device="cpu")
        logger.info("ModelRegistry — FaceDetector ready.")

        logger.info(
            "ModelRegistry — loading PGDAttack (InceptionResnetV1 / VGGFace2) "
            "with eps=8/255, alpha=2/255, steps=10 …"
        )
        self.pgd_attack = PGDAttack(
            eps=8 / 255,
            alpha=2 / 255,
            steps=10,
            device="auto",
        )
        logger.info("ModelRegistry — PGDAttack ready.")
        logger.info("ModelRegistry — all models loaded successfully.")

    def clear(self) -> None:
        """Release model references on application shutdown.

        Sets both model attributes to ``None`` so Python's garbage collector
        can reclaim the memory.  Called automatically by the lifespan
        shutdown handler in ``main.py``.
        """
        logger.info("ModelRegistry — releasing model references …")
        self.face_detector = None
        self.pgd_attack = None
        logger.info("ModelRegistry — registry cleared.")

    @property
    def is_ready(self) -> bool:
        """Return ``True`` if both models are loaded and available.

        Returns:
            ``True`` when ``face_detector`` and ``pgd_attack`` are not
            ``None``; ``False`` otherwise.
        """
        return self.face_detector is not None and self.pgd_attack is not None


# Module-level singleton — imported by lifespan handler and route services.
registry = ModelRegistry()
