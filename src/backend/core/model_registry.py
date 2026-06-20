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
    detector = registry.face_detector
    attack   = registry.fgsm_attack

    # During shutdown (lifespan):
    registry.clear()
"""

import logging
from typing import Optional

from src.ml_core.models.mtcnn_detector import FaceDetector
from src.ml_core.attacks.fgsm_attack import FGSMAttack
from src.backend.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton container for initialised ML models.

    Attributes:
        face_detector: Loaded ``FaceDetector`` instance, or ``None`` before
            ``load()`` is called.
        fgsm_attack: Loaded ``FGSMAttack`` instance, or ``None`` before
            ``load()`` is called.
    """

    def __init__(self) -> None:
        """Initialise an empty registry (models are not yet loaded)."""
        self.face_detector: Optional[FaceDetector] = None
        self.fgsm_attack: Optional[FGSMAttack] = None

    def load(self) -> None:
        """Load and cache all ML models.

        Called once during FastAPI application startup.  Failure here is
        intentionally propagated — the application should not start without
        functional ML models.

        Raises:
            RuntimeError: If either model fails to initialise (e.g. missing
                weights, no internet connection for first-time download).
        """
        logger.info("ModelRegistry — loading FaceDetector (MTCNN) …")
        self.face_detector = FaceDetector(device="cpu")
        logger.info("ModelRegistry — FaceDetector ready.")

        logger.info(
            "ModelRegistry — loading FGSMAttack (InceptionResnetV1 / VGGFace2) "
            "with epsilon=%.4f …",
            settings.default_epsilon,
        )
        self.fgsm_attack = FGSMAttack(
            epsilon=settings.default_epsilon,
            device="auto",
        )
        logger.info("ModelRegistry — FGSMAttack ready.")
        logger.info("ModelRegistry — all models loaded successfully.")

    def clear(self) -> None:
        """Release model references on application shutdown.

        Sets both model attributes to ``None`` so Python's garbage collector
        can reclaim the memory.  Called automatically by the lifespan
        shutdown handler in ``main.py``.
        """
        logger.info("ModelRegistry — releasing model references …")
        self.face_detector = None
        self.fgsm_attack = None
        logger.info("ModelRegistry — registry cleared.")

    @property
    def is_ready(self) -> bool:
        """Return ``True`` if both models are loaded and available.

        Returns:
            ``True`` when ``face_detector`` and ``fgsm_attack`` are not
            ``None``; ``False`` otherwise.
        """
        return self.face_detector is not None and self.fgsm_attack is not None


# Module-level singleton — imported by lifespan handler and route services.
registry = ModelRegistry()
