"""
attack_service.py
-----------------
Service adapter for the PGD adversarial attack.

Encapsulates the tensor preparation and attack orchestration logic adapted
directly from ``test_fgsm_pipeline.py`` — specifically the private helpers
``_prepare_face_tensor()`` and ``_tensor_to_uint8_rgb()``.

No ML logic is duplicated here.  The module provides:

1. ``prepare_face_tensor()``  — converts a uint8 RGB crop to a FaceNet-ready
   float32 tensor (adapted from ``test_fgsm_pipeline._prepare_face_tensor``).
2. ``tensor_to_uint8_rgb()``  — reverses the above (from ``_tensor_to_uint8_rgb``).
3. ``run_attack()``           — calls ``PGDAttack.attack()`` via the registry.
"""

import logging
from typing import Tuple

import numpy as np
import torch
from fastapi import HTTPException, status
from PIL import Image

from src.ml_core.attacks.pgd_attack import PGDAttack

logger = logging.getLogger(__name__)

from src.ml_core.utils.tensor_helpers import prepare_face_tensor, tensor_to_uint8_rgb


def run_attack(
    attack: PGDAttack,
    face_tensor: torch.Tensor,
    epsilon: float,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Execute the PGD attack and return the adversarial face as uint8 RGB.

    Calls ``PGDAttack.attack()`` and immediately converts the adversarial
    tensor to a uint8 RGB NumPy array suitable for image reconstruction.

    Args:
        attack: Initialised ``PGDAttack`` instance from the model registry.
        face_tensor: Float32 tensor of shape ``(1, 3, 160, 160)`` in
            ``[0.0, 1.0]`` — output of ``prepare_face_tensor()``.
        epsilon: PGD L-infinity perturbation budget.  Overrides the
            value stored in ``attack.epsilon`` for this request.

    Returns:
        A two-element tuple ``(adversarial_face_rgb, adversarial_tensor)``:

        - ``adversarial_face_rgb``: ``(160, 160, 3)`` uint8 RGB array.
        - ``adversarial_tensor``: Raw ``(1, 3, 160, 160)`` float tensor,
          retained for possible downstream use (e.g. embedding comparison).

    Raises:
        HTTPException(500, PROCESSING_ERROR): If the PGD attack
            fails due to a broken computation graph.
    """
    logger.info("attack_service — executing PGD attack | epsilon=%.4f", epsilon)

    try:
        adversarial_tensor = attack.attack(face_tensor, epsilon=epsilon)
    except (ValueError, RuntimeError) as exc:
        logger.error("attack_service — PGD attack failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error": "PGD attack failed during adversarial perturbation.",
            },
        ) from exc

    adversarial_face_rgb = tensor_to_uint8_rgb(adversarial_tensor)
    logger.info(
        "attack_service — adversarial face generated. Shape: %s | Range: [%d, %d]",
        adversarial_face_rgb.shape,
        int(adversarial_face_rgb.min()),
        int(adversarial_face_rgb.max()),
    )
    return adversarial_face_rgb, adversarial_tensor
