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

# FaceNet native input resolution (InceptionResnetV1 trained at 160 × 160).
_FACENET_INPUT_SIZE: int = 160


def prepare_face_tensor(face_crop_rgb: np.ndarray) -> torch.Tensor:
    """Convert a uint8 RGB face crop to a normalised FaceNet-ready tensor.

    Adapted directly from ``test_fgsm_pipeline._prepare_face_tensor()``.

    Steps:
        1. Resize to 160 × 160 using PIL LANCZOS (high-quality downscaling).
        2. Cast to float32 and normalise to ``[0.0, 1.0]`` by dividing by 255.
        3. Permute ``HWC → CHW`` and add batch dimension: ``(1, 3, 160, 160)``.

    Args:
        face_crop_rgb: RGB uint8 NumPy array of shape ``(h, w, 3)`` as
            returned by ``FaceDetector.detect_and_crop()``.

    Returns:
        Float32 tensor of shape ``(1, 3, 160, 160)`` with values in
        ``[0.0, 1.0]``, ready for ``FGSMAttack.attack()``.
    """
    logger.info(
        "attack_service — preparing face tensor from crop %s → (%d, %d)",
        face_crop_rgb.shape[:2],
        _FACENET_INPUT_SIZE,
        _FACENET_INPUT_SIZE,
    )

    # Resize to FaceNet's native input resolution.
    pil_face = Image.fromarray(face_crop_rgb)
    pil_resized = pil_face.resize(
        (_FACENET_INPUT_SIZE, _FACENET_INPUT_SIZE),
        resample=Image.Resampling.LANCZOS,
    )

    # Normalise to [0, 1] and build tensor.
    face_array = np.array(pil_resized, dtype=np.float32) / 255.0
    face_tensor = torch.from_numpy(face_array).permute(2, 0, 1).unsqueeze(0)

    logger.info(
        "attack_service — tensor ready. Shape: %s | Range: [%.4f, %.4f]",
        tuple(face_tensor.shape),
        float(face_tensor.min()),
        float(face_tensor.max()),
    )
    return face_tensor


def tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``[0, 1]`` float CHW tensor to a uint8 HWC RGB NumPy array.

    Adapted directly from ``test_fgsm_pipeline._tensor_to_uint8_rgb()``.

    Steps:
        1. Remove the batch dimension: ``(1, 3, H, W)`` → ``(3, H, W)``.
        2. Permute channels: ``CHW → HWC``.
        3. Move to CPU and detach from autograd graph.
        4. Multiply by 255, clip to ``[0, 255]``, cast to uint8.

    Args:
        tensor: Float tensor of shape ``(1, 3, H, W)`` with values in
            ``[0.0, 1.0]``, as returned by ``FGSMAttack.attack()``.

    Returns:
        uint8 NumPy array of shape ``(H, W, 3)`` in RGB colour space.
    """
    array = (
        tensor.squeeze(0)   # (3, H, W)
        .permute(1, 2, 0)   # (H, W, 3)
        .cpu()
        .detach()
        .numpy()
    )
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


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
        epsilon: FGSM L-infinity perturbation budget.  Overrides the
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
