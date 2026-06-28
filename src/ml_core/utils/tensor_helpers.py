"""
tensor_helpers.py
-----------------
Utility functions for converting between numpy image arrays and PyTorch tensors.
"""

import logging
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# FaceNet native input resolution (InceptionResnetV1 trained at 160 × 160).
_FACENET_INPUT_SIZE: int = 160

def prepare_face_tensor(face_crop_rgb: np.ndarray) -> torch.Tensor:
    """Convert a uint8 RGB face crop to a normalised FaceNet-ready tensor.

    Steps:
        1. Resize to 160 × 160 using PIL LANCZOS (high-quality downscaling).
        2. Cast to float32 and normalise to ``[0.0, 1.0]`` by dividing by 255.
        3. Permute ``HWC → CHW`` and add batch dimension: ``(1, 3, 160, 160)``.

    Args:
        face_crop_rgb: RGB uint8 NumPy array of shape ``(h, w, 3)`` as
            returned by ``FaceDetector.detect_and_crop()``.

    Returns:
        Float32 tensor of shape ``(1, 3, 160, 160)`` with values in
        ``[0.0, 1.0]``, ready for attack or embedding.
    """
    logger.info(
        "tensor_helpers — preparing face tensor from crop %s → (%d, %d)",
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
        "tensor_helpers — tensor ready. Shape: %s | Range: [%.4f, %.4f]",
        tuple(face_tensor.shape),
        float(face_tensor.min()),
        float(face_tensor.max()),
    )
    return face_tensor


def tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``[0, 1]`` float CHW tensor to a uint8 HWC RGB NumPy array.

    Steps:
        1. Remove the batch dimension: ``(1, 3, H, W)`` → ``(3, H, W)``.
        2. Permute channels: ``CHW → HWC``.
        3. Move to CPU and detach from autograd graph.
        4. Multiply by 255, clip to ``[0, 255]``, cast to uint8.

    Args:
        tensor: Float tensor of shape ``(1, 3, H, W)`` with values in
            ``[0.0, 1.0]``.

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
