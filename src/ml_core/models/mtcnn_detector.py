"""
mtcnn_detector.py
-----------------
Face detection model wrapper built on top of the ``facenet-pytorch`` library.

Uses MTCNN (Multi-task Cascaded Convolutional Network) — a lightweight but
highly accurate deep learning pipeline that detects faces in three sequential
stages (Proposal, Refinement, Output) to produce tight bounding boxes and
five facial landmark points.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

logger = logging.getLogger(__name__)


class FaceDetector:
    """Wraps the MTCNN model to provide face detection and cropping.

    MTCNN (Multi-task Cascaded Convolutional Network) is a classic deep
    learning model composed of three small convolutional networks (P-Net,
    R-Net, O-Net) that cascade together to accurately locate faces in an
    image at multiple scales.

    This class exposes a single high-level method — ``detect_and_crop`` —
    that returns both the cropped face region *and* the bounding box
    coordinates.  The bounding box is preserved so that a processed
    (adversarially perturbed) face can later be stitched back into the
    original image at exactly the same position.

    Attributes:
        device (str): Computation device, either ``'cpu'`` or ``'cuda'``.
        mtcnn (MTCNN): The underlying facenet-pytorch MTCNN detector instance.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialise the MTCNN face detector.

        Args:
            device: PyTorch device string.  Use ``'cpu'`` for CPU-only
                inference or ``'cuda'`` if a compatible NVIDIA GPU is
                available.  Defaults to ``'cpu'``.
        """
        self.device = device
        logger.info(
            "[INFO] Initialising MTCNN FaceDetector on device: '%s'", device
        )

        # keep_all=False → only the single highest-confidence face is returned,
        # which is the dominant face in privacy-protection use-cases.
        # select_largest=True   → when keep_all=False, MTCNN returns the
        # largest detected face, ensuring we always cloak the primary subject.
        self.mtcnn = MTCNN(
            keep_all=False,
            select_largest=True,
            device=device,
            post_process=False,   # Return raw uint8 pixel values, not tensors.
        )
        logger.info("[INFO] MTCNN model loaded and ready.")

    def detect_and_crop(
        self, image_rgb_array: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        """Detect the primary face in an RGB image and return the crop + bbox.

        Converts the NumPy array to a PIL Image (required by facenet-pytorch),
        runs MTCNN inference to locate the face, then manually crops the face
        region directly from the original NumPy array so that pixel values
        remain as close to the source as possible (MTCNN's internal crop is
        used only for the bounding box, not for the final crop returned here).

        Args:
            image_rgb_array: A NumPy array of shape ``(H, W, 3)`` in RGB
                colour space as produced by ``image_loader.load_image()``.

        Returns:
            A two-element tuple ``(face_crop, bounding_box)``:

            - ``face_crop`` (``np.ndarray`` or ``None``): Cropped face region
              as a NumPy array of shape ``(h, w, 3)`` in RGB, or ``None`` if
              no face was detected.
            - ``bounding_box`` (``list[float]`` or ``None``): Detected face
              bounding box as ``[x1, y1, x2, y2]`` in pixel coordinates
              relative to the original image, or ``None`` if no face was
              detected.

        Raises:
            ValueError: If ``image_rgb_array`` is not a valid 3-channel NumPy
                array.
        """
        if image_rgb_array is None or image_rgb_array.ndim != 3:
            logger.error(
                "[ERROR] Invalid image array provided to detect_and_crop. "
                "Expected a 3-D NumPy array (H, W, C)."
            )
            raise ValueError(
                "image_rgb_array must be a non-None NumPy array with shape "
                "(H, W, 3).  Received: "
                f"{'None' if image_rgb_array is None else image_rgb_array.shape}"
            )

        # MTCNN (facenet-pytorch) requires a PIL Image as input.
        pil_image: Image.Image = Image.fromarray(image_rgb_array)

        logger.info("[INFO] Running MTCNN face detection…")

        # detect() returns (boxes, probs).
        # boxes shape: (1, 4) for keep_all=False, or None if no face found.
        boxes, probs = self.mtcnn.detect(pil_image)

        # ── No face detected ────────────────────────────────────────────────
        if boxes is None or len(boxes) == 0:
            logger.warning(
                "[WARNING] No face detected in the provided image. "
                "The image may not contain a clear, forward-facing face."
            )
            return None, None

        # ── Face detected ────────────────────────────────────────────────────
        # boxes[0] → [x1, y1, x2, y2] for the primary (largest) face.
        raw_box = boxes[0]
        confidence: float = float(probs[0]) if probs is not None else float("nan")

        logger.info(
            "[INFO] Face detected! Bounding box: [x1=%.1f, y1=%.1f, x2=%.1f, y2=%.1f] "
            "| Confidence: %.4f",
            raw_box[0], raw_box[1], raw_box[2], raw_box[3],
            confidence,
        )

        # Extract integer coordinates and clamp to image bounds to prevent
        # negative indices or out-of-range slices.
        img_h, img_w = image_rgb_array.shape[:2]
        x1: int = max(0, int(raw_box[0]))
        y1: int = max(0, int(raw_box[1]))
        x2: int = min(img_w, int(raw_box[2]))
        y2: int = min(img_h, int(raw_box[3]))

        bounding_box: List[float] = [float(x1), float(y1), float(x2), float(y2)]

        # Crop directly from the original NumPy array (preserves full quality).
        face_crop: np.ndarray = image_rgb_array[y1:y2, x1:x2].copy()

        logger.info(
            "[INFO] Face cropped from original array. "
            "Crop shape: %s",
            face_crop.shape,
        )

        return face_crop, bounding_box
