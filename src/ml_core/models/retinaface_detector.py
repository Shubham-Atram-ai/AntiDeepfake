"""
retinaface_detector.py
----------------------
Face detection model wrapper built on top of the ``retina-face`` library.

Uses RetinaFace — a single-stage face detector that simultaneously predicts
bounding boxes, facial landmarks, and confidence scores using a Feature
Pyramid Network (FPN) backbone.  It is significantly more robust than MTCNN
on occluded, small, or non-frontal faces.

Public interface is **identical** to the previous ``mtcnn_detector.FaceDetector``
so all downstream consumers (services, pipeline script) require no changes.

RetinaFace output format
------------------------
``RetinaFace.detect_faces(img)`` returns a dictionary keyed by face label::

    {
        "face_1": {
            "facial_area": [x1, y1, x2, y2],   # ← bounding box we need
            "score": 0.9987,
            "landmarks": {
                "left_eye": [...],
                "right_eye": [...],
                "nose": [...],
                "mouth_left": [...],
                "mouth_right": [...],
            },
        },
        "face_2": { ... },
        ...
    }

When multiple faces are detected we select the one with the highest ``score``
(equivalent to MTCNN's ``select_largest=True`` heuristic but score-based).

References:
    Deng et al. (2020) "RetinaFace: Single-Shot Multi-Level Face Localisation
    in the Wild"  https://arxiv.org/abs/1905.00641
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
# pyrefly: ignore [missing-import]
from retinaface import RetinaFace
from PIL import Image

logger = logging.getLogger(__name__)


class FaceDetector:
    """Wraps the RetinaFace model to provide face detection and cropping.

    RetinaFace is a single-stage face detector that uses a Feature Pyramid
    Network (FPN) backbone to simultaneously predict bounding boxes,
    five facial landmarks, and a confidence score for each detected face.

    This class exposes a single high-level method — ``detect_and_crop`` —
    that returns both the cropped face region *and* the bounding box
    coordinates.  The bounding box is preserved so that a processed
    (adversarially perturbed) face can later be stitched back into the
    original image at exactly the same position.

    Attributes:
        device (str): Computation device, either ``'cpu'`` or ``'cuda'``
            (informational — RetinaFace manages its own device internally).
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialise the RetinaFace face detector.

        Args:
            device: PyTorch device string (``'cpu'`` or ``'cuda'``).
                RetinaFace uses TensorFlow/Keras internally; this parameter
                is stored for API compatibility with the previous MTCNN
                detector and for future GPU configuration.
        """
        self.device = device
        logger.info(
            "[INFO] Initialising RetinaFace FaceDetector on device: '%s'", device
        )
        # RetinaFace weights are downloaded automatically on first use.
        # A warm-up call is intentionally skipped here to avoid loading
        # the model at import time; it will be lazily initialised on the
        # first call to detect_and_crop().
        logger.info("[INFO] RetinaFace detector ready (lazy initialisation).")

    def detect_and_crop(
        self, image_rgb_array: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        """Detect the primary face in an RGB image and return the crop + bbox.

        Runs RetinaFace inference on the provided RGB NumPy array, selects
        the highest-confidence detected face, reads its ``facial_area``
        field ``[x1, y1, x2, y2]``, and crops the face region directly
        from the original array to preserve full pixel quality.

        Args:
            image_rgb_array: A NumPy array of shape ``(H, W, 3)`` in RGB
                colour space as produced by ``image_loader.load_image()``.

        Returns:
            A two-element tuple ``(face_crop, bounding_box)``::

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

        logger.info("[INFO] Running RetinaFace face detection…")

        # RetinaFace.detect_faces() accepts an RGB uint8 NumPy array directly.
        # Returns a dict of detected faces (empty dict when none are found).
        faces: dict = RetinaFace.detect_faces(image_rgb_array)

        # ── No face detected ────────────────────────────────────────────────
        if not faces or not isinstance(faces, dict):
            logger.warning(
                "[WARNING] No face detected in the provided image. "
                "The image may not contain a clear, forward-facing face."
            )
            return None, None

        # ── Select highest-confidence face ──────────────────────────────────
        # faces is e.g. {"face_1": {"facial_area": [...], "score": 0.99}, ...}
        best_key: str = max(faces, key=lambda k: faces[k].get("score", 0.0))
        best_face: dict = faces[best_key]

        confidence: float = float(best_face.get("score", float("nan")))
        facial_area: List[int] = best_face["facial_area"]  # [x1, y1, x2, y2]

        logger.info(
            "[INFO] Face detected! Bounding box: [x1=%d, y1=%d, x2=%d, y2=%d] "
            "| Confidence: %.4f | Total faces found: %d",
            facial_area[0], facial_area[1], facial_area[2], facial_area[3],
            confidence,
            len(faces),
        )

        # ── Extract and clamp bounding box ──────────────────────────────────
        img_h, img_w = image_rgb_array.shape[:2]
        x1: int = max(0, int(facial_area[0]))
        y1: int = max(0, int(facial_area[1]))
        x2: int = min(img_w, int(facial_area[2]))
        y2: int = min(img_h, int(facial_area[3]))

        bounding_box: List[float] = [float(x1), float(y1), float(x2), float(y2)]

        # Crop directly from the original NumPy array (preserves full quality).
        face_crop: np.ndarray = image_rgb_array[y1:y2, x1:x2].copy()

        logger.info(
            "[INFO] Face cropped from original array. Crop shape: %s",
            face_crop.shape,
        )

        return face_crop, bounding_box
