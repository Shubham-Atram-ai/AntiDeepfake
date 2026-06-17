"""
image_loader.py
---------------
Utility module for safely loading and saving images using OpenCV.

Handles the critical BGR <-> RGB conversion required when bridging between
OpenCV (which uses BGR by default) and ML models like MTCNN (which expect RGB).
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and convert it from BGR to RGB colour space.

    OpenCV reads images in BGR (Blue-Green-Red) order by default, but
    deep learning models such as MTCNN expect images in RGB (Red-Green-Blue)
    order.  This function performs the conversion transparently so that all
    downstream code can safely assume RGB input.

    Args:
        image_path: Absolute or relative path to the image file on disk.

    Returns:
        A NumPy array of shape ``(H, W, 3)`` with dtype ``uint8`` representing
        the image in RGB colour space.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If OpenCV fails to decode the file (e.g. corrupt image).
    """
    if not os.path.exists(image_path):
        logger.error("[ERROR] Image file not found: %s", image_path)
        raise FileNotFoundError(
            f"Image file not found at path: '{image_path}'. "
            "Please verify the path and try again."
        )

    logger.info("[INFO] Loading image from: %s", image_path)
    bgr_array = cv2.imread(image_path)

    if bgr_array is None:
        logger.error(
            "[ERROR] OpenCV could not decode the image at: %s. "
            "The file may be corrupt or in an unsupported format.",
            image_path,
        )
        raise ValueError(
            f"OpenCV failed to decode the image at '{image_path}'. "
            "Ensure the file is a valid image (JPEG, PNG, BMP, etc.)."
        )

    # cv2.imread returns BGR; convert to RGB for ML model compatibility.
    rgb_array: np.ndarray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    logger.info(
        "[INFO] Image loaded successfully. Shape: %s, dtype: %s",
        rgb_array.shape,
        rgb_array.dtype,
    )
    return rgb_array


def save_image(image_array: np.ndarray, save_path: str) -> None:
    """Save an RGB NumPy array to disk as an image file.

    Converts the RGB array back to BGR (required by OpenCV's ``imwrite``),
    creates any missing parent directories automatically, and writes the file.

    Args:
        image_array: A NumPy array of shape ``(H, W, 3)`` in RGB colour space.
        save_path: Destination path including the file name and extension
            (e.g. ``'data/processed/face.jpg'``).

    Returns:
        None

    Raises:
        ValueError: If ``cv2.imwrite`` returns ``False``, indicating that the
            image could not be written (e.g. unsupported extension or
            permission error).
    """
    # Ensure all parent directories in the path exist before writing.
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
        logger.info("[INFO] Ensured output directory exists: %s", parent_dir)

    # Convert RGB back to BGR for OpenCV's imwrite.
    bgr_array: np.ndarray = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(save_path, bgr_array)
    if not success:
        logger.error(
            "[ERROR] cv2.imwrite failed to save image to: %s. "
            "Check file extension and write permissions.",
            save_path,
        )
        raise ValueError(
            f"Failed to save image to '{save_path}'. "
            "Verify the path, file extension, and directory permissions."
        )

    logger.info("[INFO] Image saved successfully to: %s", save_path)


def draw_detections(
    image_rgb_array: np.ndarray,
    bounding_box: list,
    confidence: float = 1.0,
    box_color: tuple = (0, 230, 118),
    label_text: str = "Face",
    box_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw a labelled bounding box on an RGB image and return the annotated copy.

    Takes the original full-resolution RGB image and the bounding box returned
    by ``FaceDetector.detect_and_crop()``, draws a rectangle and a confidence
    label using OpenCV, and returns a *new* array so the original is never
    mutated.

    Args:
        image_rgb_array: The original image as an RGB NumPy array ``(H, W, 3)``.
        bounding_box: Detected face bounding box as ``[x1, y1, x2, y2]`` in
            pixel coordinates (as returned by ``FaceDetector.detect_and_crop``).
        confidence: Detection confidence score in the range ``[0.0, 1.0]``.
            Displayed in the label.  Defaults to ``1.0``.
        box_color: BGR colour tuple for the rectangle and label background.
            Defaults to bright green ``(0, 230, 118)``.
        label_text: Short label string shown above the bounding box.
            Defaults to ``'Face'``.
        box_thickness: Pixel thickness of the bounding box border.
            Defaults to ``2``.
        font_scale: OpenCV font scale for the label text.  Defaults to ``0.6``.

    Returns:
        A new NumPy array of shape ``(H, W, 3)`` in RGB colour space with
        the bounding box and label drawn on it.  The input array is unchanged.

    Raises:
        ValueError: If ``bounding_box`` does not contain exactly 4 elements.
    """
    if len(bounding_box) != 4:
        raise ValueError(
            f"bounding_box must have exactly 4 elements [x1, y1, x2, y2], "
            f"got {len(bounding_box)} elements."
        )

    # Work on a copy — never mutate the source array.
    annotated: np.ndarray = image_rgb_array.copy()

    x1, y1, x2, y2 = (int(v) for v in bounding_box)

    # ── Draw the bounding rectangle ──────────────────────────────────────────
    # OpenCV rectangle() works in-place; colour order matches the array which
    # is RGB, so we supply an RGB colour here.
    cv2.rectangle(
        annotated,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=box_color,
        thickness=box_thickness,
        lineType=cv2.LINE_AA,
    )

    # ── Build the label string ───────────────────────────────────────────────
    label: str = f"{label_text}: {confidence:.2%}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, thickness=1
    )

    # Background filled pill above the box (clamp so it never goes off-screen).
    label_y1 = max(y1 - text_h - baseline - 6, 0)
    label_y2 = max(y1 - 2, text_h + baseline)

    cv2.rectangle(
        annotated,
        pt1=(x1, label_y1),
        pt2=(x1 + text_w + 6, label_y2),
        color=box_color,
        thickness=cv2.FILLED,
    )

    # Label text in dark colour for contrast.
    text_color: tuple = (20, 20, 20)
    cv2.putText(
        annotated,
        label,
        org=(x1 + 3, label_y2 - baseline - 1),
        fontFace=font,
        fontScale=font_scale,
        color=text_color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    logger.info(
        "[INFO] Bounding box drawn on image. "
        "Box: [%d, %d, %d, %d] | Label: '%s'",
        x1, y1, x2, y2, label,
    )
    return annotated
