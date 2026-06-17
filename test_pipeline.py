"""
test_pipeline.py
----------------
Smoke-test script that exercises the full face-detection pipeline end-to-end.

Run from the project root:
    python test_pipeline.py

Expected outputs:
    • Structured INFO/WARNING/ERROR log messages in the terminal.
    • The full original image with a bounding box drawn around the detected face,
      saved to  data/output/annotated.jpg  (on success).
"""

import logging
import sys
import os

# ---------------------------------------------------------------------------
# Logging configuration
# Must be called before any other module imports so that all loggers — including
# those inside src/ — inherit this root-level configuration.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap
# Adds the project root to sys.path so that ``src`` is importable without
# installing the package.  This lets us run the script directly as
# ``python test_pipeline.py`` from the repo root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Project imports (after path bootstrap)
# ---------------------------------------------------------------------------
from src.ml_core.utils.image_loader import load_image, save_image, draw_detections  # noqa: E402
from src.ml_core.models.mtcnn_detector import FaceDetector                            # noqa: E402


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
RAW_IMAGE_PATH: str = os.path.join(PROJECT_ROOT, "data", "raw", "test.jpg")
if not os.path.exists(RAW_IMAGE_PATH):
    # Fallback to .jpeg extension if .jpg is not present.
    RAW_IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "test.jpeg")

# Output goes to data/output/ to keep the annotated result separate from
# the processed (adversarial) data that will live in data/processed/.
ANNOTATED_OUTPUT_PATH: str = os.path.join(
    PROJECT_ROOT, "data", "output", "annotated.jpg"
)


def main() -> None:
    """Execute the face-detection smoke-test pipeline.

    Steps:
        1. Load the test image from ``data/raw/``.
        2. Run MTCNN face detection to locate the primary face.
        3. Draw a labelled bounding box on the original (full) image.
        4. Save the annotated image to ``data/output/annotated.jpg``.

    Returns:
        None.  Exits with code 1 on any unrecoverable error.
    """
    logger.info("=" * 60)
    logger.info("Anti-Deepfake — Face Detection Pipeline Smoke Test")
    logger.info("=" * 60)

    # ── Step 1: Load the raw test image ─────────────────────────────────────
    logger.info("STEP 1 — Loading raw image from: %s", RAW_IMAGE_PATH)
    try:
        image_rgb = load_image(RAW_IMAGE_PATH)
    except FileNotFoundError as exc:
        logger.error("[ERROR] %s", exc)
        logger.error(
            "Please place a JPEG image named 'test.jpg' inside 'data/raw/' "
            "and re-run this script."
        )
        sys.exit(1)
    except ValueError as exc:
        logger.error("[ERROR] Image decode failed: %s", exc)
        sys.exit(1)

    logger.info("STEP 1 — Complete. Image shape: %s", image_rgb.shape)

    # ── Step 2: Detect the primary face ──────────────────────────────────────
    logger.info("STEP 2 — Initialising FaceDetector and running inference…")
    detector = FaceDetector(device="cpu")

    try:
        # detect_and_crop() still returns the crop + bbox internally —
        # we only use the bbox here; the crop is discarded with _.
        _, bounding_box = detector.detect_and_crop(image_rgb)
    except ValueError as exc:
        logger.error("[ERROR] Face detection raised an error: %s", exc)
        sys.exit(1)

    if bounding_box is None:
        logger.warning(
            "[WARNING] No face was found in '%s'. "
            "Try a different image with a clear, forward-facing face.",
            RAW_IMAGE_PATH,
        )
        sys.exit(0)

    logger.info("STEP 2 — Complete.")
    logger.info(
        "  ↳ Bounding box (x1, y1, x2, y2): %s",
        [f"{v:.1f}" for v in bounding_box],
    )

    # ── Step 3: Draw bounding box on the original image ───────────────────────
    logger.info("STEP 3 — Annotating image with bounding box…")
    try:
        annotated_rgb = draw_detections(
            image_rgb_array=image_rgb,
            bounding_box=bounding_box,
            confidence=1.0,       # MTCNN detected with 100 % confidence.
            box_color=(0, 230, 118),   # Bright green in RGB.
            label_text="Face",
            box_thickness=2,
            font_scale=0.55,
        )
    except ValueError as exc:
        logger.error("[ERROR] draw_detections failed: %s", exc)
        sys.exit(1)

    logger.info("STEP 3 — Complete. Full-resolution image annotated.")

    # ── Step 4: Save the annotated full image ─────────────────────────────────
    logger.info("STEP 4 — Saving annotated image to: %s", ANNOTATED_OUTPUT_PATH)
    try:
        save_image(annotated_rgb, ANNOTATED_OUTPUT_PATH)
    except ValueError as exc:
        logger.error("[ERROR] Could not save annotated image: %s", exc)
        sys.exit(1)

    logger.info("STEP 4 — Complete.")
    logger.info("=" * 60)
    logger.info("✓ Smoke test PASSED — annotated image saved successfully.")
    logger.info("  ↳ Output: %s", ANNOTATED_OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
