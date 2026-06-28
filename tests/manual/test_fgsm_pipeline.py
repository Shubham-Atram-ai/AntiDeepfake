"""
test_pgd_pipeline.py  (entry point: test_fgsm_pipeline.py)
-----------------------------------------------------------
End-to-end smoke test for the PGD adversarial cloaking pipeline.

Executes the complete workflow from raw image loading to protected image
output, covering every stage of the Phase 2 pipeline:

    Original Image
          ↓
    Image Loader            (src/ml_core/utils/image_loader.py)
          ↓
    RetinaFace Detection    (src/ml_core/models/retinaface_detector.py)
          ↓
    Face Crop
          ↓
    FaceNet Embedding       (src/ml_core/attacks/pgd_attack.py)
          ↓
    PGD Attack              (eps=8/255, alpha=2/255, steps=10)
          ↓
    Adversarial Face
          ↓
    Reconstruction Into Original Image
          ↓
    Protected Image
          ↓
    SSIM + PSNR Evaluation  (src/ml_core/evaluation/metrics.py)

Run from the project root with the virtual environment active:
    python test_fgsm_pipeline.py

Prerequisites:
    - data/raw/test.jpg  (or test.jpeg)  — a JPEG image containing a face.
    - Pretrained FaceNet (VGGFace2) weights — downloaded automatically on
      first run if internet connectivity is available.
"""

import os

# ---------------------------------------------------------------------------
# Strict enforcement of thread pool isolation to prevent TF/PyTorch segfaults
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import logging

# ---------------------------------------------------------------------------
# Logging — must be configured before any project imports so child loggers
# inherit this root-level configuration.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility seeds (top-level, before any torch/numpy import)
# ---------------------------------------------------------------------------
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Path bootstrap — allows running as ``python test_fgsm_pipeline.py``
# from the project root without installing the package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import cv2
from PIL import Image

from src.ml_core.utils.image_loader import load_image, save_image
from src.ml_core.utils.tensor_helpers import prepare_face_tensor, tensor_to_uint8_rgb
from src.ml_core.legacy.mtcnn_detector import FaceDetector
from src.ml_core.attacks.pgd_attack import PGDAttack
from src.ml_core.evaluation.metrics import compute_ssim, compute_psnr

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
# The pipeline probes both .jpg and .jpeg so it works regardless of how
# the test image was named.
_RAW_JPG = os.path.join(PROJECT_ROOT, "data", "raw", "test.jpg")
_RAW_JPEG = os.path.join(PROJECT_ROOT, "data", "raw", "test.jpeg")
RAW_IMAGE_PATH: str = _RAW_JPG if os.path.exists(_RAW_JPG) else _RAW_JPEG

OUTPUT_PATH: str = os.path.join(
    PROJECT_ROOT, "data", "output", "cloaked_protected.jpg"
)

# PGD hyperparameters (standard PGD-8 budget).
EPS:   float = 8 / 255    # L∞ perturbation budget  (~0.0314)
ALPHA: float = 2 / 255    # step size per iteration  (~0.0078)
STEPS: int   = 10         # number of gradient steps

# Alias kept so downstream log messages remain concise.
EPSILON: float = EPS

# FaceNet native input resolution (InceptionResnetV1 trained at 160 × 160).
FACENET_INPUT_SIZE: int = 160


# ---------------------------------------------------------------------------
# Pipeline stages (each as a standalone function for clarity)
# ---------------------------------------------------------------------------

def _load_raw_image() -> np.ndarray:
    """Load the raw test image from disk.

    Returns:
        RGB uint8 NumPy array of shape (H, W, 3).

    Raises:
        FileNotFoundError: If neither test.jpg nor test.jpeg is found.
        ValueError: If OpenCV fails to decode the image.
        SystemExit: On unrecoverable error (exits with code 1).
    """
    logger.info("STEP 1 — Loading raw image from: %s", RAW_IMAGE_PATH)
    try:
        image_rgb = load_image(RAW_IMAGE_PATH)
    except FileNotFoundError as exc:
        logger.error("[ERROR] %s", exc)
        logger.error(
            "Please place a JPEG image named 'test.jpg' or 'test.jpeg' "
            "inside 'data/raw/' and re-run this script."
        )
        sys.exit(1)
    except ValueError as exc:
        logger.error("[ERROR] Image decode failed: %s", exc)
        sys.exit(1)

    logger.info("STEP 1 — Complete. Image shape: %s dtype: %s",
                image_rgb.shape, image_rgb.dtype)
    return image_rgb


def _detect_face(
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Run RetinaFace detection and extract the primary face crop.

    Args:
        image_rgb: Full-resolution RGB uint8 image.

    Returns:
        Tuple (face_crop_rgb, bounding_box) where bounding_box is
        [x1, y1, x2, y2] already clamped to image bounds.

    Raises:
        SystemExit: If no face is detected (exits with code 0) or if
            the detector raises an error (exits with code 1).
    """
    logger.info("STEP 2 — Initialising RetinaFace FaceDetector…")
    detector = FaceDetector(device="cpu")

    try:
        face_crop, bounding_box = detector.detect_and_crop(image_rgb)
    except ValueError as exc:
        logger.error("[ERROR] Face detection raised an error: %s", exc)
        sys.exit(1)

    if face_crop is None or bounding_box is None:
        logger.warning(
            "[WARNING] No face detected in '%s'. "
            "Use an image with a clear, forward-facing face.",
            RAW_IMAGE_PATH,
        )
        sys.exit(0)

    logger.info(
        "STEP 2 — Complete. Bounding box (x1, y1, x2, y2): %s | "
        "Crop shape: %s",
        [f"{v:.1f}" for v in bounding_box],
        face_crop.shape,
    )
    return face_crop, bounding_box




def _run_pgd(
    attack: PGDAttack,
    face_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the baseline embedding and the PGD adversarial tensor.

    Args:
        attack: Initialised PGDAttack instance.
        face_tensor: Float32 tensor (1, 3, 160, 160) in [0.0, 1.0].

    Returns:
        Tuple (original_embedding, adversarial_tensor):
        - original_embedding: (1, 512) float tensor.
        - adversarial_tensor: (1, 3, 160, 160) float tensor in [0.0, 1.0].
    """
    logger.info("STEP 4 — Computing baseline FaceNet embedding…")
    original_embedding = attack.get_embedding(face_tensor)
    logger.info("STEP 4 — Baseline embedding computed. Shape: %s",
                tuple(original_embedding.shape))

    logger.info(
        "STEP 5 — Executing PGD attack (eps=%.5f, alpha=%.5f, steps=%d)…",
        EPS, ALPHA, STEPS,
    )
    adversarial_tensor = attack.attack(face_tensor)
    logger.info("STEP 5 — PGD attack complete.")

    return original_embedding, adversarial_tensor




def _reconstruct_image(
    original_rgb: np.ndarray,
    adversarial_face_rgb: np.ndarray,
    bounding_box: list[float],
) -> np.ndarray:
    """Reinsert the adversarial face crop into the original image canvas.

    The adversarial face tensor (160 × 160) is resized back to the original
    bounding box dimensions and pasted into an exact copy of the original
    image.  Bounding box coordinates are defensively clamped a second time
    to the absolute canvas dimensions to prevent out-of-bounds slicing —
    even though detect_and_crop() already clamps internally.

    Args:
        original_rgb: Full-resolution original RGB uint8 image.
        adversarial_face_rgb: Adversarial face as RGB uint8 array (H', W', 3).
        bounding_box: [x1, y1, x2, y2] bounding box from the detector.

    Returns:
        Protected (cloaked) full-resolution RGB uint8 image.
    """
    logger.info("STEP 6 — Reconstructing cloaked image…")

    img_h, img_w = original_rgb.shape[:2]

    # ── CRITICAL: Defensively clamp bbox to absolute canvas dimensions ────────
    # detect_and_crop() already clamps, but we add a second guard here to
    # prevent ANY possibility of out-of-bounds array slicing — e.g. if a
    # bbox is loaded from a source other than the detector.
    x1: int = max(0, min(int(bounding_box[0]), img_w - 1))
    y1: int = max(0, min(int(bounding_box[1]), img_h - 1))
    x2: int = max(x1 + 1, min(int(bounding_box[2]), img_w))
    y2: int = max(y1 + 1, min(int(bounding_box[3]), img_h))

    logger.info(
        "  ↳ Clamped bounding box: x1=%d, y1=%d, x2=%d, y2=%d "
        "(canvas: %d×%d)",
        x1, y1, x2, y2, img_w, img_h,
    )

    # Target dimensions of the face slot in the original image.
    target_w = x2 - x1
    target_h = y2 - y1

    # Resize the adversarial face back to the original bounding box size.
    # Use LANCZOS for high-quality downscaling to the original resolution.
    adv_resized = cv2.resize(
        adversarial_face_rgb,
        (target_w, target_h),
        interpolation=cv2.INTER_LANCZOS4,
    )
    logger.info(
        "  ↳ Adversarial face resized: %s → (%d, %d)",
        adversarial_face_rgb.shape[:2],
        target_h,
        target_w,
    )

    # Copy the original image (never mutate in place) and paste the face.
    cloaked_rgb = original_rgb.copy()
    cloaked_rgb[y1:y2, x1:x2] = adv_resized

    logger.info("STEP 6 — Reconstruction complete. Output shape: %s",
                cloaked_rgb.shape)
    return cloaked_rgb


def _evaluate_and_log(
    original_rgb: np.ndarray,
    cloaked_rgb: np.ndarray,
) -> tuple[float, float]:
    """Compute SSIM and PSNR between the original and cloaked images.

    Args:
        original_rgb: Original full-resolution RGB uint8 image.
        cloaked_rgb: Cloaked (adversarially perturbed) RGB uint8 image.

    Returns:
        Tuple (ssim_score, psnr_db).
    """
    logger.info("STEP 8 — Computing image quality metrics…")

    ssim_score = compute_ssim(original_rgb, cloaked_rgb)
    psnr_db = compute_psnr(original_rgb, cloaked_rgb)

    logger.info("=" * 60)
    logger.info("  SSIM : %.6f  (1.0 = identical; ≥ 0.90 = imperceptible)",
                ssim_score)
    logger.info("  PSNR : %.4f dB  (> 40 dB = imperceptible; 30–40 dB = minor noise)",
                psnr_db)
    logger.info("=" * 60)
    return ssim_score, psnr_db


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the complete PGD adversarial cloaking pipeline.

    Steps:
        1. Load raw image from data/raw/test.jpg (or test.jpeg).
        2. Run RetinaFace face detection via FaceDetector.
        3. Prepare 160×160 FaceNet-compatible face tensor.
        4. Compute baseline FaceNet embedding.
        5. Execute PGD attack (eps=8/255, alpha=2/255, steps=10).
        6. Reconstruct the adversarial face into the original image canvas.
        7. Save the protected image to data/output/cloaked_protected.jpg.
        8. Compute SSIM and PSNR and log the results.

    Returns:
        None.  Exits with code 1 on any unrecoverable error.
    """
    logger.info("=" * 60)
    logger.info("Anti-Deepfake — PGD Adversarial Cloaking Pipeline")
    logger.info("Phase 2 | eps=%.5f | alpha=%.5f | steps=%d", EPS, ALPHA, STEPS)
    logger.info("=" * 60)

    # ── Step 1: Load raw image ────────────────────────────────────────────────
    original_rgb = _load_raw_image()

    # ── Step 2: Detect face and extract crop ──────────────────────────────────
    face_crop_rgb, bounding_box = _detect_face(original_rgb)

    # ── Step 3: Prepare FaceNet-compatible tensor ─────────────────────────────
    face_tensor = prepare_face_tensor(face_crop_rgb)

    # ── Step 4 & 5: Baseline embedding + PGD attack ──────────────────────
    logger.info("STEP 4/5 — Initialising PGD attack engine…")
    try:
        attack = PGDAttack(eps=EPS, alpha=ALPHA, steps=STEPS, device="auto")
    except RuntimeError as exc:
        logger.error("[ERROR] Could not initialise PGD attack: %s", exc)
        sys.exit(1)

    try:
        original_embedding, adversarial_tensor = _run_pgd(attack, face_tensor)
    except (ValueError, RuntimeError) as exc:
        logger.error("[ERROR] PGD attack failed: %s", exc)
        sys.exit(1)

    # Log embedding distance achieved
    import torch.nn.functional as F
    adv_embedding = attack.get_embedding(adversarial_tensor)
    cosine_sim = float(
        F.cosine_similarity(original_embedding, adv_embedding, dim=1).mean()
    )
    logger.info(
        "  ↳ Embedding cosine similarity (original vs adversarial): %.6f "
        "(lower = more confused identity; 1.0 = identical)",
        cosine_sim,
    )

    # ── Step 6: Convert adversarial tensor → uint8 RGB ───────────────────────
    logger.info("STEP 6 — Converting adversarial tensor to uint8 RGB array…")
    adversarial_face_rgb = tensor_to_uint8_rgb(adversarial_tensor)
    logger.info(
        "STEP 6 — Conversion complete. Shape: %s | Range: [%d, %d]",
        adversarial_face_rgb.shape,
        int(adversarial_face_rgb.min()),
        int(adversarial_face_rgb.max()),
    )

    # ── Step 7: Reconstruct protected image ───────────────────────────────────
    cloaked_rgb = _reconstruct_image(original_rgb, adversarial_face_rgb, bounding_box)

    # ── Step 8: Save output ───────────────────────────────────────────────────
    # CRITICAL: save_image() expects an RGB array and handles RGB→BGR
    # conversion internally before calling cv2.imwrite.  We must NOT
    # manually call cv2.cvtColor here — doing so would invert the channels
    # and cause a "Smurf Face" blue-tinted colour artifact.
    logger.info("STEP 7 — Saving protected image to: %s", OUTPUT_PATH)
    try:
        save_image(cloaked_rgb, OUTPUT_PATH)
    except ValueError as exc:
        logger.error("[ERROR] Could not save protected image: %s", exc)
        sys.exit(1)
    logger.info("STEP 7 — Complete. Protected image saved.")

    # ── Step 9: Evaluate quality metrics ─────────────────────────────────────
    ssim_score, psnr_db = _evaluate_and_log(original_rgb, cloaked_rgb)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("✓ PGD Pipeline PASSED")
    logger.info("  ↳ Output      : %s", OUTPUT_PATH)
    logger.info("  ↳ Eps         : %.5f  (8/255)", EPS)
    logger.info("  ↳ Alpha       : %.5f  (2/255)", ALPHA)
    logger.info("  ↳ Steps       : %d", STEPS)
    logger.info("  ↳ SSIM        : %.6f", ssim_score)
    logger.info("  ↳ PSNR        : %.4f dB", psnr_db)
    logger.info(
        "  ↳ Cosine Sim  : %.6f  (original vs adversarial embeddings)",
        cosine_sim,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
