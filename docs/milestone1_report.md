# Milestone 1 Report — Face Detection Pipeline

**Project:** Anti-Deepfake / Adversarial Image Cloaking  
**Milestone:** 1 — Core Face Detection Pipeline (MVP Foundation)  
**Date:** 2026-06-18  
**Author:** ML-Sec Engineering  
**Status:** ✅ Complete — Smoke Test Verified & Passed

---

## Revision History

| Rev | Date | Change |
|-----|------|--------|
| v1.0 | 2026-06-18 | Initial pipeline — load, detect, crop, save |
| v1.1 | 2026-06-18 | **Replaced crop output with full-image bounding-box annotation.** Added `draw_detections()` to `image_loader.py`; output now saved to `data/output/annotated.jpg` |

---

## 1. Overview

This milestone establishes the foundational image-processing and face-detection
pipeline for the Anti-Deepfake system. Before we can apply any adversarial
perturbations ("cloaking") to a person's face, we must reliably:

1. **Load** images from disk in the correct colour format.
2. **Detect and locate** the face within an image.
3. **Annotate** the original full image with a visible bounding box to confirm detection.
4. **Save** the annotated result to disk for inspection and downstream use.

> **Design update (v1.1):** The original design cropped and saved just the face region. This was replaced with drawing a labelled bounding box on the **full original image**, which is more useful for visual validation and aligns better with the re-composition step planned for Milestone 2 (where the adversarial face patch is pasted back into the original using the exact same bounding box coordinates).

This pipeline is the prerequisite for every future component (attack generation, evaluation, API backend, and UI frontend).

---

## 2. Files Created / Modified

| Action | File Path | Reason |
|--------|-----------|--------|
| **CREATED** | `src/ml_core/utils/image_loader.py` | Image I/O utility — `load_image`, `save_image`, and (v1.1) `draw_detections` |
| **MODIFIED** | `src/ml_core/utils/__init__.py` | Exports `load_image`, `save_image`, and `draw_detections` at the package level |
| **CREATED** | `src/ml_core/models/mtcnn_detector.py` | MTCNN face detector — encapsulates stateful ML model |
| **MODIFIED** | `src/ml_core/models/__init__.py` | Exports `FaceDetector` at the package level |
| **CREATED / UPDATED** | `test_pipeline.py` | Root-level smoke test — updated in v1.1 to annotate instead of crop |
| **CREATED** | `docs/milestone1_report.md` | This report |

---

## 3. Architecture Walkthrough

### 3.1 Three-Tier Modular Design

```
┌───────────────────────────────────────────────────────────────┐
│                   test_pipeline.py (runner)                    │
│  Orchestrates: load → detect → annotate → save                │
└──────────────────────┬────────────────────────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
┌───────────────────────┐    ┌──────────────────────────┐
│   ml_core/utils/      │    │   ml_core/models/         │
│   image_loader.py     │    │   mtcnn_detector.py       │
│                       │    │                           │
│   load_image()        │    │   FaceDetector            │
│   save_image()        │    │   └─ detect_and_crop()    │
│   draw_detections() ◄─┘    └──────────────────────────┘
└───────────────────────┘
```

Each layer has a **single responsibility**:
- `utils/` — Pure, stateless I/O and drawing helpers. No ML logic.
- `models/` — Stateful ML model wrappers. No direct file I/O.
- `test_pipeline.py` — Orchestration only. No business logic.

---

## 4. Component Deep-Dives

### 4.1 `image_loader.py` — Image I/O & Drawing Utility

**What it does:**
Handles the safe loading, saving, and annotation of images using **OpenCV**
(the industry-standard computer-vision library for Python).

**Key technical detail — BGR vs. RGB:**
OpenCV reads images in **BGR** (Blue-Green-Red) byte order by default, but deep learning
models like MTCNN expect **RGB** (Red-Green-Blue) order. Swapping these silently
corrupts colour channels and degrades model accuracy.

`load_image()` always converts BGR → RGB before returning, and `save_image()`
always converts RGB → BGR before writing, so the rest of the codebase can
safely assume **RGB everywhere**.

**Functions:**

| Function | Signature | Behaviour |
|----------|-----------|-----------|
| `load_image` | `(image_path: str) → np.ndarray` | Validates existence, reads with `cv2.imread`, converts BGR→RGB |
| `save_image` | `(image_array: np.ndarray, save_path: str) → None` | Creates dirs, converts RGB→BGR, writes with `cv2.imwrite` |
| `draw_detections` *(new v1.1)* | `(image_rgb_array, bounding_box, confidence, ...) → np.ndarray` | Draws a labelled green rectangle on a **copy** of the original image; never mutates the source |

**`draw_detections()` — Design Notes:**
- Works entirely in OpenCV (`cv2.rectangle`, `cv2.putText`), so no additional dependencies.
- Accepts configurable `box_color`, `box_thickness`, `font_scale`, and `label_text` for easy reuse.
- Returns a **new array** — the original image is never modified, preserving it for adversarial processing in Milestone 2.
- Draws a filled label pill above the box showing `Face: 100.00%`.

**Error handling:**
- `FileNotFoundError` — raised if the file path does not exist.
- `ValueError` — raised if OpenCV cannot decode/write the file, or if the bounding box does not contain exactly 4 coordinates.

---

### 4.2 `mtcnn_detector.py` — MTCNN Face Detector

**What it does:**
Wraps the **MTCNN** (Multi-task Cascaded Convolutional Network) model from the
`facenet-pytorch` library to detect and locate the primary (largest) face in an image.

**What is MTCNN?**
MTCNN is a deep learning model composed of three small neural networks — called
**P-Net** (Proposal), **R-Net** (Refinement), and **O-Net** (Output) — that
work in a cascade (sequential chain). Each network progressively refines the face location:

1. **P-Net** scans the entire image quickly to propose candidate face regions.
2. **R-Net** filters out false positives from those proposals.
3. **O-Net** produces the final precise bounding box and five facial landmarks (eye corners, nose tip, mouth corners).

This cascade makes MTCNN both **fast** (small networks) and **accurate** (iterative refinement).

**Critical design decision — returning the bounding box:**
`detect_and_crop()` returns **both** the internal crop *and* the bounding box `[x1, y1, x2, y2]`.
The crop is currently discarded by the pipeline (we only use the bounding box to annotate), but
it remains in the return signature because Milestone 2 will need it: once we apply an adversarial
perturbation to the isolated face region, we paste it back at these exact coordinates.

**Key configuration choices:**

| MTCNN Parameter | Value | Reason |
|-----------------|-------|--------|
| `keep_all` | `False` | Return only the single most-prominent face (privacy cloaking targets the subject, not background people) |
| `select_largest` | `True` | Among detected faces, select the largest one — most likely the primary subject |
| `post_process` | `False` | Return raw pixel values (uint8) instead of normalised tensors, so we stay in NumPy/OpenCV land |

---

### 4.3 `test_pipeline.py` — Smoke-Test Runner *(updated v1.1)*

**What it does:**
A standalone script at the project root that exercises all four pipeline steps
(load → detect → annotate → save) in sequence, with structured logging at each step.

**Why at the root?**
Running `python test_pipeline.py` from the project root requires no package
installation. The script bootstraps `sys.path` automatically so that `src.*` imports resolve.

**`.jpg` / `.jpeg` fallback:**
The script first checks for `data/raw/test.jpg`; if absent, it automatically
falls back to `data/raw/test.jpeg` — so either extension works without manual edits.

---

## 5. Data Flow Diagram *(v1.1)*

```
data/raw/test.jpeg  (or test.jpg)
       │
       ▼
 load_image()             ← cv2.imread → BGR→RGB conversion
       │
       │  image_rgb  (np.ndarray H×W×3, dtype uint8, RGB)
       ├──────────────────────────────────────┐
       ▼                                      │ (original kept intact)
 FaceDetector.detect_and_crop()               │
       │  1. NumPy → PIL Image                │
       │  2. MTCNN.detect() → boxes, probs    │
       │  3. Clamp coords to image bounds      │
       │                                      │
       │  bounding_box  ([x1, y1, x2, y2])    │
       │  _ (crop discarded)                  │
       ▼                                      │
 draw_detections() ◄──────────────────────────┘
       │  cv2.rectangle + cv2.putText on copy
       │
       │  annotated_rgb  (np.ndarray H×W×3)
       ▼
 save_image()             ← RGB→BGR → cv2.imwrite
       │
       ▼
data/output/annotated.jpg
```

---

## 6. Actual Smoke Test Run (Verified Output)

```
2026-06-18 00:33:14  INFO  __main__ — ============================================================
2026-06-18 00:33:14  INFO  __main__ — Anti-Deepfake — Face Detection Pipeline Smoke Test
2026-06-18 00:33:14  INFO  __main__ — ============================================================
2026-06-18 00:33:14  INFO  __main__ — STEP 1 — Loading raw image from: data/raw/test.jpeg
2026-06-18 00:33:14  INFO  image_loader — [INFO] Image loaded successfully. Shape: (365, 547, 3), dtype: uint8
2026-06-18 00:33:14  INFO  __main__ — STEP 1 — Complete. Image shape: (365, 547, 3)
2026-06-18 00:33:14  INFO  __main__ — STEP 2 — Initialising FaceDetector and running inference…
2026-06-18 00:33:14  INFO  mtcnn_detector — [INFO] MTCNN model loaded and ready.
2026-06-18 00:33:14  INFO  mtcnn_detector — [INFO] Face detected! Bounding box: [x1=297.7, y1=100.1, x2=354.0, y2=176.4] | Confidence: 1.0000
2026-06-18 00:33:14  INFO  __main__ — STEP 2 — Complete.
2026-06-18 00:33:14  INFO  __main__ — STEP 3 — Annotating image with bounding box…
2026-06-18 00:33:14  INFO  image_loader — [INFO] Bounding box drawn. Box: [297, 100, 354, 176] | Label: 'Face: 100.00%'
2026-06-18 00:33:14  INFO  __main__ — STEP 3 — Complete. Full-resolution image annotated.
2026-06-18 00:33:14  INFO  __main__ — STEP 4 — Saving annotated image to: data/output/annotated.jpg
2026-06-18 00:33:14  INFO  image_loader — [INFO] Image saved successfully to: data/output/annotated.jpg
2026-06-18 00:33:14  INFO  __main__ — ✓ Smoke test PASSED — annotated image saved successfully.
```

**Result:** `data/output/annotated.jpg` — the full original image with a green bounding box
and `Face: 100.00%` label drawn over the detected face.

---

## 7. How to Run

```bash
# Activate virtual environment
source .venv/bin/activate

# Place a portrait photo (either extension works)
cp /path/to/photo.jpg data/raw/test.jpg   # or test.jpeg

# Run the pipeline
python test_pipeline.py

# Output written to:
#   data/output/annotated.jpg
```

---

## 8. Error Scenarios Handled

| Scenario | Behaviour |
|----------|-----------|
| `test.jpg` / `test.jpeg` not found | `FileNotFoundError` raised, error logged, `sys.exit(1)` |
| File exists but corrupt / unsupported format | `ValueError` raised, error logged, `sys.exit(1)` |
| Image has no detectable face | `[WARNING]` logged, graceful `sys.exit(0)` |
| Output directory does not exist | Auto-created via `os.makedirs(..., exist_ok=True)` |
| `cv2.imwrite` returns `False` | `ValueError` raised, error logged, `sys.exit(1)` |
| Bounding box wrong shape | `ValueError` from `draw_detections`, error logged, `sys.exit(1)` |

---

## 9. Next Milestones (Preview)

| Milestone | Goal |
|-----------|------|
| **2** | Implement adversarial perturbation (FGSM / PGD attack) on the cropped face region |
| **3** | Re-compose the perturbed face back into the original image using the bounding box saved here |
| **4** | Build the FastAPI backend to expose the pipeline as a REST API |
| **5** | Build the HTML/Gradio frontend for user-facing image upload and download |

---

## 10. Dependencies Used

| Library | Purpose |
|---------|---------|
| `opencv-python` | Image I/O, colour space conversion, rectangle and text drawing |
| `facenet-pytorch` | MTCNN face detector |
| `Pillow` | PIL Image — required by facenet-pytorch's MTCNN input format |
| `numpy` | N-dimensional array operations |
| `torch` / `torchvision` | PyTorch backend for MTCNN inference |

> **Note:** For production deployments, pin all versions in `requirements.txt`
> (e.g. `facenet-pytorch==2.5.3`) to ensure reproducibility.
