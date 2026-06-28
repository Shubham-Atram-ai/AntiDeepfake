# AntiDeepfake Phase 2: Deep Technical Audit (ML Core)

**Author:** Principal Machine Learning Security (ML-Sec) Engineer & Technical Auditor  
**Scope:** Machine Learning Core & Backend Orchestration Integration  
**Date:** June 26, 2026  

---

## 1. Executive Summary

This document serves as the official technical reference and deep-dive forensic audit of the **AntiDeepfake** repository, with a primary focus on its Machine Learning Core (`src/ml_core`). The system is designed to provide an API for Adversarial Image Cloaking, applying Projected Gradient Descent (PGD) perturbations to facial images to disrupt unauthorized automated face-recognition systems. The architecture transitions from a Phase 1 prototype (MTCNN + FGSM) to a more robust Phase 2 implementation utilizing RetinaFace and multi-step PGD attacks driven by a novel "Fake Logits" wrapper over FaceNet.

The codebase is highly disciplined, strictly typed, and separates ML algorithms from backend routing concerns through a clean Service Adapter pattern. This audit details the architectural decisions, algorithmic implementations, and systemic workflows that power the API.

## 2. Scope and Methodology

This audit is an evidence-based forensic analysis. It does not speculate on intent but derives its conclusions strictly from the current state of the source code. 

**In Scope:**
* `src/ml_core/*`: Models, Attacks, Evaluation, Utils.
* `src/backend/*`: Core configurations, Model Registry, Orchestration Services.
* `tests/*`: Unit and Integration test coverage for the ML core.

**Out of Scope:**
* Frontend implementation (`src/frontend/*`).
* Refactoring or modifying the source code.

## 3. System Architecture Overview

The AntiDeepfake system operates as a stateless HTTP service built on FastAPI. It follows a modular, three-tier architecture:
1. **API / Transport Layer:** Handles HTTP requests and input validation (`src/backend/routes/cloak.py`).
2. **Service Layer:** Orchestrates the complex ML pipeline, bridging the HTTP context with the raw ML algorithms (`src/backend/services/pipeline_service.py`).
3. **Machine Learning Core:** A purely functional layer that performs heavy tensor computations, face detection, and adversarial perturbation (`src/ml_core`).

A crucial architectural decision is the **Singleton Model Registry** (`src/backend/core/model_registry.py`), which loads the heavy `FaceDetector` and `PGDAttack` models into memory strictly once during the FastAPI application `lifespan` startup, ensuring subsequent API requests are unburdened by model initialization latency.

## 4. Repository Structure and Inventory

The repository is structured as a standard Python application with separated domains. Based on the 4,683 Lines of Code (LOC) inventory:

* **`/src/ml_core/`**: The algorithmic heart.
  * `/models/`: `retinaface_detector.py` (Phase 2), `mtcnn_detector.py` (Phase 1).
  * `/attacks/`: `pgd_attack.py` (Active), `fgsm_attack.py` (Legacy).
  * `/evaluation/`: `metrics.py` (SSIM, PSNR).
  * `/utils/`: `image_loader.py` (I/O, Annotations).
* **`/src/backend/`**: Application orchestration.
  * `/core/`: `config.py` (Pydantic settings), `model_registry.py`.
  * `/routes/`, `/schemas/`, `/services/`.
* **`/tests/`**: Comprehensive test suites (API, metrics, PGD execution).
* **Root Scripts**: `test_fgsm_pipeline.py`, `test_pipeline.py`.

## 5. Machine Learning Core Design Philosophy

The ML Core is designed to be **framework-agnostic at the boundaries** but deeply integrated with **PyTorch** internally. 
* **Input/Output Contract:** The core ML components (detectors, image utilities) strictly communicate via `numpy.ndarray` (RGB, `uint8`, HWC) at their external boundaries.
* **Internal Tensors:** Once inside the `PGDAttack` engine, data is aggressively mapped to `torch.Tensor` (Float32, CHW, normalized [0.0, 1.0]).
* **Immutability:** Functions like `draw_detections` and internal detection crops intentionally operate on `.copy()` arrays to prevent unintended mutation of the source image buffer.

## 6. Data Flow and Orchestration

The end-to-end data flow is orchestrated entirely by `run_cloaking_pipeline()` within `src/backend/services/pipeline_service.py`. The data lifecycle is:
1. **Ingest & Decode:** `MultipartForm` bytes → OpenCV `imdecode` → `BGR` to `RGB` conversion (`uint8` numpy array).
2. **Face Detection:** `detector_service.detect_face()` yields a cropped face array and absolute bounding box coordinates.
3. **Tensor Preparation:** `attack_service.prepare_face_tensor()` scales the crop to 160×160 (LANCZOS resampling) and translates it to a `(1, 3, 160, 160)` float32 PyTorch tensor.
4. **Adversarial Perturbation:** `registry.pgd_attack` generates a detached adversarial tensor of identical shape.
5. **Reconstruction:** The adversarial tensor is converted back to an RGB `uint8` array, scaled up to the original bounding box dimensions, and inserted into a full-resolution clone of the original image (`_reconstruct_image`).
6. **Evaluation:** `metrics_service.evaluate_metrics()` calculates SSIM and PSNR between the original and reconstructed clone.
7. **Serialization:** Reconstructed image → `RGB` to `BGR` → JPEG Encoding (95% quality) → Base64 string payload.

## 7. Face Detection Subsystem (Phase 1 & Phase 2)

The system maintains two face detectors sharing an identical public API (`detect_and_crop(self, image_rgb_array)`), allowing them to be swapped invisibly to the caller.
Both models clamp bounding boxes to absolute image bounds to prevent out-of-bounds array slicing errors.

## 8. RetinaFace Integration Analysis (Active)

Implemented in `src/ml_core/models/retinaface_detector.py`. This is the Phase 2 production detector.
* **Underlying Framework:** TensorFlow/Keras (via the `retinaface` PIP package).
* **Initialization:** Lazy-loaded. The heavy Keras model is loaded upon the first invocation of `detect_and_crop()`, bypassing standard import-time penalties.
* **Logic:** Accepts NumPy arrays directly. Iterates over detected faces, selecting the one with the maximum `score`.
* **Strengths:** Highly robust against occlusions, non-frontal angles, and varied lighting compared to MTCNN.

## 9. MTCNN Legacy Analysis (Deprecated)

Implemented in `src/ml_core/models/mtcnn_detector.py`. 
* **Underlying Framework:** PyTorch (via `facenet-pytorch`).
* **Logic:** Instantiated with `keep_all=False, select_largest=True`. Converts numpy arrays to `PIL.Image` prior to inference.
* **Status:** Fully functional but superseded by RetinaFace due to superior detection capabilities on diverse datasets.

## 10. FaceNet Embedding Space Dynamics

Both adversarial attacks (FGSM and PGD) utilize the **FaceNet** model (`InceptionResnetV1` pretrained on `vggface2`). FaceNet is an embedding model, mapping an input image to a 512-dimensional L2-normalized vector space, rather than a classification model outputting discrete probabilities.
* **The Mathematical Challenge:** Standard adversarial attacks maximize Cross-Entropy Loss to push a classifier towards a wrong class. In an embedding space, one might try to maximize the Cosine Similarity to a different identity, but as noted in `fgsm_attack.py`, `∂cos(a,a)/∂a = 0`, causing zero gradients at initialization.
* **The Solution:** The system uses Mean Squared Error (MSE) loss directed at a random unit-sphere target in $\mathbb{R}^{512}$, physically pulling the identity embedding across the hyper-sphere.

## 11. PGD Adversarial Attack Engine (Phase 2)

Implemented in `src/ml_core/attacks/pgd_attack.py`. This engine drives the production cloaking.
* **Library:** Leverages the robust `torchattacks.PGD` implementation.
* **Hyperparameters:** Configured for strong perturbation: `eps=8/255` (~0.0314), `alpha=2/255` (~0.0078), `steps=10`.
* **Reproducibility:** Fixes seeds via `torch.manual_seed(42)` and `np.random.seed(42)` to ensure identical target sphere vectors for identical faces. It saves and restores the RNG state dynamically during `attack()` to isolate state mutation.
* **Execution:** Calls `self._pgd(x, labels)` returning a detached tensor. Requires gradients to be enabled dynamically during execution while maintaining FaceNet weights as frozen (`requires_grad_(False)`).

## 12. The Fake Logits Trick (`_EmbeddingLossWrapper`)

To bridge the gap between FaceNet's embedding output and `torchattacks`' expectation of a classification model, the developer implemented a brilliant proxy wrapper: `_EmbeddingLossWrapper(nn.Module)`.
* **Mechanism:** It computes the MSE between the current FaceNet embedding and the pre-computed random target. It then outputs a tensor of shape `(B, 2)` containing `[-mse, +mse]`.
* **Exploitation:** `torchattacks` assumes these are 2-class logits. When instructed to attack label `0` (the true class, which corresponds to `-mse`), it attempts to maximize the cross-entropy loss by driving up the value of class `1` (`+mse`). Maximizing `+mse` forces the network to maximize the geometric MSE distance, driving the embedding away from the original identity towards the random target.

## 13. FGSM Legacy Attack Engine (Phase 1)

Implemented in `src/ml_core/attacks/fgsm_attack.py`.
* **Methodology:** A custom, manual implementation of the Fast Gradient Sign Method (FGSM).
* **Execution:** Enables gradients, calculates `F.mse_loss(emb_adv, target)`, executes `loss.backward()`, and perturbs via `x_adv = x_adv + eps * grad_sign`.
* **Status:** Marked as "legacy, kept for reference". It validates the core math but is significantly weaker than the iterative PGD approach.

## 14. Image Processing and I/O Pipelines

Implemented in `src/ml_core/utils/image_loader.py`.
* **Color Space Consistency:** OpenCV inherently uses `BGR`. The pipeline rigorously enforces an immediate conversion to `RGB` upon read, and `RGB` to `BGR` immediately prior to write or encode.
* **Resampling:** `attack_service.prepare_face_tensor` uses `PIL.Image.Resampling.LANCZOS` for high-quality downscaling to FaceNet's native `160x160`.
* **Defensive Clamping:** During E2E reconstruction (`pipeline_service.py`), bounding boxes are defensively double-clamped to absolute canvas bounds: `x1 = max(0, min(int(bounding_box[0]), img_w - 1))`. This prevents fatal `IndexError` slicing exceptions if bounding box integers slightly exceed matrix dimensions.

## 15. Quality Evaluation Metrics (SSIM & PSNR)

Implemented in `src/ml_core/evaluation/metrics.py`.
* **SSIM (Structural Similarity Index):** Calculated using `skimage.metrics.structural_similarity` with `win_size=7` and `channel_axis=2`. Provides a perceptual quality score `[-1.0, 1.0]`.
* **PSNR (Peak Signal-to-Noise Ratio):** Computes exact MSE manually to safely capture identical images (`MSE=0`), returning `float('inf')`.
* **Safety:** In the backend `metrics_service.py`, `float('inf')` is explicitly trapped and converted to JSON-compliant `null` to prevent serialization crashes during FastAPI responses.

## 16. Backend Integration and Model Registry

The backend strictly conforms to Pydantic validation via `src/backend/schemas/`.
* **Registry (`src/backend/core/model_registry.py`):** Encapsulates the Heavy ML state. Models are instanced at server startup and cleared during shutdown, managed by FastAPI's `@asynccontextmanager def lifespan(app)`.
* **Service Segregation:** The `cloak.py` API route knows nothing of PyTorch or OpenCV; it solely unpacks multi-part forms and delegates byte streams to the `pipeline_service`.

## 17. Thread Safety and Environment Constraints

Machine Learning frameworks combined with web servers are notoriously prone to thread-contention Segmentation Faults. The application proactively neuters this risk:
* **Constraint Directives:** In `main.py` and `tests/__init__.py`, `os.environ` is injected at import time:
  * `OMP_NUM_THREADS="1"`
  * `KMP_DUPLICATE_LIB_OK="TRUE"`
  * `TF_ENABLE_ONEDNN_OPTS="0"`
* **Rationale:** Prevents PyTorch's OpenMP backend and RetinaFace's TensorFlow/OneDNN backend from spawning competing thread pools inside the asynchronous ASGI worker (uvicorn).

## 18. E2E Pipeline Analysis (`test_fgsm_pipeline.py`)

This root-level smoke test exercises the exact physical code paths executed in production.
* Traces: `_load_raw_image` → `_detect_face` (RetinaFace) → `_prepare_face_tensor` → `_run_pgd` → `_reconstruct_image` → `_evaluate_and_log`.
* Log outputs confirm successful evasion of the zero-gradient cosine problem, noting measurable drops in Cosine Similarity between original and adversarial embeddings while maintaining PSNR > 30dB.

## 19. Testing and Validation Coverage

The `tests/` directory boasts exceptionally high engineering rigor:
* **`test_api.py` (21 Tests):** Uses `FastAPI.TestClient` with the session lifespan enabled, confirming 200 OKs on valid uploads, and explicit 400 Bad Requests on text files, PDFs, or images containing no faces (yielding code `NO_FACE_DETECTED`).
* **`test_metrics.py` (20 Tests):** Exhaustive boundary validation of `uint8` limits, shape mismatches, and `inf` handling.
* **`test_pgd_attack.py` (16 Tests):** Validates L-infinity norm perturbation bounds (`≤ epsilon + 1e-5`), ensuring the detached adversarial output is structurally constrained to `[0.0, 1.0]`.

## 20. Strengths and Security Posture

* **Exceptional Boundary Validation:** The core explicitly guards against incorrect dimensions and types using precise `ValueError` exceptions rather than allowing PyTorch to crash with cryptic internal C++ tracebacks.
* **Idempotency:** The pipeline never mutates input structures.
* **Graceful Degradation:** The `/health` endpoint checks `registry.is_ready`, ensuring load balancers will not route traffic to workers where model weights failed to download.

## 21. Vulnerabilities and Recommendations

1. **Resolution Degradation Artifacts:** The current implementation reconstructs the `160x160` adversarial tensor back to the original bounding box size. If the original face crop is $1000 \times 1000$ pixels, upscaling a `160x160` tensor via `cv2.INTER_LANCZOS4` will introduce severe blurring, rendering the deepfake protection obvious.
   * *Recommendation:* Implement a high-resolution patching strategy (e.g., iterative blending) or operate PGD at the native resolution of the crop, though the latter would require reshaping the FaceNet architecture or applying global average pooling adjustments.
2. **Device Hardcoding:** `FaceDetector` defaults to `device="cpu"` statically in `ModelRegistry.load()`, ignoring the presence of CUDA/MPS accelerators, while `PGDAttack` dynamically uses `device="auto"`.
   * *Recommendation:* Sync configuration so both models pull the device execution target from `src.backend.core.config.Settings`.
3. **Dead Code:** `request.py` defines `CloakRequest`, but the `/api/v1/cloak` route uses `Form()` injections directly due to limitations combining `UploadFile` with Pydantic JSON bodies in FastAPI.
   * *Recommendation:* Either utilize `Depends()` for form validation or document the schema strictly for Swagger reference.

---
**Audit Concluded.** All ML models, pipelines, and validation layers have been inspected and verified against the repository codebase.
