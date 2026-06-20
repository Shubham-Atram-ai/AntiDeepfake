# AntiDeepfake Backend V1 — Complete Report

> **Audience**: This document is written to be accessible to beginners.
> Every concept is explained simply before diving into technical details.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [File and Folder Explanation](#3-file-and-folder-explanation)
4. [API Documentation](#4-api-documentation)
5. [ML Integration Details](#5-ml-integration-details)
6. [Processing Workflow](#6-processing-workflow)
7. [Startup Lifecycle](#7-startup-lifecycle)
8. [Testing](#8-testing)
9. [Running the Application](#9-running-the-application)
10. [Important Design Decisions](#10-important-design-decisions)
11. [Troubleshooting Guide](#11-troubleshooting-guide)
12. [Backend Summary](#12-backend-summary)

---

## 1. Project Overview

### What is AntiDeepfake?

AntiDeepfake is a privacy-protection tool that makes your face photos resistant to automated face-recognition systems. It works by applying a tiny, carefully crafted "noise" to face images — changes so small that humans cannot see them, but large enough to confuse AI systems that try to identify you.

### What Does the Backend Do?

The backend is a **REST API server** — a program that listens for requests over the internet and responds with results. The backend exposes the existing AI/ML pipeline via HTTP endpoints so that any frontend (React, mobile app, web form) can use it without knowing how the ML code works internally.

### How Does It Connect to the ML Pipeline?

The backend acts as a **thin orchestration layer** over the existing ML code:

```
You (browser/app)
       ↕ HTTP
FastAPI Backend  ←→  FaceDetector (MTCNN)
                 ←→  FGSMAttack   (InceptionResnetV1)
                 ←→  compute_ssim / compute_psnr
```

The backend **never reimplements** ML logic. It calls the exact same classes and functions used in `test_fgsm_pipeline.py`.

---

## 2. Architecture Overview

### Backend Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                    │
│                    (src/backend/main.py)                  │
├─────────────────────────────────────────────────────────┤
│  Middleware: CORS (allow all origins for dev)             │
├──────────────┬──────────────────────────────────────────┤
│  GET /       │  System endpoints                         │
│  GET /health │  (defined directly in main.py)            │
├──────────────┴──────────────────────────────────────────┤
│           Router: /api/v1  (routes/cloak.py)             │
│              POST /api/v1/cloak                          │
├─────────────────────────────────────────────────────────┤
│                    Service Layer                          │
│  pipeline_service.py  ← orchestrates everything          │
│  ├── detector_service.py   (wraps FaceDetector)          │
│  ├── attack_service.py     (wraps FGSMAttack)            │
│  └── metrics_service.py    (wraps compute_ssim/psnr)     │
├─────────────────────────────────────────────────────────┤
│                   Model Registry                          │
│           (core/model_registry.py)                       │
│   FaceDetector (loaded once) + FGSMAttack (loaded once)  │
├─────────────────────────────────────────────────────────┤
│                   ML Core (unchanged)                    │
│   src/ml_core/models/mtcnn_detector.py                   │
│   src/ml_core/attacks/fgsm_attack.py                     │
│   src/ml_core/evaluation/metrics.py                      │
└─────────────────────────────────────────────────────────┘
```

### Request Flow

```
Client sends POST /api/v1/cloak (image + epsilon)
    ↓
cloak.py route handler
    ↓ validates content type, reads bytes
pipeline_service.run_cloaking_pipeline()
    ├── _decode_image_bytes()        → RGB NumPy array
    ├── detect_face()                → face crop + bounding box
    ├── prepare_face_tensor()        → (1,3,160,160) float32 tensor
    ├── run_attack()                 → adversarial face uint8 RGB
    ├── _reconstruct_image()         → full cloaked image
    ├── evaluate_metrics()           → SSIM + PSNR scores
    └── _encode_image_base64()       → Base64 JPEG string
    ↓
JSON response → Client
```

### Service Layer Explanation

| Service | Responsibility |
|---|---|
| `detector_service.py` | Call MTCNN, handle "no face" case |
| `attack_service.py` | Prepare tensors, run FGSM, convert result back to image |
| `metrics_service.py` | Compute SSIM + PSNR, handle infinite PSNR |
| `pipeline_service.py` | Orchestrate all the above in the correct order |

---

## 3. File and Folder Explanation

### Complete New Structure

```
src/backend/
├── __init__.py
├── main.py                      ← FastAPI app, lifespan, root/health routes
├── core/
│   ├── config.py                ← Environment-variable configuration
│   └── model_registry.py       ← Singleton model cache
├── schemas/
│   ├── request.py               ← CloakRequest schema
│   └── response.py              ← HealthResponse, CloakResponse, ErrorResponse
├── services/
│   ├── detector_service.py      ← FaceDetector wrapper
│   ├── attack_service.py        ← FGSMAttack wrapper + tensor helpers
│   ├── metrics_service.py       ← SSIM/PSNR wrapper
│   └── pipeline_service.py      ← Main orchestration layer
└── routes/
    └── cloak.py                 ← POST /api/v1/cloak handler

tests/test_api.py                ← 21 API integration tests (new)
docs/backend_v1_report.md        ← This document
requirements.txt                 ← Updated with backend deps
```

### `main.py`
Creates FastAPI app, runs lifespan model loading, adds CORS, registers all routes.

### `core/config.py`
Reads HOST, PORT, LOG_LEVEL, DEFAULT_EPSILON from environment variables with sensible defaults.

### `core/model_registry.py`
Loads FaceDetector and FGSMAttack once at startup. Exposes them to services via a module-level singleton.

### `schemas/request.py`
CloakRequest model with epsilon validated in range (0.0, 1.0].

### `schemas/response.py`
HealthResponse, MetricsResponse, CloakResponse, ErrorResponse — stable JSON contracts for the frontend.

### `services/detector_service.py`
Wraps FaceDetector.detect_and_crop(). Raises HTTP 400 if no face is found.

### `services/attack_service.py`
Adapts prepare_face_tensor() and tensor_to_uint8_rgb() from test_fgsm_pipeline.py. Calls FGSMAttack.attack().

### `services/metrics_service.py`
Wraps compute_ssim() and compute_psnr(). Converts float('inf') to None for JSON safety.

### `services/pipeline_service.py`
The single orchestration entry point. Runs the complete 8-step pipeline in memory.

### `routes/cloak.py`
Thin HTTP handler for POST /api/v1/cloak. Validates content type, reads bytes, delegates to pipeline_service.

---

## 4. API Documentation

### Base URL
```
http://127.0.0.1:8000
```

### Interactive Docs
```
http://127.0.0.1:8000/docs
```

---

### GET /
**Tag**: System

**Response 200**:
```json
{"message": "AntiDeepfake API Running"}
```

---

### GET /health
**Tag**: Health

**Response 200**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "face_detector_loaded": true,
  "fgsm_engine_loaded": true
}
```

---

### POST /api/v1/cloak
**Tag**: Cloaking  
**Content-Type**: multipart/form-data

**Request Fields**:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | Yes | — | Face image (JPEG, PNG, BMP, WebP) |
| `epsilon` | float | No | 0.02 | FGSM perturbation strength (0.0, 1.0] |

**Example — curl**:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/cloak \
  -F "file=@photo.jpg" \
  -F "epsilon=0.02"
```

**Response 200**:
```json
{
  "success": true,
  "processing_time_ms": 742.1,
  "metrics": {
    "ssim": 0.9412,
    "psnr": 36.14
  },
  "cloaked_image_base64": "/9j/4AAQSkZJRgAB..."
}
```

**Decode the Base64 output**:
```python
import base64
jpeg_bytes = base64.b64decode(response.json()["cloaked_image_base64"])
with open("cloaked.jpg", "wb") as f:
    f.write(jpeg_bytes)
```

**Error Responses**:

| HTTP | error_code | Cause |
|---|---|---|
| 400 | INVALID_IMAGE | Non-image file, corrupt upload, empty file |
| 400 | NO_FACE_DETECTED | Image has no detectable face |
| 422 | — | Missing required file field |
| 500 | PROCESSING_ERROR | Internal ML pipeline failure |
| 503 | SERVICE_UNAVAILABLE | Models not loaded |

**Error body**:
```json
{
  "success": false,
  "error_code": "NO_FACE_DETECTED",
  "error": "No face detected in the uploaded image."
}
```

---

## 5. ML Integration Details

### FaceDetector Integration

```python
# How the backend uses it:
face_crop, bounding_box = detector.detect_and_crop(image_rgb)
# Returns: (h, w, 3) uint8 RGB crop  +  [x1, y1, x2, y2]
# None, None → raises HTTP 400 NO_FACE_DETECTED
```

### FGSMAttack Integration

```python
# Per-request epsilon overrides the instance default:
adversarial_tensor = attack.attack(face_tensor, epsilon=epsilon)
# Returns: (1, 3, 160, 160) float32 in [0, 1]
```

### Metrics Integration

```python
ssim = compute_ssim(original_rgb, cloaked_rgb)   # float in [-1, 1]
psnr = compute_psnr(original_rgb, cloaked_rgb)   # float in dB or inf
# inf converted to None for JSON safety
```

### Pipeline Reference Mapping

| test_fgsm_pipeline.py | Backend equivalent |
|---|---|
| `_prepare_face_tensor()` | `attack_service.prepare_face_tensor()` |
| `_tensor_to_uint8_rgb()` | `attack_service.tensor_to_uint8_rgb()` |
| `_reconstruct_image()` | `pipeline_service._reconstruct_image()` |
| `_evaluate_and_log()` | `metrics_service.evaluate_metrics()` |

---

## 6. Processing Workflow

```
1. Upload bytes arrive
   ↓
2. Validate MIME type (JPEG/PNG/BMP/WebP only)
   ↓
3. cv2.imdecode(bytes) → BGR  →  cvtColor → RGB array
   ↓
4. FaceDetector.detect_and_crop() → face_crop + bounding_box
   ↓
5. Resize to 160×160 (LANCZOS), normalise [0,1], make tensor (1,3,160,160)
   ↓
6. FGSMAttack.attack(tensor, epsilon) → adversarial (1,3,160,160) float32
   ↓
7. tensor → uint8 RGB → resize to bbox → paste into original canvas
   ↓
8. compute_ssim() + compute_psnr() on original vs cloaked
   ↓
9. cv2.imencode JPEG 95% → base64.b64encode → UTF-8 string
   ↓
10. Return JSON {success, processing_time_ms, metrics, cloaked_image_base64}
```

---

## 7. Startup Lifecycle

### Startup Sequence

```
uvicorn starts
    → lifespan() context manager runs
    → registry.load()
        → FaceDetector("cpu")       — MTCNN ready
        → FGSMAttack(0.02, "auto")  — InceptionResnetV1 ready (~100 MB)
    → Server accepts requests
```

### During Requests

```python
# Models retrieved from registry — no loading:
face_crop, bbox = detect_face(registry.face_detector, image_rgb)
adv_face, _    = run_attack(registry.fgsm_attack, tensor, epsilon)
```

### Shutdown Sequence

```
Ctrl+C received
    → lifespan() exits yield
    → registry.clear()
        → face_detector = None  (GC reclaims RAM)
        → fgsm_attack = None    (GC reclaims RAM)
    → Process exits cleanly
```

### Startup Failure

If models cannot load, the process exits immediately with `sys.exit(1)`.
A broken API that silently fails is worse than no API.

---

## 8. Testing

### Test File: `tests/test_api.py`

21 test cases across 6 classes:

| Class | Count | Tests |
|---|---|---|
| TestRootEndpoint | 3 | GET / → 200, correct message, JSON |
| TestHealthEndpoint | 5 | GET /health → 200, healthy, both models loaded, version |
| TestSuccessfulCloak | 4 | POST with face → 200, all fields, epsilon override, Base64 valid |
| TestInvalidFileType | 4 | PDF/text → 400, INVALID_IMAGE code |
| TestMissingFile | 2 | No file → 422, empty file → 400 |
| TestNoFaceImage | 3 | Solid image → 400, NO_FACE_DETECTED code |

### Running Tests

```bash
# From project root:

# All API tests:
python -m pytest tests/test_api.py -v

# All tests including existing ML tests:
python -m pytest tests/ -v

# Fast tests only (no face image needed):
python -m pytest tests/test_api.py::TestRootEndpoint \
                 tests/test_api.py::TestInvalidFileType \
                 tests/test_api.py::TestMissingFile \
                 tests/test_api.py::TestNoFaceImage -v
```

> **Note**: Happy-path tests (TestSuccessfulCloak) skip gracefully if no face image
> exists at `data/raw/test.jpg`. Add a face photo to run those tests.

> **Performance**: First run takes ~60-90s for model loading. Subsequent classes in
> the same session are fast because models are shared via session-scope fixture.

---

## 9. Running the Application

### Quick Start

```bash
# 1. Activate virtual environment
cd /home/shubham/AntiDeepfake
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
uvicorn src.backend.main:app --reload

# 4. Open Swagger UI
# → http://127.0.0.1:8000/docs
```

### Configuration via Environment Variables

```bash
export DEFAULT_EPSILON=0.05
export LOG_LEVEL=debug
uvicorn src.backend.main:app --reload
```

Or via `.env` file in the project root:
```
DEFAULT_EPSILON=0.02
LOG_LEVEL=info
HOST=127.0.0.1
PORT=8000
```

### Verify Startup

```bash
curl http://127.0.0.1:8000/health
# Should return: {"status":"healthy","version":"1.0.0",...}
```

---

## 10. Important Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Storage | In-memory only, no disk I/O | Security, speed, simplicity |
| Model loading | Once at startup, registry singleton | FaceNet weights take 5–30s to load |
| ML integration | Adapter pattern, no ML code modified | Preserves existing test suite |
| epsilon delivery | Form field (not query param) | Consistent with multipart/form-data |
| Output format | JPEG at 95% quality | 5–10× smaller than PNG; perturbation survives |
| Infinite PSNR | Returned as null | JSON has no infinity literal |
| Lifespan pattern | asynccontextmanager, not @on_event | @on_event is deprecated since FastAPI 0.93 |

---

## 11. Troubleshooting Guide

| Problem | Cause | Fix |
|---|---|---|
| "Failed to load FaceNet weights" | No internet on first run | Ensure connectivity; weights cached after first download |
| 404 on /api/v1/cloak | Missing /api/v1 prefix | Use full path: POST /api/v1/cloak |
| 422 on cloak request | Missing file field | Add file as multipart form field |
| "No face detected" on real photo | Low quality / extreme angle | Use clear, frontal, well-lit photo |
| Slow first test run (90s) | Models loading during TestClient startup | Expected; subsequent tests are fast |
| "ModuleNotFoundError: src" | Running from wrong directory | Run from /home/shubham/AntiDeepfake |
| httpx deprecation warning | httpx version mismatch | Warning only; tests still pass |

---

## 12. Backend Summary

### API Capabilities

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Confirm API is alive |
| `/health` | GET | Check model loading status |
| `/api/v1/cloak` | POST | Apply FGSM cloaking to a face image |
| `/docs` | GET | Interactive Swagger UI |
| `/redoc` | GET | ReDoc documentation |

### Implementation Status

| Phase | Status |
|---|---|
| Repository Analysis | ✅ Complete |
| Interface Discovery | ✅ Complete |
| Integration Design | ✅ Complete |
| Backend Implementation | ✅ Complete (13 new files) |
| Testing | ✅ Complete (21 test cases) |
| Documentation | ✅ Complete (this document) |

### Files Never Modified (ML Safety Rules)

- `src/ml_core/attacks/fgsm_attack.py` ✅ Untouched
- `src/ml_core/evaluation/metrics.py` ✅ Untouched
- `src/ml_core/models/mtcnn_detector.py` ✅ Untouched
- `src/ml_core/utils/image_loader.py` ✅ Untouched
- `test_fgsm_pipeline.py` ✅ Untouched
- `tests/test_fgsm_attack.py` ✅ Untouched
- `tests/test_metrics.py` ✅ Untouched
