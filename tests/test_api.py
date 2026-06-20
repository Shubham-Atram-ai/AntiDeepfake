"""
test_api.py
-----------
API integration tests for the AntiDeepfake Backend V1.

Tests use FastAPI's ``TestClient`` (backed by ``httpx``) which runs the
application in-process without starting a real server.  The ``lifespan``
context manager is exercised for every session-scoped client, so both
ML models are loaded exactly once for the entire test suite — matching
the production startup behaviour.

Test Coverage:
    1. Root endpoint (GET /)
    2. Health endpoint (GET /health)
    3. Successful image upload (POST /api/v1/cloak)
    4. Invalid file type upload (POST /api/v1/cloak)
    5. Empty file upload (POST /api/v1/cloak)
    6. Non-face image upload (POST /api/v1/cloak)

Run from the project root:
    python -m pytest tests/test_api.py -v
    python -m pytest tests/test_api.py -v -s   # show log output
"""

import io
import logging
import os
import sys

import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Path bootstrap — allows running pytest from the project root without
# installing the package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backend.main import app  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — suppress verbose ML model output during test runs
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client() -> TestClient:
    """Create a single TestClient for the entire test session.

    ``scope='session'`` ensures the lifespan startup (model loading) runs
    exactly once, just like production, keeping the test suite fast.

    Returns:
        A ``TestClient`` instance backed by the FastAPI application with
        all ML models loaded.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _make_jpeg_bytes(
    height: int = 100,
    width: int = 100,
    fill_color: tuple = (120, 80, 60),
) -> bytes:
    """Create a minimal in-memory JPEG image as raw bytes.

    Generates a solid-colour NumPy array and encodes it to JPEG bytes using
    OpenCV.  No disk I/O is performed.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        fill_color: BGR colour tuple used to fill the image.

    Returns:
        Raw JPEG bytes ready to upload as a multipart form file.
    """
    img = np.full((height, width, 3), fill_color, dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


def _make_face_jpeg_bytes() -> bytes:
    """Generate a synthetic face-like JPEG for the happy-path cloak test.

    If the project contains a test face image at ``data/raw/test.jpg`` or
    ``data/raw/test.jpeg``, that file is used for a more realistic test.
    Otherwise, falls back to a solid-colour synthetic image.

    Returns:
        Raw JPEG bytes of a face (real or synthetic).
    """
    for candidate in ("data/raw/test.jpg", "data/raw/test.jpeg"):
        path = os.path.join(PROJECT_ROOT, candidate)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()

    # Fallback: synthetic image — will cause NO_FACE_DETECTED response,
    # but the route still handles it gracefully (tested in test 6 separately).
    return _make_jpeg_bytes(160, 160, (100, 130, 90))


# ---------------------------------------------------------------------------
# Test 1 — Root endpoint
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    """Verify the GET / endpoint returns the expected greeting."""

    def test_root_returns_200(self, client: TestClient) -> None:
        """GET / must return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}"
        )

    def test_root_response_body(self, client: TestClient) -> None:
        """GET / must return the standard API greeting message."""
        response = client.get("/")
        body = response.json()
        assert "message" in body, "Response missing 'message' key."
        assert body["message"] == "AntiDeepfake API Running", (
            f"Unexpected message: {body['message']}"
        )

    def test_root_content_type_is_json(self, client: TestClient) -> None:
        """GET / must return application/json content type."""
        response = client.get("/")
        assert "application/json" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Test 2 — Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Verify the GET /health endpoint returns complete model status."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health must return HTTP 200 (all models loaded)."""
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. "
            f"Body: {response.text}"
        )

    def test_health_status_is_healthy(self, client: TestClient) -> None:
        """Health status must be 'healthy' when both models are loaded."""
        response = client.get("/health")
        body = response.json()
        assert body.get("status") == "healthy", (
            f"Expected 'healthy', got '{body.get('status')}'. "
            "Check that both ML models loaded during startup."
        )

    def test_health_face_detector_loaded(self, client: TestClient) -> None:
        """face_detector_loaded must be True after successful startup."""
        response = client.get("/health")
        body = response.json()
        assert body.get("face_detector_loaded") is True, (
            "face_detector_loaded is False — MTCNN did not load."
        )

    def test_health_fgsm_engine_loaded(self, client: TestClient) -> None:
        """fgsm_engine_loaded must be True after successful startup."""
        response = client.get("/health")
        body = response.json()
        assert body.get("fgsm_engine_loaded") is True, (
            "fgsm_engine_loaded is False — InceptionResnetV1 did not load."
        )

    def test_health_version_present(self, client: TestClient) -> None:
        """Health response must include a non-empty version string."""
        response = client.get("/health")
        body = response.json()
        assert "version" in body, "Health response missing 'version' key."
        assert body["version"], "Health response version is empty."


# ---------------------------------------------------------------------------
# Test 3 — Successful image upload (happy path)
# ---------------------------------------------------------------------------

class TestSuccessfulCloak:
    """Verify the cloak endpoint processes a face image successfully.

    This test requires either:
    - A real face image at ``data/raw/test.jpg`` or ``data/raw/test.jpeg``.
    - Or it will be skipped gracefully if only a synthetic image is available
      (which MTCNN would correctly reject as NO_FACE_DETECTED).
    """

    def test_cloak_with_real_face_returns_200(self, client: TestClient) -> None:
        """POST /api/v1/cloak with a real face image must return HTTP 200."""
        # Use real face if available, otherwise skip this test.
        face_path = None
        for candidate in ("data/raw/test.jpg", "data/raw/test.jpeg"):
            path = os.path.join(PROJECT_ROOT, candidate)
            if os.path.exists(path):
                face_path = path
                break

        if face_path is None:
            pytest.skip(
                "No face image found at data/raw/test.jpg — "
                "happy-path test skipped. Add a face image to run this test."
            )

        with open(face_path, "rb") as fh:
            image_bytes = fh.read()

        response = client.post(
            "/api/v1/cloak",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Body: {response.text}"
        )

    def test_cloak_success_response_has_expected_keys(
        self, client: TestClient
    ) -> None:
        """Successful cloak response must contain all required keys."""
        face_path = None
        for candidate in ("data/raw/test.jpg", "data/raw/test.jpeg"):
            path = os.path.join(PROJECT_ROOT, candidate)
            if os.path.exists(path):
                face_path = path
                break

        if face_path is None:
            pytest.skip("No face image at data/raw/test.jpg — skipping.")

        with open(face_path, "rb") as fh:
            image_bytes = fh.read()

        response = client.post(
            "/api/v1/cloak",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()

        assert body.get("success") is True, "success field must be True."
        assert "processing_time_ms" in body, "Missing processing_time_ms."
        assert "metrics" in body, "Missing metrics."
        assert "ssim" in body["metrics"], "Missing metrics.ssim."
        assert "psnr" in body["metrics"], "Missing metrics.psnr."
        assert "cloaked_image_base64" in body, "Missing cloaked_image_base64."
        assert isinstance(body["cloaked_image_base64"], str), (
            "cloaked_image_base64 must be a string."
        )
        assert len(body["cloaked_image_base64"]) > 0, (
            "cloaked_image_base64 is empty."
        )

    def test_cloak_custom_epsilon(self, client: TestClient) -> None:
        """POST /api/v1/cloak with custom epsilon must succeed."""
        face_path = None
        for candidate in ("data/raw/test.jpg", "data/raw/test.jpeg"):
            path = os.path.join(PROJECT_ROOT, candidate)
            if os.path.exists(path):
                face_path = path
                break

        if face_path is None:
            pytest.skip("No face image at data/raw/test.jpg — skipping.")

        with open(face_path, "rb") as fh:
            image_bytes = fh.read()

        response = client.post(
            "/api/v1/cloak",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"epsilon": "0.05"},
        )
        assert response.status_code == 200, (
            f"Expected 200 with epsilon=0.05, got {response.status_code}. "
            f"Body: {response.text}"
        )

    def test_cloak_base64_is_decodable(self, client: TestClient) -> None:
        """The cloaked_image_base64 field must be valid decodable Base64."""
        import base64

        face_path = None
        for candidate in ("data/raw/test.jpg", "data/raw/test.jpeg"):
            path = os.path.join(PROJECT_ROOT, candidate)
            if os.path.exists(path):
                face_path = path
                break

        if face_path is None:
            pytest.skip("No face image at data/raw/test.jpg — skipping.")

        with open(face_path, "rb") as fh:
            image_bytes = fh.read()

        response = client.post(
            "/api/v1/cloak",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

        b64_str = response.json()["cloaked_image_base64"]
        try:
            decoded = base64.b64decode(b64_str)
            assert len(decoded) > 0, "Decoded Base64 payload is empty."
        except Exception as exc:
            pytest.fail(f"cloaked_image_base64 is not valid Base64: {exc}")


# ---------------------------------------------------------------------------
# Test 4 — Invalid file type upload
# ---------------------------------------------------------------------------

class TestInvalidFileType:
    """Verify that non-image file uploads are rejected with HTTP 400."""

    def test_text_file_is_rejected(self, client: TestClient) -> None:
        """Uploading a plain text file must return HTTP 400."""
        response = client.post(
            "/api/v1/cloak",
            files={"file": ("document.txt", b"This is not an image.", "text/plain")},
        )
        assert response.status_code == 400, (
            f"Expected 400, got {response.status_code}."
        )

    def test_pdf_is_rejected(self, client: TestClient) -> None:
        """Uploading a PDF file must return HTTP 400."""
        response = client.post(
            "/api/v1/cloak",
            files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 400

    def test_invalid_type_error_code(self, client: TestClient) -> None:
        """Rejected file must include 'INVALID_IMAGE' error_code in the body."""
        response = client.post(
            "/api/v1/cloak",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        body = response.json()
        # FastAPI may wrap the detail in a 'detail' key via HTTPException
        detail = body.get("detail", body)
        assert detail.get("error_code") == "INVALID_IMAGE", (
            f"Expected error_code INVALID_IMAGE, got: {detail}"
        )

    def test_invalid_type_success_is_false(self, client: TestClient) -> None:
        """Rejected file response must have success=False."""
        response = client.post(
            "/api/v1/cloak",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        body = response.json()
        detail = body.get("detail", body)
        assert detail.get("success") is False


# ---------------------------------------------------------------------------
# Test 5 — Missing / empty file upload
# ---------------------------------------------------------------------------

class TestMissingFile:
    """Verify appropriate errors when the file field is missing or empty."""

    def test_no_file_returns_422(self, client: TestClient) -> None:
        """POST /api/v1/cloak with no file must return HTTP 422."""
        response = client.post("/api/v1/cloak")
        assert response.status_code == 422, (
            f"Expected 422 Unprocessable Entity, got {response.status_code}."
        )

    def test_empty_file_returns_400(self, client: TestClient) -> None:
        """POST /api/v1/cloak with an empty image file must return HTTP 400."""
        response = client.post(
            "/api/v1/cloak",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400, (
            f"Expected 400 for empty file, got {response.status_code}."
        )


# ---------------------------------------------------------------------------
# Test 6 — No-face image upload
# ---------------------------------------------------------------------------

class TestNoFaceImage:
    """Verify that images without a detectable face are rejected with HTTP 400."""

    def test_solid_color_image_no_face(self, client: TestClient) -> None:
        """A solid-colour image with no face must return HTTP 400."""
        image_bytes = _make_jpeg_bytes(200, 200, (50, 200, 100))
        response = client.post(
            "/api/v1/cloak",
            files={"file": ("solid.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 400, (
            f"Expected 400 (no face), got {response.status_code}. "
            f"Body: {response.text}"
        )

    def test_no_face_error_code(self, client: TestClient) -> None:
        """No-face response must include 'NO_FACE_DETECTED' error_code."""
        image_bytes = _make_jpeg_bytes(200, 200, (50, 200, 100))
        response = client.post(
            "/api/v1/cloak",
            files={"file": ("solid.jpg", image_bytes, "image/jpeg")},
        )
        body = response.json()
        detail = body.get("detail", body)
        assert detail.get("error_code") == "NO_FACE_DETECTED", (
            f"Expected NO_FACE_DETECTED, got: {detail}"
        )

    def test_no_face_success_is_false(self, client: TestClient) -> None:
        """No-face response must have success=False."""
        image_bytes = _make_jpeg_bytes(200, 200, (50, 200, 100))
        response = client.post(
            "/api/v1/cloak",
            files={"file": ("solid.jpg", image_bytes, "image/jpeg")},
        )
        body = response.json()
        detail = body.get("detail", body)
        assert detail.get("success") is False
