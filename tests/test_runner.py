import pytest
from fastapi.testclient import TestClient
from src.backend.main import app
import base64
import os

def test_process_user_image():
    with TestClient(app) as client:
        with open("data/raw/test.jpg", "rb") as f:
            file_bytes = f.read()
        response = client.post("/api/v1/cloak", files={"file": ("test.jpg", file_bytes, "image/jpeg")})
        assert response.status_code == 200, response.text
        data = response.json()
        os.makedirs("data/output", exist_ok=True)
        with open("data/output/cloaked_protected.jpg", "wb") as f:
            f.write(base64.b64decode(data["cloaked_image_base64"]))
        print("\nSuccessfully cloaked! Metrics:", data["metrics"])
