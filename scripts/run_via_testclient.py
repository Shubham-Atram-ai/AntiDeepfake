import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import torch

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from src.backend.main import app
import base64

def run():
    print("Starting client...")
    client = TestClient(app)
    print("Reading image...")
    with open("data/raw/test.jpg", "rb") as f:
        file_bytes = f.read()
    
    print("Sending POST request to /api/v1/cloak...")
    response = client.post("/api/v1/cloak", files={"file": ("test.jpg", file_bytes, "image/jpeg")})
    
    if response.status_code == 200:
        print("Success!")
        data = response.json()
        print("Metrics:", data["metrics"])
        with open("data/output/cloaked_protected.jpg", "wb") as f:
            f.write(base64.b64decode(data["cloaked_image_base64"]))
        print("Image saved to data/output/cloaked_protected.jpg")
    else:
        print("Failed:", response.status_code, response.text)

if __name__ == "__main__":
    run()
