import os
import sys

# ---------------------------------------------------------------------------
# Strict enforcement of thread pool isolation to prevent TF/PyTorch segfaults
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---------------------------------------------------------------------------
# CRITICAL IMPORT ORDER: PyTorch -> TorchVision -> TensorFlow
# On Python 3.13, loading TF before TorchVision causes a segfault during
# C-extension initialisation. This specific order bypasses the bug.
# ---------------------------------------------------------------------------
import torch
import torchvision

import uvicorn

if __name__ == "__main__":
    print("Starting AntiDeepfake API Server...")
    # Run the FastAPI app via Uvicorn
    uvicorn.run(
        "src.backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
