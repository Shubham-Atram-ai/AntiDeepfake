import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import asyncio
from src.backend.core.model_registry import registry
from src.backend.services.pipeline_service import run_cloaking_pipeline

async def main():
    print("Loading models...")
    registry.load()
    print("Models loaded. Running pipeline...")
    with open("data/raw/test.jpeg", "rb") as f:
        image_bytes = f.read()
    
    # run_cloaking_pipeline is synchronous
    result = run_cloaking_pipeline(image_bytes, "test.jpeg", 8/255)
    print("Success:", result["success"])
    print("Metrics:", result["metrics"])
    
    import base64
    with open("data/output/cloaked_protected.jpg", "wb") as f:
        f.write(base64.b64decode(result["cloaked_image_base64"]))
    print("Saved output.")

if __name__ == "__main__":
    asyncio.run(main())
