# tests/__init__.py
# -----------------
# Marks the tests/ directory as a Python package.
# Required for pytest test discovery and for relative imports to work
# correctly when running ``python -m pytest`` from the project root.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torchvision
