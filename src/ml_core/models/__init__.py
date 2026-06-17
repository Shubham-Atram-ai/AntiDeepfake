"""
src/ml_core/models/__init__.py
-------------------------------
Public API for the ml_core.models package.

Exposes the FaceDetector class so callers can import it directly from the
package:

    from src.ml_core.models import FaceDetector
"""

from src.ml_core.models.mtcnn_detector import FaceDetector

__all__ = ["FaceDetector"]
