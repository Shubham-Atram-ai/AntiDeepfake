"""
src/ml_core/attacks/__init__.py
--------------------------------
Public API for the ml_core.attacks package.

Exposes both attack classes so callers can import directly from the package:

    from src.ml_core.attacks import PGDAttack
    from src.ml_core.attacks import FGSMAttack  # legacy, kept for reference
"""

from src.ml_core.attacks.pgd_attack import PGDAttack
from src.ml_core.legacy.fgsm_attack import FGSMAttack

__all__ = ["PGDAttack", "FGSMAttack"]
