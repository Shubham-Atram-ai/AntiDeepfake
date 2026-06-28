"""
test_pgd_attack.py
-------------------
Unit tests for the PGD adversarial attack module.

Tests are fully self-contained — no disk I/O, no real face images required.
All tests use synthetic random tensors shaped to match FaceNet's native
160 × 160 input resolution.

Run from project root:
    python -m pytest tests/test_pgd_attack.py -v
"""

import logging

import numpy as np
import pytest
# pyrefly: ignore [missing-import]
import torch

# ---------------------------------------------------------------------------
# Path bootstrap — allows running pytest from the project root without
# installing the package.
# ---------------------------------------------------------------------------
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml_core.attacks.pgd_attack import PGDAttack  # noqa: E402

# ---------------------------------------------------------------------------
# Logging configuration for test runs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO logs during test runs for cleaner output
    format="[%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_H = 160
FACE_W = 160
BATCH_SIZE = 1
CHANNELS = 3
EPSILON = 0.02


# ---------------------------------------------------------------------------
# Shared fixture — instantiate once per module to avoid repeated weight loading
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def attack_instance() -> PGDAttack:
    """Instantiate a single PGDAttack for the entire test module.

    Using ``scope='module'`` means the FaceNet model is loaded only once,
    significantly reducing test suite runtime.

    Returns:
        A ready-to-use ``PGDAttack`` instance on CPU.
    """
    return PGDAttack(eps=EPSILON, device="cpu")


@pytest.fixture()
def random_face_tensor() -> torch.Tensor:
    """Create a reproducible random face tensor in [0, 1].

    Returns:
        Float tensor of shape ``(1, 3, 160, 160)`` with values in [0, 1].
    """
    torch.manual_seed(42)
    return torch.rand(BATCH_SIZE, CHANNELS, FACE_H, FACE_W, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Tests — output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Validate that the adversarial tensor preserves the input shape."""

    def test_attack_output_shape_matches_input(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Adversarial tensor must have the same shape as the input tensor."""
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        assert adversarial.shape == random_face_tensor.shape, (
            f"Shape mismatch: expected {random_face_tensor.shape}, "
            f"got {adversarial.shape}"
        )

    def test_embedding_output_shape(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Embedding must be a (1, 512) tensor (standard FaceNet output)."""
        embedding = attack_instance.get_embedding(random_face_tensor)
        assert embedding.shape == (1, 512), (
            f"Expected embedding shape (1, 512), got {embedding.shape}"
        )


# ---------------------------------------------------------------------------
# Tests — value range bounds
# ---------------------------------------------------------------------------

class TestValueRange:
    """Verify that the adversarial tensor remains within [0.0, 1.0]."""

    def test_adversarial_values_within_bounds(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """All adversarial pixel values must be clamped to [0.0, 1.0]."""
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        assert float(adversarial.min()) >= 0.0, (
            f"Adversarial tensor contains values below 0.0: {float(adversarial.min())}"
        )
        assert float(adversarial.max()) <= 1.0, (
            f"Adversarial tensor contains values above 1.0: {float(adversarial.max())}"
        )

    def test_adversarial_values_min_not_negative(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Explicitly guard against negative values in the clamped output."""
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        assert (adversarial < 0.0).sum().item() == 0, (
            "Found pixel values < 0.0 in adversarial tensor."
        )

    def test_adversarial_values_max_not_above_one(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Explicitly guard against values > 1.0 in the clamped output."""
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        assert (adversarial > 1.0).sum().item() == 0, (
            "Found pixel values > 1.0 in adversarial tensor."
        )


# ---------------------------------------------------------------------------
# Tests — perturbation generation
# ---------------------------------------------------------------------------

class TestPerturbationGeneration:
    """Verify that the PGD perturbation is actually applied."""

    def test_adversarial_differs_from_input(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """The adversarial tensor must differ from the original input."""
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        delta = (adversarial - random_face_tensor).abs()
        assert float(delta.max()) > 1e-6, (
            "Adversarial tensor is identical to the input — "
            "no perturbation was applied."
        )

    def test_perturbation_magnitude_bounded_by_epsilon(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Maximum perturbation (L-inf norm) must not exceed epsilon.

        Pixels that were already at 0.0 or 1.0 before adding epsilon may be
        clamped, so the max delta can be < epsilon.  The test checks ≤ epsilon
        with a small floating-point tolerance.
        """
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        delta_max = float((adversarial - random_face_tensor).abs().max())
        # Allow a tiny tolerance for floating-point rounding
        assert delta_max <= EPSILON + 1e-5, (
            f"Perturbation magnitude {delta_max:.6f} exceeds epsilon {EPSILON}."
        )

    def test_embedding_distance_increases_after_attack(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """The adversarial embedding must be further from the original.

        The PGD loss maximises cosine distance.  After a successful attack,
        the cosine similarity between original and adversarial embeddings
        should be lower than 1.0 (i.e. some distance was introduced).
        """
        
        import torch.nn.functional as F

        emb_orig = attack_instance.get_embedding(random_face_tensor)
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        emb_adv = attack_instance.get_embedding(adversarial)

        similarity = float(F.cosine_similarity(emb_orig, emb_adv, dim=1).mean())
        assert similarity < 1.0, (
            f"Cosine similarity did not decrease after attack: {similarity:.6f}"
        )


# ---------------------------------------------------------------------------
# Tests — tensor compatibility
# ---------------------------------------------------------------------------

class TestTensorCompatibility:
    """Validate tensor dtype, device, and detach status."""

    def test_output_is_float32(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Adversarial tensor must be float32 to match FaceNet input convention."""
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        assert adversarial.dtype == torch.float32, (
            f"Expected dtype torch.float32, got {adversarial.dtype}."
        )

    def test_output_does_not_require_grad(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Adversarial tensor must be detached (no gradient tracking)."""
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        assert not adversarial.requires_grad, (
            "Adversarial tensor should be detached from the computation graph."
        )

    def test_embedding_does_not_require_grad(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Embedding returned by get_embedding must be detached."""
        embedding = attack_instance.get_embedding(random_face_tensor)
        assert not embedding.requires_grad, (
            "Embedding should be detached from the computation graph."
        )


# ---------------------------------------------------------------------------
# Tests — CPU execution path
# ---------------------------------------------------------------------------

class TestCPUExecution:
    """Verify the complete attack executes on CPU without error."""

    def test_full_attack_completes_on_cpu(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """End-to-end attack on CPU must complete without raising exceptions."""
        try:
            adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
            assert adversarial is not None
        except Exception as exc:
            pytest.fail(f"PGD attack raised an exception on CPU: {exc}")

    def test_attack_returns_torch_tensor(
        self,
        attack_instance: PGDAttack,
        random_face_tensor: torch.Tensor,
    ) -> None:
        """Return type must be a torch.Tensor."""
        adversarial = attack_instance.attack(random_face_tensor, epsilon=EPSILON)
        assert isinstance(adversarial, torch.Tensor), (
            f"Expected torch.Tensor, got {type(adversarial).__name__}."
        )


# ---------------------------------------------------------------------------
# Tests — input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify that invalid inputs raise appropriate exceptions."""

    def test_attack_rejects_wrong_ndim(
        self,
        attack_instance: PGDAttack,
    ) -> None:
        """attack() must raise ValueError for non-4D tensors."""
        bad_tensor = torch.rand(3, FACE_H, FACE_W)  # Missing batch dimension
        with pytest.raises(ValueError, match="4-D"):
            attack_instance.attack(bad_tensor)

    def test_get_embedding_rejects_wrong_ndim(
        self,
        attack_instance: PGDAttack,
    ) -> None:
        """get_embedding() must raise ValueError for non-4D tensors."""
        bad_tensor = torch.rand(3, FACE_H, FACE_W)
        with pytest.raises(ValueError, match="4-D"):
            attack_instance.get_embedding(bad_tensor)


# ---------------------------------------------------------------------------
# Tests — reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Verify that results are reproducible given the same seed and input."""

    def test_attack_is_deterministic_on_cpu(
        self,
        attack_instance: PGDAttack,
    ) -> None:
        """Two calls with the same seed and input must produce identical output.

        Note: Determinism is guaranteed on CPU.  GPU execution with CUDA
        atomics may produce minor floating-point differences between runs.
        """
        torch.manual_seed(42)
        face1 = torch.rand(BATCH_SIZE, CHANNELS, FACE_H, FACE_W)
        adv1 = attack_instance.attack(face1, epsilon=EPSILON)

        torch.manual_seed(42)
        face2 = torch.rand(BATCH_SIZE, CHANNELS, FACE_H, FACE_W)
        adv2 = attack_instance.attack(face2, epsilon=EPSILON)

        assert torch.allclose(adv1, adv2, atol=1e-6), (
            "PGD attack output is not reproducible given the same seed and input."
        )
