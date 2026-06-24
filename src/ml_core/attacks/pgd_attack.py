"""
pgd_attack.py
-------------
Projected Gradient Descent (PGD) adversarial attack implemented in
FaceNet's **embedding space** using the ``torchattacks`` library.

FaceNet (``InceptionResnetV1``) is an embedding model, not a classifier.
Standard classification-based PGD (which maximises cross-entropy w.r.t. a
class label) is therefore not directly applicable.  Instead this module:

1.  Wraps FaceNet in a thin ``nn.Module`` (``_EmbeddingLossWrapper``) that:
    - Normalises the ``[0, 1]`` input tensor to the ``[-1, 1]`` range that
      FaceNet expects.
    - Runs the FaceNet forward pass to obtain a 512-d identity embedding.
    - Returns **logits** shaped ``(batch, 2)`` constructed from the MSE loss
      toward a fixed random unit-sphere target:

          target = normalize(randn(512))        # random unit vector in R^512
          loss   = MSE(embedding, target)        # scalar
          logits = [-loss, loss]                 # fake 2-class logits

    This fools ``torchattacks.PGD`` into maximising the MSE loss, which
    drives the adversarial embedding away from the true identity.

2.  Passes a dummy ``labels`` tensor of zeros to satisfy ``torchattacks``'
    API (it calls ``criterion(logits, labels)`` internally).

PGD perturbation rule (L∞ projection)::

    x_adv = x
    for t in range(steps):
        grad = ∇_x CrossEntropy(logits(x_adv), 0)
        x_adv = x_adv + alpha * sign(grad)
        x_adv = clamp(x_adv, x - eps, x + eps)  ← L∞ projection
        x_adv = clamp(x_adv, 0.0, 1.0)           ← valid pixel range

Public interface is **identical** to the previous ``FGSMAttack`` class so all
downstream consumers require no changes.

References:
    Madry et al. (2018) "Towards Deep Learning Models Resistant to Adversarial
    Attacks"  https://arxiv.org/abs/1706.06083
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from facenet_pytorch import InceptionResnetV1

# ---------------------------------------------------------------------------
# Reproducibility seed
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal model wrapper — not part of the public API
# ---------------------------------------------------------------------------

class _EmbeddingLossWrapper(nn.Module):
    """Thin nn.Module adapter that makes FaceNet compatible with torchattacks.

    ``torchattacks.PGD`` expects a model that maps ``(inputs, labels)`` to
    classification logits and then maximises cross-entropy loss.  Since FaceNet
    is an embedding model we fabricate 2-class logits from the MSE distance
    between the current adversarial embedding and a fixed random target so
    that maximising the cross-entropy loss is equivalent to driving the
    embedding away from the true identity.

    Attributes:
        facenet (InceptionResnetV1): Frozen pretrained FaceNet model.
        target (torch.Tensor): Fixed random unit-sphere embedding target of
            shape ``(1, 512)``.
    """

    def __init__(self, facenet: InceptionResnetV1, target: torch.Tensor) -> None:
        super().__init__()
        self.facenet = facenet
        self.register_buffer("target", target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fake 2-class logits from FaceNet embedding MSE loss.

        Args:
            x: Float tensor of shape ``(B, 3, H, W)`` with values in
                ``[0.0, 1.0]``.

        Returns:
            Logits tensor of shape ``(B, 2)``.
        """
        # Normalise [0,1] → [-1,1] as FaceNet expects.
        x_norm = (x - 0.5) / 0.5
        embedding = self.facenet(x_norm)  # (B, 512)

        # Expand target to match batch dimension.
        target_expanded = self.target.expand(embedding.size(0), -1)

        # Scalar MSE per sample — shape (B,)
        mse = F.mse_loss(embedding, target_expanded, reduction="none").mean(dim=1)

        # Construct fake 2-class logits: class 0 = -mse (low), class 1 = +mse (high).
        # torchattacks.PGD calls CrossEntropy(logits, labels=0) and maximises it,
        # which is equivalent to maximising mse — i.e. pushing the embedding away
        # from the true identity toward the random target.
        logits = torch.stack([-mse, mse], dim=1)  # (B, 2)
        return logits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PGDAttack:
    """Projected Gradient Descent attack for FaceNet embedding-space disruption.

    Loads a pretrained ``InceptionResnetV1`` model (VGGFace2) and uses it
    strictly as a **feature extractor** to generate 512-dimensional identity
    embeddings.  The multi-step PGD perturbation iteratively drives the
    adversarial embedding toward a random unit-sphere target, maximising the
    probability that automated face-recognition systems misidentify the person.

    Attributes:
        device (torch.device): Computation device (CPU or CUDA).
        eps (float): Maximum L-infinity perturbation magnitude in [0, 1].
        alpha (float): Step size per PGD iteration.
        steps (int): Number of PGD iterations.
        model (InceptionResnetV1): Frozen FaceNet embedding model.
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 10,
        device: str = "auto",
    ) -> None:
        """Initialise the PGD attack with a pretrained FaceNet model.

        Args:
            eps: L-infinity perturbation budget (max pixel change).
                Defaults to ``8/255`` (~0.0314), the standard PGD-8 budget.
            alpha: Step size per PGD iteration.  Defaults to ``2/255``.
            steps: Number of PGD gradient steps.  Defaults to ``10``.
            device: Computation device string.  Pass ``'cpu'`` or ``'cuda'``
                to force a specific device, or ``'auto'`` (default) to
                automatically select CUDA when available.

        Raises:
            RuntimeError: If the pretrained FaceNet weights cannot be loaded.
        """
        self.eps = eps
        # Store epsilon as an alias so callers that pass epsilon= still work.
        self.epsilon = eps
        self.alpha = alpha
        self.steps = steps

        # ── Device selection ─────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(
            "PGDAttack initialising on device: '%s' | eps: %.5f | "
            "alpha: %.5f | steps: %d",
            self.device,
            self.eps,
            self.alpha,
            self.steps,
        )

        # ── Load FaceNet model ───────────────────────────────────────────────
        self.model = self._load_facenet()

        # ── Sample and freeze the random embedding target ────────────────────
        # A fixed seed produces a reproducible target direction across runs.
        _rng = torch.get_rng_state()
        torch.manual_seed(42)
        target = F.normalize(
            torch.randn(1, 512, device=self.device), p=2, dim=1
        ).detach()
        torch.set_rng_state(_rng)
        self._target = target

        # ── Build wrapper and PGD attacker ───────────────────────────────────
        self._wrapper = _EmbeddingLossWrapper(self.model, self._target).to(self.device)
        self._wrapper.eval()

        self._pgd = torchattacks.PGD(
            self._wrapper,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
        )

        logger.info("PGDAttack ready.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_facenet(self) -> InceptionResnetV1:
        """Load the pretrained InceptionResnetV1 model in evaluation mode.

        Returns:
            A frozen ``InceptionResnetV1`` instance in ``eval()`` mode.

        Raises:
            RuntimeError: If weights cannot be loaded from cache or network.
        """
        logger.info("Loading InceptionResnetV1 (pretrained='vggface2') …")
        try:
            model = InceptionResnetV1(pretrained="vggface2").to(self.device)
        except Exception as exc:
            logger.error("Failed to load FaceNet weights: %s", exc)
            raise RuntimeError(
                "Could not load InceptionResnetV1 pretrained weights. "
                "Possible causes:\n"
                "  1. No internet connection and weights not cached locally.\n"
                "  2. Corrupt local cache — delete ~/.cache/torch/checkpoints/ "
                "     and retry.\n"
                "  3. Incompatible facenet-pytorch version.\n"
                f"Original error: {exc}"
            ) from exc

        # Freeze all parameters — used as a fixed feature extractor only.
        for param in model.parameters():
            param.requires_grad_(False)

        model.eval()
        logger.info("InceptionResnetV1 loaded and frozen (eval mode, no grad).")
        return model

    @staticmethod
    def _preprocess(face_tensor: torch.Tensor) -> torch.Tensor:
        """Normalise a [0, 1] face tensor to the [-1, 1] range.

        Args:
            face_tensor: Float tensor with values in ``[0.0, 1.0]``.

        Returns:
            Normalised float tensor with values in ``[-1.0, 1.0]``.
        """
        return (face_tensor - 0.5) / 0.5  # maps [0,1] → [-1,1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Compute the L2-normalised 512-d identity embedding for a face.

        Args:
            face_tensor: Float tensor of shape ``(1, 3, H, W)`` with pixel
                values in ``[0.0, 1.0]``.

        Returns:
            A detached float tensor of shape ``(1, 512)`` representing the
            unit-normalised identity embedding vector.

        Raises:
            ValueError: If ``face_tensor`` is not a 4-D tensor.
        """
        if face_tensor.ndim != 4:
            raise ValueError(
                f"face_tensor must be 4-D (batch, channels, H, W), "
                f"got shape {tuple(face_tensor.shape)}."
            )

        tensor = face_tensor.to(self.device)
        tensor_normalised = self._preprocess(tensor)

        with torch.no_grad():
            embedding = self.model(tensor_normalised)

        logger.info(
            "Embedding computed. Shape: %s | L2-norm: %.4f",
            tuple(embedding.shape),
            float(torch.norm(embedding, p=2, dim=1).mean()),
        )
        return embedding.detach()

    def attack(
        self,
        face_tensor: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate a PGD adversarial perturbation in embedding space.

        Performs a multi-step Projected Gradient Descent attack using a
        **random target embedding** loss.  The loss drives the adversarial
        face embedding toward a random unit-sphere vector over ``steps``
        iterations with L∞ projection after each step.

        Args:
            face_tensor: Float tensor of shape ``(1, 3, H, W)`` with pixel
                values in ``[0.0, 1.0]``.
            epsilon: Optional override for the perturbation budget.  If
                ``None``, uses the value supplied at construction time.
                Note: this updates the underlying PGD attacker in-place.

        Returns:
            Adversarial float tensor of shape ``(1, 3, H, W)`` with pixel
            values clamped to ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``face_tensor`` is not a 4-D tensor.
        """
        if face_tensor.ndim != 4:
            raise ValueError(
                f"face_tensor must be 4-D (batch, channels, H, W), "
                f"got shape {tuple(face_tensor.shape)}."
            )

        eps = epsilon if epsilon is not None else self.eps

        # Update eps on the live attacker if an override was supplied.
        if epsilon is not None and epsilon != self._pgd.eps:
            self._pgd.eps = eps

        logger.info(
            "Running PGD attack | eps: %.5f | alpha: %.5f | steps: %d",
            eps, self.alpha, self.steps,
        )

        x = face_tensor.to(self.device)

        # torchattacks expects a labels tensor — we pass zeros (dummy class 0).
        labels = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        # Run multi-step PGD; returns adversarial tensor in [0, 1].
        adversarial = self._pgd(x, labels)

        # Log embedding distance achieved.
        with torch.no_grad():
            emb_orig = self.get_embedding(face_tensor)
            emb_adv  = self.get_embedding(adversarial)
            cosine_sim = float(
                F.cosine_similarity(emb_orig, emb_adv, dim=1).mean()
            )

        logger.info(
            "PGD attack complete. "
            "Shape: %s | Range: [%.4f, %.4f] | "
            "Cosine sim (orig→adv): %.4f",
            tuple(adversarial.shape),
            float(adversarial.min()),
            float(adversarial.max()),
            cosine_sim,
        )

        return adversarial.detach()
