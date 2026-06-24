"""
fgsm_attack.py
--------------
Fast Gradient Sign Method (FGSM) adversarial attack implemented in
FaceNet's **embedding space**.

FaceNet (``InceptionResnetV1``) is an embedding model, not a classifier.
Standard classification-based FGSM (which maximises cross-entropy loss w.r.t.
a class label) is therefore not applicable.  Instead this module defines a
differentiable loss that **drives the adversarial embedding toward a random
unit-sphere target**, maximising the probability that automated face-recognition
systems misidentify the person:

    target  = normalize(randn(512))   # random unit vector in R^512
    loss    = MSE(emb_adv, target)    # push embedding away from true identity

This formulation is preferred over the more intuitive ``-cosine_similarity``
loss because at initialisation (when ``emb_adv == emb_orig``), the cosine
similarity gradient is mathematically zero:

    ∂cos(a, a)/∂a = a/|a|² − a(a·a)/|a|³ = a − a = 0

The MSE gradient ``∂MSE(b, t)/∂b = 2(b − t)`` is non-zero whenever ``b ≠ t``,
which is guaranteed for a random target at initialisation.

FGSM perturbation rule::

    X_adv = X + epsilon * sign(∇_X MSE(f(X), target))
    X_adv = clamp(X_adv, 0.0, 1.0)

References:
    Goodfellow et al. (2015) "Explaining and Harnessing Adversarial Examples"
    https://arxiv.org/abs/1412.6572
"""


import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

# ---------------------------------------------------------------------------
# Reproducibility seed
# Note: MTCNN and InceptionResnetV1 forward passes involve deterministic ops
# on CPU.  GPU execution may still be non-deterministic due to CUDA atomics.
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

logger = logging.getLogger(__name__)


class FGSMAttack:
    """Fast Gradient Sign Method attack for FaceNet embedding-space disruption.

    Loads a pretrained ``InceptionResnetV1`` model (VGGFace2) and uses it
    strictly as a **feature extractor** to generate 512-dimensional
    identity embeddings.  The FGSM perturbation is computed by maximising
    the cosine distance between the original and adversarial embeddings.

    Attributes:
        device (torch.device): Computation device (CPU or CUDA).
        epsilon (float): Maximum L-infinity perturbation magnitude in [0, 1].
        model (InceptionResnetV1): Frozen FaceNet embedding model.
    """

    def __init__(self, epsilon: float = 0.02, device: str = "auto") -> None:
        """Initialise the FGSM attack with a pretrained FaceNet model.

        Args:
            epsilon: L-infinity perturbation budget.  Controls the maximum
                per-pixel change allowed.  Defaults to ``0.02`` (2 % of the
                normalised [0, 1] range), which typically preserves human
                visual quality while confusing face-recognition systems.
            device: Computation device string.  Pass ``'cpu'`` or ``'cuda'``
                to force a specific device, or ``'auto'`` (default) to
                automatically select CUDA when available and fall back to CPU.

        Raises:
            RuntimeError: If the pretrained FaceNet weights cannot be loaded
                (e.g. no internet connection and no cached weights found).
        """
        self.epsilon = epsilon

        # ── Device selection ─────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(
            "FGSMAttack initialising on device: '%s' | epsilon: %.4f",
            self.device,
            self.epsilon,
        )

        # ── Load FaceNet model ───────────────────────────────────────────────
        self.model = self._load_facenet()
        logger.info("FGSMAttack ready.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_facenet(self) -> InceptionResnetV1:
        """Load the pretrained InceptionResnetV1 model in evaluation mode.

        Attempts to load the VGGFace2-pretrained weights.  If the weights
        are not cached locally, ``facenet-pytorch`` will attempt to download
        them automatically.  A clear error is raised if this fails.

        Returns:
            A frozen ``InceptionResnetV1`` instance in ``eval()`` mode.

        Raises:
            RuntimeError: If weights cannot be loaded from cache or network.
        """
        logger.info(
            "Loading InceptionResnetV1 (pretrained='vggface2') …"
        )
        logger.info(
            "  If weights are not cached this will trigger a one-time download. "
            "  Ensure internet connectivity or pre-download the weights."
        )

        try:
            model = InceptionResnetV1(pretrained="vggface2").to(self.device)
        except Exception as exc:
            logger.error(
                "Failed to load FaceNet weights: %s", exc
            )
            raise RuntimeError(
                "Could not load InceptionResnetV1 pretrained weights. "
                "Possible causes:\n"
                "  1. No internet connection and weights not cached locally.\n"
                "  2. Corrupt local cache — delete ~/.cache/torch/checkpoints/ "
                "     and retry.\n"
                "  3. Incompatible facenet-pytorch version.\n"
                f"Original error: {exc}"
            ) from exc

        # Freeze all parameters — the model is used as a fixed feature extractor
        # only.  No fine-tuning occurs during adversarial perturbation.
        for param in model.parameters():
            param.requires_grad_(False)

        model.eval()
        logger.info(
            "InceptionResnetV1 loaded and frozen (eval mode, no grad)."
        )
        return model

    @staticmethod
    def _preprocess(face_tensor: torch.Tensor) -> torch.Tensor:
        """Normalise a [0, 1] face tensor to the [-1, 1] range.

        ``InceptionResnetV1`` was trained with pixel values in [-1, 1]
        (mean = 0.5, std = 0.5 per channel).  This normalisation must be
        applied before every forward pass.

        Args:
            face_tensor: Float tensor of shape ``(1, 3, H, W)`` with values
                in ``[0.0, 1.0]``.

        Returns:
            Normalised float tensor of shape ``(1, 3, H, W)`` with values
            in ``[-1.0, 1.0]``.
        """
        return (face_tensor - 0.5) / 0.5  # maps [0,1] → [-1,1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Compute the L2-normalised 512-d identity embedding for a face.

        Args:
            face_tensor: Float tensor of shape ``(1, 3, H, W)`` with pixel
                values in ``[0.0, 1.0]``.  The spatial dimensions should be
                160 × 160 pixels (the native FaceNet input size); other sizes
                are accepted but may reduce accuracy.

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
        """Generate an FGSM adversarial perturbation in embedding space.

        Performs a single-step Fast Gradient Sign Method attack using a
        **random target embedding** loss.  The loss drives the adversarial
        face embedding toward a random unit-sphere vector:

            target = normalize(randn(512))      # random unit vector
            loss   = MSE(emb_adv, target)       # push toward alien identity

        This formulation avoids the mathematically degenerate zero-gradient
        case of cosine-similarity loss.  When ``emb_adv == emb_orig`` (the
        initial condition before any perturbation), the gradient of cosine
        similarity w.r.t. the embedding is exactly zero::

            ∂cos(a, a)/∂a = a/|a|² − a(a·a)/|a|³ = a − a = 0

        This produces an all-zero sign matrix and no perturbation whatsoever.
        MSE toward a random target has the gradient::

            ∂MSE(b, t)/∂b = 2(b − t)

        which is zero only if ``b == t`` — vanishingly unlikely for a random
        target in R^512.

        Pipeline steps:
            1. Compute the original identity embedding ``emb_orig``.
            2. Sample a reproducible random unit-sphere target ``target``.
            3. Enable gradient tracking on the input tensor.
            4. Forward-pass the adversarial input → ``emb_adv``.
            5. Compute: ``loss = MSE(emb_adv, target)``.
            6. Back-propagate to get ``∇_X loss``.
            7. Apply: ``X_adv = X + epsilon * sign(∇_X loss)``.
            8. Clamp: ``X_adv = clamp(X_adv, 0.0, 1.0)``.

        Args:
            face_tensor: Float tensor of shape ``(1, 3, H, W)`` with pixel
                values in ``[0.0, 1.0]``.  Represents the face region to
                perturb.
            epsilon: Optional override for the perturbation budget.  If
                ``None``, uses the value supplied at construction time.

        Returns:
            Adversarial float tensor of shape ``(1, 3, H, W)`` with pixel
            values clamped to ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``face_tensor`` is not a 4-D tensor.
            RuntimeError: If the backward pass yields an all-zero gradient,
                indicating a broken computation graph.
        """
        if face_tensor.ndim != 4:
            raise ValueError(
                f"face_tensor must be 4-D (batch, channels, H, W), "
                f"got shape {tuple(face_tensor.shape)}."
            )

        eps = epsilon if epsilon is not None else self.epsilon
        logger.info("Running FGSM attack | epsilon: %.4f", eps)

        # ── Step 1: Compute original embedding (no grad) ─────────────────────
        emb_original = self.get_embedding(face_tensor)

        # ── Step 2: Sample random unit-sphere target embedding ───────────────
        # A fixed seed produces a reproducible target direction across runs.
        # The target is orthogonal to the original embedding on average, so
        # optimising toward it maximises the probability of identity confusion.
        _rng_state = torch.get_rng_state()          # save current RNG state
        torch.manual_seed(42)
        target = F.normalize(
            torch.randn(emb_original.shape, device=self.device), p=2, dim=1
        ).detach()
        torch.set_rng_state(_rng_state)             # restore RNG state

        logger.info(
            "Random target sampled. "
            "Cosine sim (original → target): %.4f",
            float(F.cosine_similarity(emb_original, target, dim=1).mean()),
        )

        # ── Step 3: Prepare adversarial input with gradient tracking ─────────
        x_adv = face_tensor.detach().clone().to(self.device)
        x_adv.requires_grad_(True)

        # ── Steps 4 & 5: Forward pass + loss ─────────────────────────────────
        x_adv_normalised = self._preprocess(x_adv)

        with torch.enable_grad():
            emb_adv = self.model(x_adv_normalised)

            # MSE toward random target — non-zero gradient guaranteed at init.
            loss = F.mse_loss(emb_adv, target)
            logger.info(
                "FGSM loss (MSE→random target): %.6f  "
                "| Cosine sim (orig→adv): %.4f",
                loss.item(),
                F.cosine_similarity(emb_original, emb_adv, dim=1).mean().detach().item(),
            )

            # ── Step 6: Back-propagation ──────────────────────────────────────
            loss.backward()

        if x_adv.grad is None:
            raise RuntimeError(
                "Gradient is None after backward(). "
                "The computation graph from x_adv through the model "
                "may be broken."
            )

        # Warn if gradient is all-zero (should never happen with MSE loss).
        grad_abs_max = float(x_adv.grad.abs().max())
        if grad_abs_max < 1e-12:
            logger.warning(
                "All-zero gradient detected — perturbation will have no "
                "effect.  This may indicate a disconnected computation graph."
            )

        grad_sign = x_adv.grad.sign()
        logger.info(
            "Gradient sign computed — non-zero elements: %d / %d (%.1f%%)",
            int((grad_sign != 0).sum()),
            grad_sign.numel(),
            100.0 * float((grad_sign != 0).sum()) / grad_sign.numel(),
        )

        # ── Steps 7 & 8: Apply perturbation and clamp ────────────────────────
        with torch.no_grad():
            adversarial = x_adv + eps * grad_sign
            adversarial = torch.clamp(adversarial, 0.0, 1.0)

        logger.info(
            "Adversarial tensor generated. "
            "Shape: %s | Value range: [%.4f, %.4f]",
            tuple(adversarial.shape),
            float(adversarial.min()),
            float(adversarial.max()),
        )
        return adversarial.detach()
