# Adapted-source provenance:
# Paper: Lippe et al., CITRIS (ICML 2022); Lippe et al., iCITRIS (ICLR 2023).
# Official implementation: vendored under gcmts/vendor/citris (MIT licence).
# Status: official image model wrapped for the shared benchmark protocol.

"""Official image CITRIS/iCITRIS adapters.

The vector implementations in :mod:`gcmts.causal_representation_learning.methods`
are paper-based adaptations and cannot consume rendered videos.  The adapters
here instantiate the authors' ``CITRISVAE`` and ``iCITRISVAE`` models from the
vendored release, so the visual benchmark uses the original convolutional
encoder/decoder and target classifier/prior rather than a surrogate.

Only glue code lives here: the official module is trained through its own
``_get_loss``; this wrapper maps the benchmark ``Batch`` (flattened frames plus
intervention targets) to the tensor layout expected by the release.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "citris"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from models.citris_vae import CITRISVAE  # noqa: E402
from models.icitris_vae import iCITRISVAE  # noqa: E402

__all__ = ["CITRISImage", "iCITRISImage"]


class _OfficialImageAdapter(BaseCausalRepresentationLearner):
    """Shared batch/latent glue for the official VAE modules."""

    official: torch.nn.Module

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        n_vars: int,
        image_shape: tuple[int, int, int],
        c_hid: int = 32,
        lr: float = 1e-3,
        **official_kwargs: object,
    ) -> None:
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=1,
            image_shape=image_shape,
        )
        self.n_vars = int(n_vars)
        self.c_hid = int(c_hid)
        self.trainer_lr = float(lr)
        self._official_kwargs = dict(official_kwargs)
        self.official = self._build_official()
        # Lightning's ``self.log`` expects an attached trainer; the shared
        # trainer has no Lightning logger, so logging is deliberately a no-op.
        self.official.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    def _build_official(self) -> torch.nn.Module:  # pragma: no cover - overridden
        raise NotImplementedError

    def _images(self, x: Tensor) -> Tensor:
        if self.image_shape is None:
            raise ValueError("The official image adapter requires image_shape.")
        batch, time, dim = x.shape
        channels, height, width = self.image_shape
        expected = channels * height * width
        if dim != expected:
            raise ValueError(f"Expected {expected} values per frame, got {dim}.")
        return x.reshape(batch, time, channels, height, width)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        del context
        images = self._images(x)
        batch, time = images.shape[:2]
        with torch.no_grad():
            z = self.official.encode(images.flatten(0, 1), random=False)
        return z.reshape(batch, time, -1)

    def decode(self, z: Tensor) -> Tensor:
        output = self.official.decoder(z.reshape(-1, z.shape[-1]))
        batch, time = z.shape[:2]
        return output.reshape(batch, time, -1)

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        del context
        return z_past[:, -1]

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x = batch.x
        images = self._images(x)
        batch_size, time = images.shape[:2]
        z_mean, z_logstd = self.official.encoder(images.flatten(0, 1))
        z = (z_mean + torch.randn_like(z_mean) * z_logstd.exp()).reshape(
            batch_size, time, -1
        )
        x_recon = self.decode(z)
        return CausalRepresentationOutput(
            z=z,
            x_recon=x_recon,
            log_prob=torch.zeros(batch_size, device=x.device, dtype=x.dtype),
            extras={"z_mean": z_mean, "z_logstd": z_logstd},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if batch.context is None or "I" not in batch.context:
            raise ValueError("Official CITRIS needs intervention targets in context['I'].")
        images = self._images(batch.x)
        targets = batch.context["I"].to(images.device)
        if targets.ndim != 3 or targets.shape[-1] != self.n_vars:
            raise ValueError(
                f"Expected targets (B, T, {self.n_vars}), got {tuple(targets.shape)}."
            )
        # The official loss compares z_{t+1} with the intervention at t+1.
        official_loss = self.official._get_loss((images, targets[:, 1:]), mode="train")
        recon = torch.nn.functional.mse_loss(outputs.x_recon, batch.x)
        return {"loss": official_loss, "recon": recon}

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "Official CITRIS/iCITRIS image model: recovers minimal causal "
            "variables up to permutation and element-wise invertible maps "
            "given known intervention targets; iCITRIS additionally identifies "
            "instantaneous structure."
        )

    def ambiguity_class(self) -> AmbiguityClass:
        return AmbiguityClass.PERM_COMPONENTWISE


class CITRISImage(_OfficialImageAdapter):  # noqa: N801
    """CITRIS with the authors' convolutional VAE (Lippe et al., 2022)."""

    def _build_official(self) -> torch.nn.Module:
        channels = self.image_shape[0] if self.image_shape is not None else 3
        width = self.image_shape[1] if self.image_shape is not None else 64
        kwargs = {
            "c_hid": self.c_hid,
            "num_latents": self.latent_dim,
            "lr": self.trainer_lr,
            "num_causal_vars": self.n_vars,
            "warmup": 100,
            "max_iters": 100000,
            "img_width": width,
            "c_in": channels,
            "beta_t1": 1.0,
            "beta_classifier": 1.0,
            "autoregressive_prior": False,
            "use_flow_prior": True,
            "lambda_reg": 0.01,
            "imperfect_interventions": False,
        }
        kwargs.update(self._official_kwargs)
        return CITRISVAE(**kwargs)


class iCITRISImage(_OfficialImageAdapter):  # noqa: N801
    """iCITRIS with the authors' instantaneous-effect VAE (Lippe et al., 2023)."""

    def _build_official(self) -> torch.nn.Module:
        channels = self.image_shape[0] if self.image_shape is not None else 3
        width = self.image_shape[1] if self.image_shape is not None else 64
        kwargs = {
            "c_hid": self.c_hid,
            "num_latents": self.latent_dim,
            "lr": self.trainer_lr,
            "num_causal_vars": self.n_vars,
            "warmup": 100,
            "max_iters": 100000,
            "img_width": width,
            "c_in": channels,
            "beta_t1": 1.0,
            "beta_classifier": 1.0,
            "beta_mi_estimator": 1.0,
            "graph_learning_method": "ENCO",
            "graph_lr": 5e-4,
            "autoregressive_prior": False,
            "use_flow_prior": True,
        }
        kwargs.update(self._official_kwargs)
        return iCITRISVAE(**kwargs)
