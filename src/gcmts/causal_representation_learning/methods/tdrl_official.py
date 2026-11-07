"""TDRL, adapted from the official implementation.

Paper: Yao et al., *Temporally Disentangled Representation Learning* (2022),
BibTeX ``yaoTemporallyDisentangledRepresentation2022``.
Official source: https://github.com/weirayao/tdrl (MIT license).
The official modular temporal-VAE (ModularShifts) and nonparametric change prior are used;
this file only adapts the batch/interface and registers the model in ``gcmts``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput
from gcmts.vendor.tdrl.modules.modular import ModularShifts

logger = logging.getLogger(__name__)

__all__ = ["TDRLOfficial"]


def _class_index(context: dict[str, Tensor] | None, batch_size: int, n_classes: int, device: torch.device) -> Tensor:
    if context is None or "u" not in context or n_classes <= 1:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    u = context["u"]
    if u.ndim == 3:
        u = u[:, -1]
    if u.ndim == 2 and u.shape[1] not in (1, n_classes):
        # Physics/control suites store one regime index per timestep; the
        # official models need the regime that labels the sequence window.
        u = u[:, -1]
    if u.ndim == 1:
        return u.long().clamp(0, n_classes - 1).to(device)
    if u.shape[-1] == 1:
        return u[:, 0].long().clamp(0, n_classes - 1).to(device)
    return u.argmax(dim=-1).long().clamp(0, n_classes - 1).to(device)


class TDRLOfficial(BaseCausalRepresentationLearner):
    """Thin interface around the official TDRL factorized model."""

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_fix_dim: int,
        latent_dyn_dim: int,
        latent_obs_dim: int,
        max_lag: int = 1,
        u_dim: int = 0,
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        beta: float = 0.0025,
        gamma: float = 0.0075,
        lr: float = 1e-3,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 0.01,
        obs_noise: float = 1.0,
        obs_weight: float = 1.0,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        del kl_weight, decoder_hidden, obs_noise, obs_weight  # official ELBO uses beta/gamma
        latent_dim = int(latent_fix_dim + latent_dyn_dim + latent_obs_dim)
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=max_lag,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        self.n_classes = max(int(u_dim), 1)
        self.trainer_lr = float(lr)
        self.core = ModularShifts(
            input_dim=self.encoder_input_dim(observed_dim),
            length=1,
            obs_dim=int(latent_obs_dim),
            dyn_dim=int(latent_fix_dim + latent_dyn_dim),
            lag=max_lag,
            nclass=self.n_classes,
            hidden_dim=int(hidden_dim),
            dyn_embedding_dim=int(embedding_dim),
            obs_embedding_dim=2,
            trans_prior="NP",
            lr=float(lr),
            infer_mode="F",
            beta=float(beta),
            gamma=float(gamma),
            sigma=0.01,
            decoder_dist="gaussian",
        )
        # The official LightningModule logs through self.log(); we drive it
        # directly, so the logger is intentionally a no-op.
        self.core.log = lambda *args, **kwargs: None

    def _obs(self, x: Tensor) -> Tensor:
        if self.image_shape is None:
            return x
        features, (batch, time) = self.observation_features(x)
        return features.reshape(batch, time, -1)

    def _ct(self, batch: Batch) -> Tensor:
        return _class_index(batch.context, batch.x.shape[0], self.n_classes, batch.x.device)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        obs = self._obs(x)
        flat = obs.reshape(-1, obs.shape[-1])
        encoded = self.core.net._encode(flat)
        mean = encoded[:, : self.latent_dim]
        return mean.reshape(obs.shape[0], obs.shape[1], self.latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])
        recon = self.core.net._decode(flat)
        return recon.reshape(z.shape[0], z.shape[1], -1) if self.image_shape is not None else recon.reshape(z.shape[0], z.shape[1], self.observed_dim)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        obs = self._obs(batch.x)
        self.core.length = int(obs.shape[1])
        flat = obs.reshape(-1, obs.shape[-1])
        x_recon, mean, logvar, z = self.core.net(flat)
        z = z.reshape(obs.shape[0], obs.shape[1], self.latent_dim)
        return CausalRepresentationOutput(
            z=z,
            x_recon=x_recon.reshape_as(obs),
            extras={
                "mean": mean.reshape(obs.shape[0], obs.shape[1], self.latent_dim),
                "logvar": logvar.reshape(obs.shape[0], obs.shape[1], self.latent_dim),
            },
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        obs = self._obs(batch.x)
        self.core.length = int(obs.shape[1])
        ct = self._ct(batch)
        loss = self.core.training_step(
            {"xt": obs, "yt": obs, "ct": ct},
            0,
        )
        return {"loss": loss, "recon": torch.zeros((), device=loss.device)}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        # The official code exposes a residual prior, not a one-step transition
        # API.  Persistence is used only for the auxiliary forecast diagnostic;
        # representation metrics do not use this path.
        return z_past[:, -1]

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "TDRL: under modular distribution shifts, an invertible mixing map "
            "and sufficient change in the transition or observation mechanisms, "
            "the latents are identifiable up to permutation and component-wise "
            "invertible maps; the official factorized implementation is used."
        )

    def ambiguity_class(self) -> AmbiguityClass:
        return AmbiguityClass.PERM_COMPONENTWISE
