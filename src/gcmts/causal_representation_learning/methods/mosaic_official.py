"""MOSAIC, adapted from the official implementation.

Paper: Fan et al., *MOSAIC: Module Discovery via Sparse Additive Identifiable
Causal Learning for Scientific Time Series* (2026), BibTeX
``shichengfanMOSAICModuleDiscovery2026``.
Official source: https://github.com/shichengf/mosaic (MIT license).
The official additive decoder, regime-conditioned temporal prior, sparsity
schedule and two-stage freeze schedule are used; this file adapts the batch and
epoch interfaces to ``gcmts``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput
from gcmts.vendor.mosaic.mosaic.models.mosaic import TimeVaryingProcess_v2

logger = logging.getLogger(__name__)

__all__ = ["MOSAICOfficial"]


def _ct(context: dict[str, Tensor] | None, batch_size: int, n_classes: int, device: torch.device) -> Tensor:
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


class MOSAICOfficial(BaseCausalRepresentationLearner):
    """Interface around the official MOSAIC additive temporal VAE."""

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        u_dim: int = 0,
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        hidden_per_z: int = 8,
        beta: float = 0.0025,
        gamma: float = 0.0075,
        lambda_sparse: float = 1e-3,
        warmup_epochs: int = 2,
        rampup_epochs: int = 3,
        freeze_epochs: int = 3,
        lr: float = 1e-3,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (64,),
        sparsity: float = 0.01,
        kl_weight: float = 1.0,
        obs_noise: float = 1.0,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        del hidden_dims, decoder_hidden, sparsity, kl_weight, obs_noise
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=1,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        self.n_classes = max(int(u_dim), 1)
        self.trainer_lr = float(lr)
        self.core = TimeVaryingProcess_v2(
            input_dim=self.encoder_input_dim(observed_dim),
            length=1,
            z_dim=latent_dim,
            lag=1,
            nclass=self.n_classes,
            hidden_dim=int(hidden_dim),
            embedding_dim=int(embedding_dim),
            hidden_per_z=int(hidden_per_z),
            decoder_type="additive",
            trans_prior="NP",
            lr=float(lr),
            infer_mode="F",
            beta=float(beta),
            gamma=float(gamma),
            lambda_sparse=float(lambda_sparse),
            warmup_epochs=int(warmup_epochs),
            rampup_epochs=int(rampup_epochs),
            freeze_epochs=int(freeze_epochs),
        )
        self.core.log = lambda *args, **kwargs: None

    def _obs(self, x: Tensor) -> Tensor:
        if self.image_shape is None:
            return x
        features, (batch, time) = self.observation_features(x)
        return features.reshape(batch, time, -1)

    def _batch(self, batch: Batch) -> dict[str, Tensor]:
        obs = self._obs(batch.x)
        self.core.length = int(obs.shape[1])
        ct = _ct(batch.context, obs.shape[0], self.n_classes, obs.device)
        return {"xt": obs, "yt": obs, "ct": ct}

    def on_epoch_start(self, epoch: int) -> None:
        self.core.manual_epoch = int(epoch)
        self.core.on_train_epoch_start()

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        obs = self._obs(x)
        flat = obs.reshape(-1, obs.shape[-1])
        encoded = self.core.net._encode(flat)
        return encoded[:, : self.latent_dim].reshape(obs.shape[0], obs.shape[1], self.latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])
        rec = self.core.net._decode(flat)
        return rec.reshape(z.shape[0], z.shape[1], self.observed_dim)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        obs = self._obs(batch.x)
        self.core.length = int(obs.shape[1])
        flat = obs.reshape(-1, obs.shape[-1])
        x_recon, mean, logvar, z = self.core.net(flat)
        return CausalRepresentationOutput(
            z=z.reshape(obs.shape[0], obs.shape[1], self.latent_dim),
            x_recon=x_recon.reshape_as(obs),
            extras={
                "mean": mean.reshape(obs.shape[0], obs.shape[1], self.latent_dim),
                "logvar": logvar.reshape(obs.shape[0], obs.shape[1], self.latent_dim),
            },
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        loss = self.core.training_step(self._batch(batch), 0)
        return {"loss": loss, "recon": torch.zeros((), device=loss.device)}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        return z_past[:, -1]

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "MOSAIC: a regime-conditioned temporal prior identifies the latent "
            "factors up to permutation and component-wise transforms; the "
            "additive decoder then exposes the observation support of each "
            "latent. The official two-stage/additive model is used."
        )

    def ambiguity_class(self) -> AmbiguityClass:
        return AmbiguityClass.PERM_COMPONENTWISE
