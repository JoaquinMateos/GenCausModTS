"""iVAE, adapted from the official implementation.

Paper: Khemakhem et al., *Variational Autoencoders and Nonlinear ICA: A
Unifying Framework* (AISTATS 2020), BibTeX
``khemakhemVariationalAutoencodersNonlinear2020``.
Official source: https://github.com/ilkhem/iVAE.
The official exponential-family prior, encoder and decoder are used; temporal
observations are flattened per timestep and the auxiliary variable is repeated
across the sequence.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput
from gcmts.vendor.ivaE.nets import iVAE as IVAECore

logger = logging.getLogger(__name__)

__all__ = ["IVAEOfficial"]


def _aux(context: dict[str, Tensor] | None, batch: int, time: int, dim: int, device: torch.device) -> Tensor:
    if context is None or "u" not in context or dim <= 0:
        return torch.zeros(batch * time, dim, device=device)
    u = context["u"]
    if u.ndim == 2:
        u = u.unsqueeze(1).expand(-1, time, -1)
    return u.reshape(batch * time, -1).to(device)


class IVAEOfficial(BaseCausalRepresentationLearner):
    """Interface around the official iVAE model."""

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        u_dim: int = 0,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 1.0,
        obs_noise: float = 1.0,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        del hidden_dims, decoder_hidden, kl_weight, obs_noise
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=1,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        self.n_aux = max(int(u_dim), 1)
        self.x_scale = 1.0
        self.core = IVAECore(
            latent_dim=latent_dim,
            data_dim=self.encoder_input_dim(observed_dim),
            aux_dim=self.n_aux,
            n_layers=3,
            hidden_dim=100,
            activation="xtanh",
            device="cpu",
        )

    def to(self, *args, **kwargs):
        """Move the official iVAE plain-tensor distribution state with the module."""
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.core.prior_mean = self.core.prior_mean.to(device)
        self.core.decoder_var = self.core.decoder_var.to(device)
        for dist in (self.core.prior_dist, self.core.decoder_dist, self.core.encoder_dist):
            dist.device = device
            dist.c = dist.c.to(device)
            dist._dist = type(dist._dist)(
                torch.zeros(1, device=device), torch.ones(1, device=device)
            )
        return result

    def _obs(self, x: Tensor) -> Tensor:
        if self.image_shape is None:
            return x
        features, (batch, time) = self.observation_features(x)
        return features.reshape(batch, time, -1)

    def _flat(self, x: Tensor, context: dict[str, Tensor] | None) -> tuple[Tensor, Tensor, int, int]:
        obs = self._obs(x)
        batch, time, dim = obs.shape
        flat = obs.reshape(batch * time, dim) * self.x_scale
        return flat, _aux(context, batch, time, self.n_aux, obs.device), batch, time

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        flat, u, batch, time = self._flat(x, context)
        mean, _ = self.core.encoder_params(flat, u)
        return mean.reshape(batch, time, self.latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])
        rec = self.core.f(flat)
        return rec.reshape(z.shape[0], z.shape[1], self.observed_dim)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        flat, u, b, t = self._flat(batch.x, batch.context)
        decoder_params, (mean, logvar), z, prior = self.core.forward(flat, u)
        rec = (self.core.f(z) / self.x_scale).reshape(b, t, self.observed_dim)
        return CausalRepresentationOutput(
            z=z.reshape(b, t, self.latent_dim),
            x_recon=rec,
            extras={"mean": mean.reshape(b, t, self.latent_dim), "logvar": logvar.reshape(b, t, self.latent_dim)},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        flat, u, _, _ = self._flat(batch.x, batch.context)
        prior_params = self.core.prior_params(u)
        g, v = self.core.encoder_params(flat, u)
        z = self.core.encoder_dist.sample(g, v).clamp(-8.0, 8.0)
        decoder_params = self.core.decoder_params(z)
        decoder_mean = decoder_params[0]
        log_px_z = self.core.decoder_dist.log_pdf(flat, decoder_mean, decoder_params[1])
        log_qz_xu = self.core.encoder_dist.log_pdf(z, g, v)
        log_pz_u = self.core.prior_dist.log_pdf(z, *prior_params)
        elbo = (log_px_z + log_pz_u - log_qz_xu).mean()
        return {"loss": elbo, "recon": torch.zeros((), device=elbo.device)}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        return z_past[:, -1]

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "iVAE: with a sufficiently variable auxiliary variable and an "
            "injective mixing map, the latents are identifiable up to "
            "permutation and component-wise transforms. The official iVAE "
            "exponential-family prior and networks are used."
        )

    def ambiguity_class(self) -> AmbiguityClass:
        return AmbiguityClass.PERM_COMPONENTWISE
