"""Identifiable VAE (iVAE).

Reference: Khemakhem et al., *Variational Autoencoders and Nonlinear ICA: A
Unifying Framework*, AISTATS 2020 — BibTeX ``khemakhemVariationalAutoencodersNonlinear2020``.

The model learns ``x = g(z) + eta`` with an auxiliary-conditioned factorised
exponential-family prior ``p(z|u) = prod_i p(z_i|u)``. Identifiability holds up
to permutation and component-wise invertible transformations.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import (
    MLP,
    ExpFamilyPrior,
    GaussianHead,
    flatten_sequence,
    gaussian_logpdf,
    prepare_context,
    reparameterise,
    unflatten_sequence,
)
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["IVAE"]


class IVAE(BaseCausalRepresentationLearner):
    """Auxiliary-variable identifiable VAE (iVAE).

    Args:
        observed_dim: dimension ``D`` of ``x``.
        latent_dim: dimension ``d`` of ``z``.
        u_dim: dimension of the auxiliary/context variable ``u`` (0 = VAE).
        hidden_dims: hidden widths for the encoder/prior heads.
        decoder_hidden: hidden widths for the decoder.
        kl_weight: scaling of the KL term (beta-VAE style).
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        u_dim: int = 0,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__(observed_dim=observed_dim, latent_dim=latent_dim, max_lag=1)
        self.u_dim = u_dim
        self.kl_weight = kl_weight
        self.encoder = GaussianHead(
            observed_dim + u_dim, latent_dim, hidden_dims=hidden_dims
        )
        self.decoder = MLP(latent_dim, observed_dim, hidden_dims=decoder_hidden)
        self.prior = ExpFamilyPrior(u_dim, latent_dim, hidden_dims=hidden_dims)
        logger.debug("Initialised IVAE (D=%d, d=%d, u=%d)", observed_dim, latent_dim, u_dim)

    def _condition(self, x: Tensor, u: Tensor | None) -> Tensor:
        return x if u is None else torch.cat([x, u], dim=-1)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        x_flat, (batch, time) = flatten_sequence(x)
        u = prepare_context(context, x_flat.shape[0])
        mean, _ = self.encoder(self._condition(x_flat, u))
        return unflatten_sequence(mean, batch, time)

    def decode(self, z: Tensor) -> Tensor:
        z_flat, (batch, time) = flatten_sequence(z)
        return unflatten_sequence(self.decoder(z_flat), batch, time)

    def transition(
        self, z_past: Tensor, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        """The prior is static, so the one-step mean is the identity."""
        return z_past[:, -1]

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x_flat, (batch_size, time) = flatten_sequence(batch.x)
        u = prepare_context(batch.context, x_flat.shape[0])
        q_mean, q_logvar = self.encoder(self._condition(x_flat, u))
        z = reparameterise(q_mean, q_logvar)
        p_mean, p_logvar = self.prior(u)
        log_q = gaussian_logpdf(z, q_mean, q_logvar)
        log_p = gaussian_logpdf(z, p_mean, p_logvar)
        return CausalRepresentationOutput(
            z=unflatten_sequence(z, batch_size, time),
            x_recon=unflatten_sequence(self.decoder(z), batch_size, time),
            log_prob=log_p.reshape(batch_size, time),
            extras={
                "q_mean": q_mean,
                "q_logvar": q_logvar,
                "p_mean": p_mean,
                "p_logvar": p_logvar,
                "log_q": log_q,
                "log_p": log_p,
            },
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None:
            raise ValueError("IVAE forward output is missing the reconstruction.")
        recon = F.mse_loss(outputs.x_recon.reshape_as(batch.x), batch.x)
        extras = outputs.extras or {}
        kl = (extras["log_q"] - extras["log_p"]).mean()
        total = recon + self.kl_weight * kl
        return {"loss": total, "recon": recon, "kl": kl}

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "iVAE: identifiable up to permutation and component-wise invertible "
            "transformations given an auxiliary variable u that modulates an "
            "exponential-family prior p(z|u) with sufficient variability."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
