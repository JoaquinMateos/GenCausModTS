"""LEAP: Learning Temporally Causal Latent Processes.

Reference: Yao et al., *Learning Temporally Causal Latent Processes from
General Temporal Data*, ICLR 2022 — BibTeX
``weiranyaoLearningTemporallyCausal2021``.

LEAP is an iVAE whose prior is a non-stationary causal-process prior
``p(z_t | z_{t-p:t-1}, u_t)``. Identifiability follows from non-stationary,
regime-dependent conditionals whose log-density gradients are linearly
independent across regimes.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import (
    MLP,
    GaussianHead,
    flatten_sequence,
    gaussian_logpdf,
    prepare_context,
    reparameterise,
    unflatten_sequence,
)
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["LEAP"]


class LEAP(BaseCausalRepresentationLearner):
    """Temporal iVAE with a causal-process prior.

    Args:
        observed_dim: dimension ``D`` of ``x``.
        latent_dim: dimension ``d`` of ``z``.
        max_lag: maximum temporal lag ``p`` of the latent mechanisms.
        u_dim: dimension of the regime/context variable ``u`` (0 = stationary).
        hidden_dims: hidden widths for the encoder/transition heads.
        decoder_hidden: hidden widths for the decoder.
        kl_weight: scaling of the KL term.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        max_lag: int = 1,
        u_dim: int = 0,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__(
            observed_dim=observed_dim, latent_dim=latent_dim, max_lag=max_lag
        )
        self.u_dim = u_dim
        self.kl_weight = kl_weight
        self.encoder = GaussianHead(
            observed_dim + u_dim, latent_dim, hidden_dims=hidden_dims
        )
        self.decoder = MLP(latent_dim, observed_dim, hidden_dims=decoder_hidden)
        self.transition_net = GaussianHead(
            max_lag * latent_dim + u_dim, latent_dim, hidden_dims=hidden_dims
        )
        logger.debug(
            "Initialised LEAP (D=%d, d=%d, p=%d, u=%d)",
            observed_dim,
            latent_dim,
            max_lag,
            u_dim,
        )

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

    def _temporal_prior(
        self, z: Tensor, u_seq: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Compute ``p(z_t | z_{t-p:t-1}, u_t)`` means/logvars for all ``t``.

        Positions ``t < p`` fall back to a standard normal prior.
        """
        batch, time, dim = z.shape
        p = self.max_lag
        mean = torch.zeros(batch, time, dim, device=z.device, dtype=z.dtype)
        logvar = torch.zeros_like(mean)
        if time <= p:
            return mean, logvar
        windows = z.unfold(1, p, 1)[:, : time - p]  # (B, T-p, d, p)
        windows = windows.permute(0, 1, 3, 2).reshape(batch, time - p, p * dim)
        features = windows
        if self.u_dim:
            if u_seq is None:
                raise ValueError("LEAP with u_dim>0 requires a context variable.")
            features = torch.cat([features, u_seq[:, p:, :]], dim=-1)
        flat_mean, flat_logvar = self.transition_net(
            features.reshape(-1, features.shape[-1])
        )
        mean[:, p:] = flat_mean.reshape(batch, time - p, dim)
        logvar[:, p:] = flat_logvar.reshape(batch, time - p, dim)
        return mean, logvar

    def transition(
        self, z_past: Tensor, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        """One-step prior mean given ``(B, p, d)`` lagged latents and context."""
        batch, p, dim = z_past.shape
        features = z_past.reshape(batch, p * dim)
        if self.u_dim:
            u = prepare_context(context, batch)
            if u is None:
                raise ValueError("LEAP with u_dim>0 requires a context variable.")
            features = torch.cat([features, u], dim=-1)
        mean, _ = self.transition_net(features)
        return cast(Tensor, mean)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x_flat, (batch_size, time) = flatten_sequence(batch.x)
        u = prepare_context(batch.context, x_flat.shape[0])
        q_mean, q_logvar = self.encoder(self._condition(x_flat, u))
        z = reparameterise(q_mean, q_logvar)
        z_seq = unflatten_sequence(z, batch_size, time)
        u_seq = (
            unflatten_sequence(u, batch_size, time) if u is not None else None
        )
        p_mean, p_logvar = self._temporal_prior(z_seq, u_seq)
        q_mean_seq = q_mean.reshape(batch_size, time, -1)
        q_logvar_seq = q_logvar.reshape(batch_size, time, -1)
        log_q = gaussian_logpdf(z_seq, q_mean_seq, q_logvar_seq)
        log_p = gaussian_logpdf(z_seq, p_mean, p_logvar)
        return CausalRepresentationOutput(
            z=z_seq,
            x_recon=unflatten_sequence(self.decoder(z), batch_size, time),
            log_prob=log_p,
            extras={
                "q_mean": q_mean,
                "q_logvar": q_logvar,
                "p_mean": p_mean.reshape(batch_size * time, -1),
                "p_logvar": p_logvar.reshape(batch_size * time, -1),
                "log_q": log_q.reshape(batch_size, time),
                "log_p": log_p,
            },
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None:
            raise ValueError("LEAP forward output is missing the reconstruction.")
        recon = F.mse_loss(outputs.x_recon.reshape_as(batch.x), batch.x)
        extras = outputs.extras or {}
        kl = (extras["log_q"] - extras["log_p"]).sum(dim=1).mean()
        total = recon + self.kl_weight * kl
        return {"loss": total, "recon": recon, "kl": kl}

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "LEAP: identifiable up to permutation and component-wise invertible "
            "transformations when the regime-dependent conditionals "
            "p(z_t | z_{t-p:t-1}, u_t) have linearly independent log-density "
            "gradients across regimes."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
