"""TDRL: Temporally Disentangled Representation Learning.

Reference: Yao et al., *Temporally Disentangled Representation Learning*, 2022
— BibTeX ``yaoTemporallyDisentangledRepresentation2022``.

The latent space is factorised into three functional groups:

* ``z_fix`` — stationary invariants: transition invariant across domains;
* ``z_dyn`` — transition shifts: transition modulated by a domain parameter;
* ``z_obs`` — observation shifts: invariant (independent) dynamics whose
  observation rendering is domain-modulated by the decoder.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.causal_representation_learning.methods._temporal import lagged_features
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

__all__ = ["TDRL"]


class TDRL(BaseCausalRepresentationLearner):
    """Factorised temporal VAE with modular transition/observation shifts.

    Args:
        observed_dim: dimension ``D`` of ``x``.
        latent_fix_dim / latent_dyn_dim / latent_obs_dim: sizes of the three
            latent groups.
        max_lag: temporal lag ``p`` of the transition priors.
        u_dim: dimension of the domain/context variable.
        hidden_dims / decoder_hidden: network widths.
        kl_weight: scaling of the KL term.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_fix_dim: int,
        latent_dyn_dim: int,
        latent_obs_dim: int,
        max_lag: int = 1,
        u_dim: int = 0,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 1.0,
    ) -> None:
        latent_dim = latent_fix_dim + latent_dyn_dim + latent_obs_dim
        super().__init__(observed_dim=observed_dim, latent_dim=latent_dim, max_lag=max_lag)
        self.fix_dim = latent_fix_dim
        self.dyn_dim = latent_dyn_dim
        self.obs_dim = latent_obs_dim
        self.u_dim = u_dim
        self.kl_weight = kl_weight
        self.encoder = GaussianHead(observed_dim + u_dim, latent_dim, hidden_dims=hidden_dims)
        self.decoder = MLP(latent_dim, observed_dim, hidden_dims=decoder_hidden)
        self.fix_transition = GaussianHead(
            max_lag * latent_dim, latent_fix_dim, hidden_dims=hidden_dims
        )
        self.dyn_transition = GaussianHead(
            max_lag * latent_dim + u_dim, latent_dyn_dim, hidden_dims=hidden_dims
        )
        # Global observation change: z^obs = f_o(theta^obs, eps), i.e. a u-conditioned
        # prior with no temporal dependence (TDRL Eq. 1).
        self.obs_prior = (
            GaussianHead(u_dim, latent_obs_dim, hidden_dims=hidden_dims)
            if u_dim > 0 and latent_obs_dim > 0
            else None
        )
        logger.debug(
            "Initialised TDRL (D=%d, fix=%d, dyn=%d, obs=%d)",
            observed_dim,
            latent_fix_dim,
            latent_dyn_dim,
            latent_obs_dim,
        )

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        x_flat, (batch, time) = flatten_sequence(x)
        u = prepare_context(context, x_flat.shape[0])
        if u is not None:
            x_flat = torch.cat([x_flat, u], dim=-1)
        mean, _ = self.encoder(x_flat)
        return unflatten_sequence(mean, batch, time)

    def decode(self, z: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        # The observation map is x = g(z) (no domain conditioning); observation
        # shifts are carried by the z^obs latent group.
        z_flat, (batch, time) = flatten_sequence(z)
        return unflatten_sequence(self.decoder(z_flat), batch, time)

    def transition(
        self, z_past: Tensor, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        """One-step prior mean (uses the dynamic, domain-conditioned group)."""
        batch, p, dim = z_past.shape
        features = z_past.reshape(batch, p * dim)
        dyn_features = features
        if self.u_dim:
            u = prepare_context(context, batch)
            if u is not None:
                dyn_features = torch.cat([features, u], dim=-1)
        fix_mean, _ = self.fix_transition(features)
        dyn_mean, _ = self.dyn_transition(dyn_features)
        return torch.cat([fix_mean, dyn_mean, z_past[:, -1, self.fix_dim + self.dyn_dim :]], dim=-1)

    def _group_prior(
        self, z_seq: Tensor, u_seq: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        batch, time, dim = z_seq.shape
        mean = torch.zeros(batch, time, dim, device=z_seq.device, dtype=z_seq.dtype)
        logvar = torch.zeros_like(mean)
        features = lagged_features(z_seq, self.max_lag)
        if features is None:
            return mean, logvar
        fix_mean, fix_logvar = self.fix_transition(features.reshape(-1, features.shape[-1]))
        fix_mean = fix_mean.reshape(batch, time - self.max_lag, self.fix_dim)
        fix_logvar = fix_logvar.reshape(batch, time - self.max_lag, self.fix_dim)
        mean[:, self.max_lag :, : self.fix_dim] = fix_mean
        logvar[:, self.max_lag :, : self.fix_dim] = fix_logvar

        dyn_features = features
        if self.u_dim and u_seq is not None:
            dyn_features = torch.cat([features, u_seq[:, self.max_lag :, :]], dim=-1)
        dyn_mean, dyn_logvar = self.dyn_transition(
            dyn_features.reshape(-1, dyn_features.shape[-1])
        )
        dyn_slice = slice(self.fix_dim, self.fix_dim + self.dyn_dim)
        mean[:, self.max_lag :, dyn_slice] = dyn_mean.reshape(
            batch, time - self.max_lag, self.dyn_dim
        )
        logvar[:, self.max_lag :, dyn_slice] = dyn_logvar.reshape(
            batch, time - self.max_lag, self.dyn_dim
        )
        obs_slice = slice(self.fix_dim + self.dyn_dim, dim)
        if self.obs_prior is not None and u_seq is not None:
            obs_mean, obs_logvar = self.obs_prior(u_seq.reshape(-1, self.u_dim))
            mean[:, :, obs_slice] = obs_mean.reshape(batch, time, self.obs_dim)
            logvar[:, :, obs_slice] = obs_logvar.reshape(batch, time, self.obs_dim)
        return mean, logvar

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x_flat, (batch_size, time) = flatten_sequence(batch.x)
        u = prepare_context(batch.context, x_flat.shape[0])
        encoder_input = torch.cat([x_flat, u], dim=-1) if u is not None else x_flat
        q_mean, q_logvar = self.encoder(encoder_input)
        z = reparameterise(q_mean, q_logvar)
        z_seq = unflatten_sequence(z, batch_size, time)
        u_seq = unflatten_sequence(u, batch_size, time) if u is not None else None
        p_mean, p_logvar = self._group_prior(z_seq, u_seq)

        x_recon = self.decoder(z)
        q_mean_seq = q_mean.reshape(batch_size, time, -1)
        q_logvar_seq = q_logvar.reshape(batch_size, time, -1)
        log_q = gaussian_logpdf(z_seq, q_mean_seq, q_logvar_seq)
        log_p = gaussian_logpdf(z_seq, p_mean, p_logvar)
        return CausalRepresentationOutput(
            z=z_seq,
            x_recon=unflatten_sequence(x_recon, batch_size, time),
            log_prob=log_p,
            extras={"q_mean": q_mean, "q_logvar": q_logvar, "log_q": log_q, "log_p": log_p},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None:
            raise ValueError("TDRL forward output is missing the reconstruction.")
        recon = F.mse_loss(outputs.x_recon.reshape_as(batch.x), batch.x)
        extras = outputs.extras or {}
        kl = (extras["log_q"] - extras["log_p"]).sum(dim=1).mean()
        return {"loss": recon + self.kl_weight * kl, "recon": recon, "kl": kl}

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "TDRL: the fix/dyn/obs factorisation makes each group identifiable up "
            "to permutation and component-wise invertible transformations under "
            "domain shifts that modulate only one group at a time."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
