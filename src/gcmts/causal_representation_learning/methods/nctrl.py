"""NCTRL: Temporally Disentangled Representation Learning under Unknown
Non-stationarity.

Reference: Song et al., *Temporally Disentangled Representation Learning under
Unknown Nonstationarity*, NeurIPS 2023 — BibTeX
``songTemporallyDisentangledRepresentation2023``.

Latent regimes ``c_t`` follow an autoregressive hidden Markov model and the
latent transition is regime-dependent, ``p(z_t | z_{t-1}, c_t)``. The discrete
regimes are marginalised exactly with the forward algorithm, so component-wise
identifiability is obtained from observations alone (no domain labels).
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.causal_representation_learning.methods._temporal import hmm_log_marginal
from gcmts.core.backbones import (
    MLP,
    GaussianHead,
    flatten_sequence,
    gaussian_logpdf,
    gaussian_reconstruction_nll,
    reparameterise,
    unflatten_sequence,
)
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["NCTRL"]


class NCTRL(BaseCausalRepresentationLearner):
    """Switching temporal VAE with an exact HMM over latent regimes.

    Args:
        observed_dim: dimension ``D`` of ``x``.
        latent_dim: dimension ``d`` of ``z``.
        n_regimes: number of latent regimes ``R``.
        transition_hidden: hidden width of each regime's transition network.
        hidden_dims / decoder_hidden: network widths.
        kl_weight: scaling of the KL term.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        n_regimes: int = 3,
        transition_hidden: int = 128,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 0.01,
        obs_noise: float = 1.0,
    ) -> None:
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        super().__init__(observed_dim=observed_dim, latent_dim=latent_dim, max_lag=1)
        self.n_regimes = n_regimes
        self.kl_weight = kl_weight
        self.encoder = GaussianHead(observed_dim, latent_dim, hidden_dims=hidden_dims)
        self.decoder = MLP(latent_dim, observed_dim, hidden_dims=decoder_hidden)
        self.transition_nets = nn.ModuleList(
            MLP(latent_dim, latent_dim, hidden_dims=(transition_hidden, transition_hidden))
            for _ in range(n_regimes)
        )
        self.transition_logvar = nn.Parameter(torch.zeros(latent_dim))
        self.hmm_transition = nn.Parameter(torch.zeros(n_regimes, n_regimes))
        self.hmm_initial = nn.Parameter(torch.zeros(n_regimes))
        self.register_buffer("obs_logvar", math.log(obs_noise**2) * torch.ones(1))
        self.obs_logvar: Tensor
        logger.debug(
            "Initialised NCTRL (D=%d, d=%d, R=%d)", observed_dim, latent_dim, n_regimes
        )

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        x_flat, (batch, time) = flatten_sequence(x)
        mean, _ = self.encoder(x_flat)
        return unflatten_sequence(mean, batch, time)

    def decode(self, z: Tensor) -> Tensor:
        z_flat, (batch, time) = flatten_sequence(z)
        return unflatten_sequence(self.decoder(z_flat), batch, time)

    def transition(
        self, z_past: Tensor, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        previous = z_past[:, -1]
        log_pi = torch.log_softmax(self.hmm_initial, dim=-1)
        weights = log_pi.exp().to(previous.device)
        means = torch.stack([net(previous) for net in self.transition_nets], dim=0)
        return (weights.view(-1, 1) * means).sum(dim=0)

    def _log_emissions(self, z_seq: Tensor) -> Tensor:
        batch, time, dim = z_seq.shape
        logvar = self.transition_logvar.clamp(-12.0, 8.0).expand(dim)
        log_emission = torch.zeros(batch, time, self.n_regimes, device=z_seq.device)
        log_emission[:, 0] = gaussian_logpdf(
            z_seq[:, 0], torch.zeros_like(z_seq[:, 0]), torch.zeros_like(z_seq[:, 0])
        ).unsqueeze(-1)
        if time > 1:
            previous = z_seq[:, :-1]
            target = z_seq[:, 1:]
            stack = torch.stack(
                [
                    gaussian_logpdf(target, net(previous), logvar.expand_as(target))
                    for net in self.transition_nets
                ],
                dim=-1,
            )  # (B, T-1, R)
            log_emission[:, 1:] = stack
        return log_emission

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x_flat, (batch_size, time) = flatten_sequence(batch.x)
        q_mean, q_logvar = self.encoder(x_flat)
        z = reparameterise(q_mean, q_logvar)
        z_seq = unflatten_sequence(z, batch_size, time)

        log_emission = self._log_emissions(z_seq)
        log_transition = torch.log_softmax(self.hmm_transition, dim=-1)
        log_initial = torch.log_softmax(self.hmm_initial, dim=-1)
        log_prior = hmm_log_marginal(log_emission, log_transition, log_initial)

        log_q = gaussian_logpdf(
            z_seq, q_mean.reshape(batch_size, time, -1), q_logvar.reshape(batch_size, time, -1)
        ).sum(dim=1)
        return CausalRepresentationOutput(
            z=z_seq,
            x_recon=unflatten_sequence(self.decoder(z), batch_size, time),
            log_prob=log_prior,
            extras={"log_q": log_q, "log_prior": log_prior},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None:
            raise ValueError("NCTRL forward output is missing the reconstruction.")
        recon = gaussian_reconstruction_nll(
            outputs.x_recon.reshape_as(batch.x), batch.x, self.obs_logvar
        )
        extras = outputs.extras or {}
        kl = (extras["log_q"] - extras["log_prior"]).mean()
        return {"loss": recon + self.kl_weight * kl, "recon": recon, "kl": kl}

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "NCTRL: component-wise identifiable from observations alone when latent "
            "regimes follow an HMM and the regime-dependent transitions satisfy a "
            "non-stationarity condition; recovery is up to permutation and "
            "component-wise invertible transforms."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
