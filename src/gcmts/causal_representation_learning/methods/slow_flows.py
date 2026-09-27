"""Slow Flows: time-series source separation with slow normalizing flows.

Reference: Pineau et al., *Time Series Source Separation with Slow Flows*,
2020 — BibTeX ``edouardpineauTimeSeriesSource2020``.

A volume-preserving normalizing flow maps observations to latent sources; a
slow-feature prior ``Delta z_t = z_t - z_{t-1} ~ N(0, I)`` is imposed on the
latent increments, identifying the sources up to a linear demixing.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import RealNVP, flatten_sequence, gaussian_logpdf, unflatten_sequence
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["SlowFlows"]


class SlowFlows(BaseCausalRepresentationLearner):
    """Slow Flows model (SFA inside a normalizing flow).

    Args:
        observed_dim: dimension ``D`` of ``x`` (also the latent dimension).
        n_flow_layers: number of affine-coupling layers.
        hidden_dims: hidden widths of the coupling networks.
        volume_preserving: constrain couplings to have unit Jacobian determinant.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        n_flow_layers: int = 6,
        hidden_dims: tuple[int, ...] = (128, 128),
        volume_preserving: bool = True,
    ) -> None:
        super().__init__(observed_dim=observed_dim, latent_dim=observed_dim, max_lag=1)
        self.flow = RealNVP(
            observed_dim,
            n_layers=n_flow_layers,
            hidden_dims=hidden_dims,
            volume_preserving=volume_preserving,
        )
        logger.debug(
            "Initialised SlowFlows (D=%d, layers=%d, vp=%s)",
            observed_dim,
            n_flow_layers,
            volume_preserving,
        )

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        x_flat, (batch, time) = flatten_sequence(x)
        z, _ = self.flow.forward(x_flat)
        return unflatten_sequence(z, batch, time)

    def decode(self, z: Tensor) -> Tensor:
        z_flat, (batch, time) = flatten_sequence(z)
        return unflatten_sequence(self.flow.inverse(z_flat), batch, time)

    def transition(
        self, z_past: Tensor, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        """Random-walk prior mean: the last latent state."""
        return z_past[:, -1]

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x = batch.x
        batch_size, time, dim = x.shape
        x_flat = x.reshape(batch_size * time, dim)
        z_flat, logdet = self.flow.forward(x_flat)
        z_seq = z_flat.reshape(batch_size, time, dim)

        zeros = torch.zeros(dim, device=x.device, dtype=x.dtype)
        log_prob = gaussian_logpdf(z_seq[:, 0], zeros, zeros)
        if time > 1:
            increments = z_seq[:, 1:] - z_seq[:, :-1]
            log_prob = log_prob + gaussian_logpdf(
                increments.reshape(-1, dim), zeros, zeros
            ).reshape(batch_size, time - 1).sum(dim=1)
        log_prob = log_prob + logdet.reshape(batch_size, time).sum(dim=1)
        nll = -log_prob.mean()

        x_recon = self.flow.inverse(z_flat).reshape(batch_size, time, dim)
        return CausalRepresentationOutput(
            z=z_seq,
            x_recon=x_recon,
            log_prob=log_prob,
            extras={"nll": nll, "logdet": logdet},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None:
            raise ValueError("SlowFlows forward output is missing the reconstruction.")
        recon = F.mse_loss(outputs.x_recon.reshape_as(batch.x), batch.x)
        nll = outputs.extras["nll"] if outputs.extras else torch.tensor(0.0)
        return {"loss": nll, "nll": nll, "recon": recon}

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "Slow Flows: the temporal-slowness prior identifies the nonlinear "
            "sources up to a linear demixing s_t = M f^{-1}(x_t), resolvable by "
            "subsequent linear ICA."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.LINEAR_DEMIXING
