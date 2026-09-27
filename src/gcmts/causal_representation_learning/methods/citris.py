"""CITRIS: Causal Identifiability from Temporal Intervened Sequences.

Reference: Lippe et al., *CITRIS: Causal Identifiability from Temporal
Intervened Sequences*, ICML 2022 — BibTeX
``lippeCITRISCausalIdentifiability2022``.

An invertible autoencoder maps observations to ``M`` latent components that
are assigned to ``K`` causal variables plus one shared group. The transition
prior is conditioned on binary intervention targets ``I``; minimising the
information in the shared group recovers minimal causal variables up to
element-wise invertible transformations.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import (
    MLP,
    GradientReversal,
    RealNVP,
    gaussian_logpdf,
)
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["CITRIS"]


class CITRIS(BaseCausalRepresentationLearner):
    """Intervention-conditioned causal representation learner (CITRIS).

    Args:
        observed_dim: dimension ``D`` of ``x`` (equal to the latent dimension
            because the autoencoder is invertible).
        n_vars: number ``K`` of causal variables.
        var_dim: dimension of each causal variable group.
        n_shared: dimension of the shared (intervention-independent) group.
        n_flow_layers: number of coupling layers in the invertible autoencoder.
        hidden_dims: hidden widths of the coupling/transition networks.
        cls_weight: weight of the shared-group intervention classifier.
        recon_weight: weight of an explicit reconstruction penalty.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        n_vars: int,
        var_dim: int,
        n_shared: int = 1,
        n_flow_layers: int = 6,
        hidden_dims: tuple[int, ...] = (128, 128),
        cls_weight: float = 1.0,
        recon_weight: float = 1.0,
    ) -> None:
        latent_dim = n_vars * var_dim + n_shared
        if latent_dim != observed_dim:
            raise ValueError(
                "CITRIS uses an invertible autoencoder, so n_vars*var_dim + "
                f"n_shared must equal observed_dim (got {latent_dim} vs {observed_dim})."
            )
        super().__init__(observed_dim=observed_dim, latent_dim=latent_dim, max_lag=1)
        self.n_vars = n_vars
        self.var_dim = var_dim
        self.n_shared = n_shared
        self.cls_weight = cls_weight
        self.recon_weight = recon_weight
        self.flow = RealNVP(
            observed_dim, n_layers=n_flow_layers, hidden_dims=hidden_dims
        )
        self.transition_nets = nn.ModuleList(
            MLP(observed_dim, 2 * var_dim, hidden_dims=hidden_dims)
            for _ in range(n_vars)
        )
        self.shared_net = (
            MLP(observed_dim, 2 * n_shared, hidden_dims=hidden_dims)
            if n_shared > 0
            else None
        )
        self.intervention_mean = nn.Parameter(torch.zeros(var_dim))
        self.intervention_logvar = nn.Parameter(torch.zeros(var_dim))
        self.classifier = (
            MLP(n_shared, n_vars, hidden_dims=hidden_dims) if n_shared > 0 else None
        )
        self.grl = GradientReversal(1.0)
        logger.debug(
            "Initialised CITRIS (D=%d, K=%d, var_dim=%d, shared=%d)",
            observed_dim,
            n_vars,
            var_dim,
            n_shared,
        )

    def _groups(self) -> list[slice]:
        groups = [
            slice(i * self.var_dim, (i + 1) * self.var_dim) for i in range(self.n_vars)
        ]
        if self.n_shared > 0:
            groups.append(slice(self.n_vars * self.var_dim, self.latent_dim))
        return groups

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        z_flat, (batch, time) = self._flatten(x)
        z, _ = self.flow.forward(z_flat)
        return z.reshape(batch, time, -1)

    def decode(self, z: Tensor) -> Tensor:
        z_flat, (batch, time) = self._flatten(z)
        return self.flow.inverse(z_flat).reshape(batch, time, -1)

    @staticmethod
    def _flatten(value: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if value.ndim == 2:
            return value, (value.shape[0], 1)
        batch, time, dim = value.shape
        return value.reshape(batch * time, dim), (batch, time)

    def transition(
        self, z_past: Tensor, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        """Return the last latent state (used only by the generic forecaster)."""
        return z_past[:, -1]

    def _transition_log_prob(
        self, z_seq: Tensor, interventions: Tensor | None
    ) -> Tensor:
        """Sum of transition log-densities for ``t = 1..T-1`` per trajectory.

        Fully vectorised over time: each variable group's transition network is
        applied once to all ``(B, T-1)`` previously-latent states.
        """
        batch, time, _ = z_seq.shape
        if time <= 1:
            return torch.zeros(batch, device=z_seq.device, dtype=z_seq.dtype)
        groups = self._groups()
        previous = z_seq[:, :-1].reshape(batch * (time - 1), -1)  # (B*(T-1), M)
        prediction = z_seq[:, 1:]  # (B, T-1, M)
        log_prob = torch.zeros(batch, time - 1, device=z_seq.device, dtype=z_seq.dtype)

        for index, group in enumerate(groups[:-1] if self.n_shared > 0 else groups):
            mean_all, logvar_all = self.transition_nets[index](previous).chunk(2, dim=-1)
            mean_all = mean_all.reshape(batch, time - 1, self.var_dim)
            logvar_all = logvar_all.reshape(batch, time - 1, self.var_dim)
            if interventions is not None:
                mask = interventions[:, 1:, index].unsqueeze(-1)  # (B, T-1, 1)
                mean = mask * self.intervention_mean + (1.0 - mask) * mean_all
                logvar = mask * self.intervention_logvar + (1.0 - mask) * logvar_all
            else:
                mean, logvar = mean_all, logvar_all
            log_prob = log_prob + gaussian_logpdf(
                prediction[:, :, group], mean, logvar.clamp(-12.0, 8.0)
            )

        if self.n_shared > 0 and self.shared_net is not None:
            shared_mean, shared_logvar = self.shared_net(previous).chunk(2, dim=-1)
            shared_mean = shared_mean.reshape(batch, time - 1, self.n_shared)
            shared_logvar = shared_logvar.reshape(batch, time - 1, self.n_shared)
            log_prob = log_prob + gaussian_logpdf(
                prediction[:, :, groups[-1]],
                shared_mean,
                shared_logvar.clamp(-12.0, 8.0),
            )
        return log_prob.sum(dim=1)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x = batch.x
        batch_size, time, dim = x.shape
        x_flat = x.reshape(batch_size * time, dim)
        z_flat, logdet = self.flow.forward(x_flat)
        z_seq = z_flat.reshape(batch_size, time, dim)

        interventions = None
        if batch.context is not None and "I" in batch.context:
            interventions = batch.context["I"].to(x.device)

        zeros = torch.zeros(dim, device=x.device, dtype=x.dtype)
        initial_log_prob = gaussian_logpdf(z_seq[:, 0], zeros, zeros)
        log_prob = initial_log_prob + self._transition_log_prob(z_seq, interventions)
        log_prob = log_prob + logdet.reshape(batch_size, time).sum(dim=1)
        nll = -log_prob.mean()

        cls_loss = torch.zeros((), device=x.device)
        if self.classifier is not None and interventions is not None and time > 1:
            shared = z_seq[:, 1:, self._groups()[-1]].reshape(
                batch_size * (time - 1), self.n_shared
            )
            logits = self.classifier(self.grl(shared)).reshape(
                batch_size, time - 1, self.n_vars
            )
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, interventions[:, 1:]
            )

        x_recon = self.flow.inverse(z_flat).reshape(batch_size, time, dim)
        return CausalRepresentationOutput(
            z=z_seq,
            x_recon=x_recon,
            log_prob=log_prob,
            extras={"nll": nll, "cls": cls_loss, "logdet": logdet},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None:
            raise ValueError("CITRIS forward output is missing the reconstruction.")
        extras = outputs.extras or {}
        nll = extras["nll"]
        cls = extras["cls"]
        recon = F.mse_loss(outputs.x_recon.reshape_as(batch.x), batch.x)
        total = nll + self.recon_weight * recon + self.cls_weight * cls
        return {"loss": total, "nll": nll, "recon": recon, "cls": cls}

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "CITRIS: recovers minimal causal variables up to element-wise "
            "invertible transformations given known intervention targets and an "
            "invertible observation map; the shared group carries no "
            "intervention information after training."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
