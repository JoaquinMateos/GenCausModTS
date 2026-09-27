"""MOSAIC: Module Discovery via Sparse Additive Identifiable Causal Learning.

Reference: Fan et al., *MOSAIC: Module Discovery via Sparse Additive
Identifiable Causal Learning for Scientific Time Series*, 2026 — BibTeX
``shichengfanMOSAICModuleDiscovery2026``.

The decoder is additive, ``x_i = sum_j g_ij(z_j) + b_i``, so the ANOVA
main-effect supports ``A_ij = 1{x_i depends on z_j}`` are identifiable up to a
permutation of the latent factors. An L1 penalty on the per-pair contributions
encourages sparse module supports.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import (
    MLP,
    GaussianHead,
    flatten_sequence,
    gaussian_logpdf,
    gaussian_reconstruction_nll,
    prepare_context,
    reparameterise,
    unflatten_sequence,
)
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["MOSAIC"]


class _SparseAdditiveDecoder(nn.Module):
    """Additive decoder ``x_i = sum_j g_ij(z_j) + b_i`` with per-pair MLPs."""

    def __init__(
        self,
        observed_dim: int,
        latent_dim: int,
        hidden: tuple[int, ...] = (64,),
    ) -> None:
        super().__init__()
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.nets = nn.ModuleList(
            MLP(1, 1, hidden_dims=hidden) for _ in range(observed_dim * latent_dim)
        )
        self.bias = nn.Parameter(torch.zeros(observed_dim))

    def _contributions(self, z: Tensor) -> Tensor:
        batch = z.shape[0]
        out = self.bias.unsqueeze(0).expand(batch, -1).clone()
        for i in range(self.observed_dim):
            for j in range(self.latent_dim):
                out[:, i] = out[:, i] + self.nets[i * self.latent_dim + j](
                    z[:, j : j + 1]
                ).squeeze(-1)
        return out

    def forward(self, z: Tensor) -> Tensor:
        return self._contributions(z)

    def importance(self, z: Tensor) -> Tensor:
        """Mean absolute contribution ``|g_ij(z_j)|`` for each pair ``(i, j)``."""
        batch = z.shape[0]
        importance = torch.zeros(
            self.observed_dim, self.latent_dim, device=z.device, dtype=z.dtype
        )
        for i in range(self.observed_dim):
            for j in range(self.latent_dim):
                contribution = self.nets[i * self.latent_dim + j](z[:, j : j + 1])
                importance[i, j] = contribution.abs().mean()
        del batch
        return importance


class MOSAIC(BaseCausalRepresentationLearner):
    """Sparse additive temporal VAE with module-support discovery.

    Args:
        observed_dim: dimension ``D`` of ``x``.
        latent_dim: dimension ``d`` of modules ``z``.
        hidden_dims / decoder_hidden: encoder and per-pair decoder widths.
        sparsity: weight of the L1 support penalty.
        kl_weight: scaling of the KL term.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        u_dim: int = 0,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (64,),
        sparsity: float = 0.01,
        kl_weight: float = 1.0,
        obs_noise: float = 1.0,
    ) -> None:
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        super().__init__(observed_dim=observed_dim, latent_dim=latent_dim, max_lag=1)
        self.u_dim = u_dim
        self.sparsity = sparsity
        self.kl_weight = kl_weight
        # Latents are identified from regime-conditioned temporal variation, so the
        # encoder conditions on the regime/domain context u.
        self.encoder = GaussianHead(observed_dim + u_dim, latent_dim, hidden_dims=hidden_dims)
        self.decoder = _SparseAdditiveDecoder(observed_dim, latent_dim, hidden=decoder_hidden)
        self.support: Tensor | None = None
        self.register_buffer("obs_logvar", math.log(obs_noise**2) * torch.ones(1))
        self.obs_logvar: Tensor
        logger.debug("Initialised MOSAIC (D=%d, d=%d, u=%d)", observed_dim, latent_dim, u_dim)

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
        return z_past[:, -1]

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x_flat, (batch_size, time) = flatten_sequence(batch.x)
        u = prepare_context(batch.context, x_flat.shape[0])
        q_mean, q_logvar = self.encoder(self._condition(x_flat, u))
        z = reparameterise(q_mean, q_logvar)
        importance = self.decoder.importance(z)
        self.support = importance.detach()
        log_q = gaussian_logpdf(z, q_mean, q_logvar)
        log_p = gaussian_logpdf(z, torch.zeros_like(z), torch.zeros_like(z))
        return CausalRepresentationOutput(
            z=unflatten_sequence(z, batch_size, time),
            x_recon=unflatten_sequence(self.decoder(z), batch_size, time),
            extras={
                "log_q": log_q,
                "log_p": log_p,
                "importance": importance,
            },
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None or outputs.extras is None:
            raise ValueError("MOSAIC forward output is incomplete.")
        recon = gaussian_reconstruction_nll(
            outputs.x_recon.reshape_as(batch.x), batch.x, self.obs_logvar
        )
        kl = (outputs.extras["log_q"] - outputs.extras["log_p"]).mean()
        sparsity = outputs.extras["importance"].sum()
        total = recon + self.kl_weight * kl + self.sparsity * sparsity
        return {"loss": total, "recon": recon, "kl": kl, "sparsity": sparsity}

    def get_latent_adjacency(self) -> Tensor | None:
        """Return the discovered ``(D, d)`` module-support matrix if available."""
        return self.support

    def identifiability_statement(self) -> str:
        return (
            "MOSAIC: the ANOVA main-effect supports A_ij = 1{x_i depends on z_j} "
            "are identifiable up to permutation of the latent modules under "
            "smooth additive mixing."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
