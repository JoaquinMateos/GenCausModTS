"""CEGEN: Conditional Loss and Deep Euler Scheme for time-series generation.

Reference: Remlinger et al., *Conditional Loss and Deep Euler Scheme for Time
Series Generation*, 2021 — BibTeX ``carlremlingerConditionalLossDeep2021``.

The process is modelled directly in observation space by a Deep Euler step

.. math::
    z_{t+1} = z_t + f_\\theta(z_t) \\Delta t
              + G_\\theta(z_t) \\sqrt{\\Delta t}\\,\\epsilon_t,

and trained by matching step-wise conditional transitions in regions
``(I_k)`` with the closed-form Wasserstein-2 (Bures) distance between
Gaussians: ``sum_k W2^2(p(z_{t+1}|z_t in I_k), p(z_hat_{t+1}|z_hat_t in I_k))``.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import MLP
from gcmts.core.solvers import bures_w2
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["CEGEN"]

_EPS = 1e-4


class CEGEN(BaseCausalRepresentationLearner):
    """Deep-Euler neural SDE trained with a conditional Wasserstein loss.

    Operates directly in observation space, so ``encode``/``decode`` are the
    identity and ``latent_dim == observed_dim``.

    Args:
        observed_dim: dimension ``D`` of the process (also the latent dimension).
        hidden_dims: width of the drift/volatility MLPs.
        n_regions: number ``K`` of regions partitioning the state space.
        dt: Euler step size (matches the sampling interval of the data).
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        n_regions: int = 4,
        dt: float = 1.0,
        diffusion_min: float = 1e-3,
    ) -> None:
        super().__init__(observed_dim=observed_dim, latent_dim=observed_dim, max_lag=1)
        self.drift_net = MLP(observed_dim, observed_dim, hidden_dims=hidden_dims)
        # Volatility outputs a full D x D matrix so the model covariance G G^T can
        # match non-diagonal empirical covariances (CEGEN, Ito diffusion).
        self.diffusion_net = MLP(observed_dim, observed_dim * observed_dim, hidden_dims=hidden_dims)
        self.n_regions = n_regions
        self.dt = dt
        self.diffusion_min = diffusion_min
        logger.debug("Initialised CEGEN (D=%d, K=%d, dt=%.3g)", observed_dim, n_regions, dt)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        return x

    def decode(self, z: Tensor) -> Tensor:
        return z

    def _diffusion_matrix(self, z: Tensor) -> Tensor:
        """Return the ``(N, D, D)`` volatility matrix ``G_theta(z)``."""
        batch = z.shape[0]
        raw = self.diffusion_net(z).reshape(batch, self.observed_dim, self.observed_dim)
        return torch.tril(raw) + self.diffusion_min * torch.eye(
            self.observed_dim, device=z.device, dtype=z.dtype
        )

    def transition(
        self, z_past: Tensor, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        state = z_past[:, -1]
        return cast(Tensor, state + self.drift_net(state) * self.dt)

    def _regions(self, states: Tensor) -> Tensor:
        if states.shape[0] < 2:
            return torch.zeros(states.shape[0], dtype=torch.long, device=states.device)
        n_regions = min(self.n_regions, states.shape[0])
        norms = states.norm(dim=-1)
        quantiles = torch.quantile(
            norms,
            torch.linspace(0.0, 1.0, n_regions + 1, device=states.device)[1:-1],
        )
        return torch.bucketize(norms, quantiles)

    def _conditional_w2(self, start: Tensor, target: Tensor) -> Tensor:
        regions = self._regions(start)
        drift = self.drift_net(start)
        diffusion = self._diffusion_matrix(start)
        dim = start.shape[-1]
        eye = torch.eye(dim, device=start.device, dtype=start.dtype)
        losses: list[Tensor] = []
        for region in torch.unique(regions):
            mask = regions == region
            if int(mask.sum()) < 2:
                continue
            dx_empirical = (target - start)[mask]
            mean_empirical = dx_empirical.mean(dim=0)
            centered = dx_empirical - mean_empirical
            cov_empirical = (centered.T @ centered) / (dx_empirical.shape[0] - 1) + _EPS * eye
            mean_model = (drift[mask] * self.dt).mean(dim=0)
            region_g = diffusion[mask]
            cov_model = (region_g @ region_g.transpose(-1, -2)).mean(dim=0) * self.dt + _EPS * eye
            losses.append(
                bures_w2(
                    mean_model.unsqueeze(0),
                    cov_model.unsqueeze(0),
                    mean_empirical.unsqueeze(0),
                    cov_empirical.unsqueeze(0),
                ).sum()
            )
        if not losses:
            return torch.zeros((), device=start.device, dtype=start.dtype)
        return torch.stack(losses).mean()

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x = batch.x
        batch_size, time, dim = x.shape
        start = x[:, :-1].reshape(-1, dim)
        target = x[:, 1:].reshape(-1, dim)
        w2 = self._conditional_w2(start, target)
        x_recon = x.clone()
        x_recon[:, 1:] = (start + self.drift_net(start) * self.dt).reshape(
            batch_size, time - 1, dim
        )
        return CausalRepresentationOutput(
            z=x,
            x_recon=x_recon,
            extras={"w2": w2},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.extras is None or outputs.x_recon is None:
            raise ValueError("CEGEN forward output is incomplete.")
        w2 = outputs.extras["w2"]
        recon = F.mse_loss(outputs.x_recon.reshape_as(batch.x), batch.x)
        return {"loss": w2, "w2": w2, "one_step_mse": recon}

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "CEGEN: the conditional-transition loss guarantees exact estimation of "
            "the drift f and volatility G when the observation process is directly "
            "observed (no mixing); no latent demixing ambiguity."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.UNKNOWN
