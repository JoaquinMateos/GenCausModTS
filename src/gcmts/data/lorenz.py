r"""Lorenz physical SCM generator for CRL benchmarks.

The Lorenz-63 system is a canonical nonlinear, chaotic dynamical system used as
a physical SCM in the causal-representation literature (e.g.\ self-supervised
temporal contrastive learning). We simulate it with a regime-dependent Rayleigh
parameter so the process is non-stationary, treat the three state coordinates as
the latent causal variables, and map them to observations through a fixed
nonlinear mixing. The temporal order and process-noise independence supply the
identifiability signal; the true latents are exposed for MCC/$R^2$ evaluation.

.. math::
    \dot x = \sigma(y-x),\quad
    \dot y = x(\rho_r - z) - y,\quad
    \dot z = xy - \beta z,
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.data.synthetic import _frozen_mlp
from gcmts.typing import Batch

__all__ = ["LorenzGenerator"]


class LorenzGenerator(BaseDataGenerator):
    """Non-stationary Lorenz-63 process with a nonlinear observation map.

    Args:
        observed_dim: observation dimension ``D``; ``3`` gives an invertible map.
        horizon: number of stored time steps ``T``.
        n_regimes: number of regimes (the Rayleigh parameter varies across them).
        substeps: Euler substeps per stored step (integration resolution).
        dt: Euler step size.
        noise_std: observation-noise std.
        mixing: ``"mlp"`` (nonlinear) or ``"linear"``.
        hidden: hidden width of the mixing MLP.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        observed_dim: int = 6,
        horizon: int = 16,
        n_regimes: int = 3,
        substeps: int = 10,
        dt: float = 0.01,
        noise_std: float = 0.05,
        mixing: str = "mlp",
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.latent_dim = 3
        self.n_regimes = max(n_regimes, 1)
        self.substeps = substeps
        self.dt = dt
        self.noise_std = noise_std
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.rho = torch.linspace(20.0, 35.0, self.n_regimes)
        if mixing == "mlp" or observed_dim != 3:
            self.mixing: torch.nn.Module = _frozen_mlp(
                self.latent_dim, observed_dim, hidden=hidden, generator=self.rng
            )
        else:
            self.mixing = torch.nn.Identity()

    def _drift(self, state: Tensor, rho: Tensor) -> Tensor:
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack(
            [
                self.sigma * (y - x),
                x * (rho - z) - y,
                x * y - self.beta * z,
            ],
            dim=-1,
        )

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        state = torch.randn(n_samples, 3, generator=self.rng) * 2.0 + torch.tensor([1.0, 1.0, 20.0])
        regimes = torch.randint(0, self.n_regimes, (n_samples, horizon), generator=self.rng)
        states = torch.empty(n_samples, horizon, 3)
        for t in range(horizon):
            for _ in range(self.substeps):
                rho = self.rho[regimes[:, t]]
                state = state + self.dt * self._drift(state, rho)
            states[:, t] = state
        # Standardise per trajectory to a common scale (component-wise invertible).
        mean = states.mean(dim=(0, 1), keepdim=True)
        std = states.std(dim=(0, 1), keepdim=True) + 1e-6
        z = (states - mean) / std
        with torch.no_grad():
            x = self.mixing(z.reshape(-1, 3)).reshape(n_samples, horizon, self.observed_dim)
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z, context={"u": regimes})
