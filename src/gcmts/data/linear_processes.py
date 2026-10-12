r"""Linear Gaussian process generators for CRL benchmarks.

``VARMA`` and Ornstein--Uhlenbeck (``ou``) processes with regime-dependent
coefficients. The state is the latent causal variable and a fixed invertible or
linear mixing maps it to observations.
"""

from __future__ import annotations

from typing import Any

import torch

from gcmts.data.base import BaseDataGenerator
from gcmts.data.synthetic import random_invertible_mixing
from gcmts.typing import Batch

__all__ = ["VARMAProcessGenerator", "OUProcessGenerator", "LINEAR_PROCESSES"]

LINEAR_PROCESSES = ("varma", "ou")


class VARMAProcessGenerator(BaseDataGenerator):
    """Vector ARMA process with a regime-dependent AR scale.

    ``X_t = s_r Σ_τ A_τ X_{t-τ} + ε_t + Σ_τ B_τ ε_{t-τ}`` with a regime gain
    ``s_r`` and fixed random ``A_τ, B_τ``.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 4,
        observed_dim: int | None = None,
        horizon: int = 16,
        max_lag: int = 2,
        n_regimes: int = 3,
        ar_scale: float = 0.5,
        noise_std: float = 0.1,
        mixing: str = "invertible",
        seed: int | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        observed_dim = observed_dim or latent_dim
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.max_lag = max_lag
        self.n_regimes = max(n_regimes, 1)
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.ar = ar_scale * torch.randn(
            max_lag, latent_dim, latent_dim, generator=self.rng
        ) / latent_dim
        self.ma = 0.2 * torch.randn(
            max_lag, latent_dim, latent_dim, generator=self.rng
        ) / latent_dim
        self.regime_gain = torch.linspace(0.6, 1.4, self.n_regimes)
        if mixing == "invertible" and observed_dim == latent_dim:
            self.mixing: torch.nn.Module = random_invertible_mixing(latent_dim, generator=self.rng)
        else:
            self.mixing = torch.nn.Linear(latent_dim, observed_dim, bias=False)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        d = self.latent_dim
        regimes = torch.randint(0, self.n_regimes, (n_samples, horizon), generator=self.rng)
        noise = self.noise_std * torch.randn(n_samples, horizon, d, generator=self.rng)
        x = torch.zeros(n_samples, horizon, d)
        for t in range(horizon):
            gain = self.regime_gain[regimes[:, t]].unsqueeze(-1)
            value = noise[:, t]
            for tau in range(1, self.max_lag + 1):
                if t - tau >= 0:
                    value = value + gain * (x[:, t - tau] @ self.ar[tau - 1].T)
                    value = value + noise[:, t - tau] @ self.ma[tau - 1].T
            x[:, t] = value
        z = x
        with torch.no_grad():
            obs = self.mixing(z.reshape(-1, d)).reshape(n_samples, horizon, self.observed_dim)
        return Batch(x=obs, z=z, context={"u": regimes})


class OUProcessGenerator(BaseDataGenerator):
    """Multivariate Ornstein--Uhlenbeck process with regime-dependent mean reversion."""

    def __init__(
        self,
        *,
        latent_dim: int = 4,
        observed_dim: int | None = None,
        horizon: int = 16,
        n_regimes: int = 3,
        dt: float = 0.2,
        noise_std: float = 0.2,
        mixing: str = "invertible",
        seed: int | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        observed_dim = observed_dim or latent_dim
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.n_regimes = max(n_regimes, 1)
        self.dt = dt
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.theta = torch.linspace(0.5, 2.0, self.n_regimes)
        self.mean = torch.randn(latent_dim, generator=self.rng)
        if mixing == "invertible" and observed_dim == latent_dim:
            self.mixing: torch.nn.Module = random_invertible_mixing(latent_dim, generator=self.rng)
        else:
            self.mixing = torch.nn.Linear(latent_dim, observed_dim, bias=False)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        d = self.latent_dim
        regimes = torch.randint(0, self.n_regimes, (n_samples, horizon), generator=self.rng)
        state = torch.randn(n_samples, d, generator=self.rng)
        states = torch.empty(n_samples, horizon, d)
        for t in range(horizon):
            theta = self.theta[regimes[:, t]].unsqueeze(-1)
            state = state + theta * (self.mean - state) * self.dt + self.noise_std * (
                self.dt**0.5
            ) * torch.randn(n_samples, d, generator=self.rng)
            states[:, t] = state
        z = states
        with torch.no_grad():
            obs = self.mixing(z.reshape(-1, d)).reshape(n_samples, horizon, self.observed_dim)
        return Batch(x=obs, z=z, context={"u": regimes})
