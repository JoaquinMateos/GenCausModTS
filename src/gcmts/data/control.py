r"""Interactive control-process generators for CRL benchmarks.

Classic control systems whose state is the latent causal variable and whose
physics is known; a regime-dependent actuation makes the process non-stationary.
Supported: ``pendulum`` and ``mountain_car``.
"""

from __future__ import annotations

from typing import Any

import torch

from gcmts.data.base import BaseDataGenerator
from gcmts.data.synthetic import _frozen_mlp
from gcmts.typing import Batch

__all__ = ["PendulumGenerator", "MountainCarGenerator", "CONTROL_SYSTEMS"]

CONTROL_SYSTEMS = ("pendulum", "mountain_car")


class PendulumGenerator(BaseDataGenerator):
    """Torque-limited pendulum with a regime-dependent gain.

    ``dθ/dt = ω``, ``dω/dt = -(g/l) sin θ - b ω + u_r``; the actuation ``u_r``
    varies across regimes. Observations are a fixed nonlinear mixing of ``(θ, ω)``.
    """

    def __init__(
        self,
        *,
        observed_dim: int = 4,
        horizon: int = 16,
        n_regimes: int = 3,
        substeps: int = 4,
        dt: float = 0.05,
        noise_std: float = 0.02,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.latent_dim = 2
        self.n_regimes = max(n_regimes, 1)
        self.substeps = substeps
        self.dt = dt
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.torque = torch.linspace(-1.5, 1.5, self.n_regimes)
        self.mixing: torch.nn.Module = _frozen_mlp(
            self.latent_dim, observed_dim, hidden=hidden, generator=self.rng
        )

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        theta = torch.pi * (torch.rand(n_samples, generator=self.rng) - 0.5)
        omega = 0.5 * torch.randn(n_samples, generator=self.rng)
        regimes = torch.randint(0, self.n_regimes, (n_samples, horizon), generator=self.rng)
        states = torch.empty(n_samples, horizon, 2)
        for t in range(horizon):
            u = self.torque[regimes[:, t]]
            for _ in range(self.substeps):
                domega = -(10.0 * torch.sin(theta)) - 0.2 * omega + u
                omega = omega + self.dt * domega
                theta = theta + self.dt * omega
            states[:, t] = torch.stack([torch.sin(theta), omega], dim=-1)
        mean = states.mean(dim=(0, 1), keepdim=True)
        std = states.std(dim=(0, 1), keepdim=True) + 1e-6
        z = (states - mean) / std
        with torch.no_grad():
            x = self.mixing(z.reshape(-1, 2)).reshape(n_samples, horizon, self.observed_dim)
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z, context={"u": regimes})


class MountainCarGenerator(BaseDataGenerator):
    """Under-actuated mountain-car with a regime-dependent thrust."""

    def __init__(
        self,
        *,
        observed_dim: int = 4,
        horizon: int = 16,
        n_regimes: int = 3,
        dt: float = 0.05,
        noise_std: float = 0.02,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.latent_dim = 2
        self.n_regimes = max(n_regimes, 1)
        self.dt = dt
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.thrust = torch.linspace(-0.002, 0.003, self.n_regimes)
        self.mixing: torch.nn.Module = _frozen_mlp(
            self.latent_dim, observed_dim, hidden=hidden, generator=self.rng
        )

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        pos = -0.5 + 0.2 * torch.rand(n_samples, generator=self.rng)
        vel = torch.zeros(n_samples)
        regimes = torch.randint(0, self.n_regimes, (n_samples, horizon), generator=self.rng)
        states = torch.empty(n_samples, horizon, 2)
        for t in range(horizon):
            force = self.thrust[regimes[:, t]]
            vel = (vel + force - 0.0025 * torch.cos(3.0 * pos)).clamp(-0.07, 0.07)
            pos = (pos + vel).clamp(-1.2, 0.6)
            states[:, t] = torch.stack([pos, vel], dim=-1)
        mean = states.mean(dim=(0, 1), keepdim=True)
        std = states.std(dim=(0, 1), keepdim=True) + 1e-6
        z = (states - mean) / std
        with torch.no_grad():
            x = self.mixing(z.reshape(-1, 2)).reshape(n_samples, horizon, self.observed_dim)
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z, context={"u": regimes})
