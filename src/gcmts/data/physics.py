r"""Physical and nonlinear SCM generators for CRL benchmarks.

A family of canonical nonlinear dynamical systems used as ground-truth SCMs in
the causal-representation literature. Each system is integrated with a
regime-dependent parameter (so the process is non-stationary), the state
coordinates are treated as the latent causal variables, and a fixed nonlinear
mixing maps them to observations. The true latents are exposed for MCC/$R^2$
evaluation; the temporal order and process-noise independence supply the
identifiability signal.

Supported systems (``system=``): ``lorenz96``, ``pcl`` (coupled Lorenz-63),
``kuramoto``, ``lotka_volterra``, ``duffing``, ``fitzhugh_nagumo``, ``chua``,
``van_der_pol``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.data.synthetic import _frozen_mlp
from gcmts.typing import Batch

__all__ = ["PhysicsSCMGenerator", "PHYSICS_SYSTEMS"]

PHYSICS_SYSTEMS = (
    "lorenz96",
    "pcl",
    "kuramoto",
    "lotka_volterra",
    "duffing",
    "fitzhugh_nagumo",
    "chua",
    "van_der_pol",
)

# system -> (latent dim, parameter range across regimes)
_DIMS = {
    "lorenz96": 5,
    "pcl": 6,
    "kuramoto": 6,
    "lotka_volterra": 4,
    "duffing": 2,
    "fitzhugh_nagumo": 2,
    "chua": 3,
    "van_der_pol": 2,
}
_PARAM_RANGE = {
    "lorenz96": (4.0, 12.0),
    "pcl": (20.0, 35.0),
    "kuramoto": (0.5, 3.0),
    "lotka_volterra": (0.6, 1.4),
    "duffing": (0.2, 0.8),
    "fitzhugh_nagumo": (0.2, 0.8),
    "chua": (13.0, 16.0),
    "van_der_pol": (0.5, 2.5),
}


def _lorenz96(x: Tensor, p: Tensor) -> Tensor:
    return (torch.roll(x, -1, dims=1) - torch.roll(x, 2, dims=1)) * torch.roll(
        x, 1, dims=1
    ) - x + p.unsqueeze(-1)


def _pcl(x: Tensor, p: Tensor) -> Tensor:
    # Two coupled Lorenz-63 blocks; the Rayleigh parameter differs by block.
    a, b = x[:, :3], x[:, 3:]
    rho = p.unsqueeze(-1)
    da = torch.stack(
        [10.0 * (a[:, 1] - a[:, 0]), a[:, 0] * (rho[:, 0] - a[:, 2]) - a[:, 1],
         a[:, 0] * a[:, 1] - (8.0 / 3.0) * a[:, 2]], dim=-1)
    db = torch.stack(
        [10.0 * (b[:, 1] - b[:, 0]) + 0.1 * a[:, 0],
         b[:, 0] * (rho[:, 0] + 3.0 - b[:, 2]) - b[:, 1],
         b[:, 0] * b[:, 1] - (8.0 / 3.0) * b[:, 2]], dim=-1)
    return torch.cat([da, db], dim=-1)


def _kuramoto(x: Tensor, p: Tensor) -> Tensor:
    diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, d, d)
    coupling = torch.sin(diff).mean(dim=2)
    return 1.0 + p.unsqueeze(-1) * coupling


def _lotka_volterra(x: Tensor, p: Tensor) -> Tensor:
    d = x.shape[1]
    interaction = torch.eye(d, device=x.device) * -1.0 + 0.15
    return x * (1.0 - p.unsqueeze(-1) * (x @ interaction.T))


def _duffing(x: Tensor, p: Tensor) -> Tensor:
    pos, vel = x[:, 0], x[:, 1]
    return torch.stack([vel, -0.3 * vel - pos - pos**3 + p], dim=-1)


def _fitzhugh_nagumo(x: Tensor, p: Tensor) -> Tensor:
    v, w = x[:, 0], x[:, 1]
    return torch.stack([v - v**3 / 3.0 - w + p, 0.08 * (v + 0.7 - 0.8 * w)], dim=-1)


def _chua(x: Tensor, p: Tensor) -> Tensor:
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    f = -0.6 * x0 + 0.5 * (x0.abs() + 1 - (x0.abs() - 1).abs())
    return torch.stack([p * (x1 - x0 - f), x0 - x1 + x2, -15.0 * x1], dim=-1)


def _van_der_pol(x: Tensor, p: Tensor) -> Tensor:
    pos, vel = x[:, 0], x[:, 1]
    return torch.stack([vel, p * (1.0 - pos**2) * vel - pos], dim=-1)


_DRIFT: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "lorenz96": _lorenz96,
    "pcl": _pcl,
    "kuramoto": _kuramoto,
    "lotka_volterra": _lotka_volterra,
    "duffing": _duffing,
    "fitzhugh_nagumo": _fitzhugh_nagumo,
    "chua": _chua,
    "van_der_pol": _van_der_pol,
}


class PhysicsSCMGenerator(BaseDataGenerator):
    """Regime-parameterised physical SCM with a nonlinear observation map.

    Args:
        system: one of :data:`PHYSICS_SYSTEMS`.
        observed_dim: observation dimension ``D`` (defaults to the latent dim).
        horizon: sequence length ``T``.
        n_regimes: number of regimes (the system parameter varies across them).
        substeps: Euler substeps per stored step.
        dt: Euler step size.
        noise_std: observation-noise std.
        mixing: ``"mlp"`` (nonlinear) or ``"identity"``.
        hidden: hidden width of the mixing MLP.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        system: str = "lorenz96",
        observed_dim: int | None = None,
        horizon: int = 16,
        n_regimes: int = 3,
        substeps: int = 8,
        dt: float = 0.01,
        noise_std: float = 0.05,
        mixing: str = "mlp",
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        if system not in _DRIFT:
            raise ValueError(f"Unknown system '{system}'. Choose from {PHYSICS_SYSTEMS}.")
        self.latent_dim = _DIMS[system]
        observed_dim = observed_dim or self.latent_dim
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.system = system
        self.n_regimes = max(n_regimes, 1)
        self.substeps = substeps
        self.dt = dt
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        low, high = _PARAM_RANGE[system]
        self.param = torch.linspace(low, high, self.n_regimes)
        self.drift = _DRIFT[system]
        if mixing == "mlp" and observed_dim != self.latent_dim:
            self.mixing: torch.nn.Module = _frozen_mlp(
                self.latent_dim, observed_dim, hidden=hidden, generator=self.rng
            )
        else:
            self.mixing = torch.nn.Identity()

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        d = self.latent_dim
        state = torch.randn(n_samples, d, generator=self.rng)
        if self.system in {"kuramoto"}:
            state = state * 3.0
        regimes = torch.randint(0, self.n_regimes, (n_samples, horizon), generator=self.rng)
        states = torch.empty(n_samples, horizon, d)
        for t in range(horizon):
            p = self.param[regimes[:, t]]
            for _ in range(self.substeps):
                state = state + self.dt * self.drift(state, p)
            states[:, t] = state
        mean = states.mean(dim=(0, 1), keepdim=True)
        std = states.std(dim=(0, 1), keepdim=True) + 1e-6
        z = (states - mean) / std
        with torch.no_grad():
            x = self.mixing(z.reshape(-1, d)).reshape(n_samples, horizon, self.observed_dim)
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z, context={"u": regimes})
