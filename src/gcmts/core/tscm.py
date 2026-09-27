"""Temporal Structural Causal Model (TSCM) and Latent TSCM formalism."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__ = ["TSCM", "LTSCM"]

NoiseSampler = Callable[[int, int, int], Tensor]
"""Signature: ``(n_samples, horizon, dim) -> noise tensor``."""


@dataclass
class TSCM:
    r"""Discrete-time temporal structural causal model.

    Implements the tuple ``M = <X, N, F, P_ε>`` from the review:

    .. math::
        X_{i,t} = f_i( Pa(X_{i,t}), ε_{i,t} ),
        \quad Pa(X_{i,t}) \subseteq \{X_{j,t-\tau}\}_{\tau=1}^{p} \cup \{X_{j,t}\}_{j\neq i}

    The contemporaneous subgraph must be a DAG.
    """

    observed_dim: int
    max_lag: int
    mechanisms: list[Callable[[Tensor, Tensor], Tensor]] = field(default_factory=list)
    noise_sampler: NoiseSampler | None = None
    adjacency: Tensor | None = None

    def __post_init__(self) -> None:
        if len(self.mechanisms) != self.observed_dim:
            raise ValueError(
                "Number of mechanisms must equal observed_dim."
            )

    def sample(
        self,
        n_samples: int,
        horizon: int,
        *,
        initial: Tensor | None = None,
        intervention: dict[tuple[int, int], Callable[[Tensor], Tensor]] | None = None,
    ) -> Tensor:
        """Generate ``(n_samples, horizon, observed_dim)`` observations.

        Args:
            n_samples: number of trajectories
            horizon: length of each trajectory
            initial: optional warm-start ``(n_samples, max_lag, observed_dim)``
            intervention: mapping ``(time, variable) -> callable`` that replaces
                the mechanism for that node/time with the supplied function.
        """
        device = "cpu"
        x = torch.zeros(n_samples, horizon + self.max_lag, self.observed_dim)
        if initial is not None:
            x[:, : self.max_lag] = initial
        else:
            x[:, : self.max_lag] = torch.randn(n_samples, self.max_lag, self.observed_dim)

        for t in range(self.max_lag, horizon + self.max_lag):
            parents = x[:, t - self.max_lag : t]
            for i, f in enumerate(self.mechanisms):
                if intervention and (t - self.max_lag, i) in intervention:
                    x[:, t, i] = intervention[(t - self.max_lag, i)](parents)
                else:
                    eps = self._noise(n_samples, 1, device)
                    x[:, t, i] = f(parents, eps.squeeze(1))
        return x[:, self.max_lag :]

    def _noise(self, n: int, horizon: int, device: str) -> Tensor:
        if self.noise_sampler is not None:
            return self.noise_sampler(n, horizon, self.observed_dim)
        return torch.randn(n, horizon, self.observed_dim, device=device)


@dataclass
class LTSCM:
    """Latent temporal structural causal model.

    .. math::
        z_{i,t} = f_i( Pa(z_{i,t}), u_t, ε_{i,t} )
        x_t     = g(z_t, η_t)

    The latent process is first sampled, then mixed through ``decoder`` to
    produce observations.
    """

    latent_dim: int
    observed_dim: int
    max_lag: int
    mechanisms: list[Callable[[Tensor, Tensor, Tensor | None], Tensor]]
    decoder: Callable[[Tensor, Tensor], Tensor]
    noise_sampler: NoiseSampler | None = None
    context_sampler: Callable[[int, int], Tensor] | None = None
    adjacency: Tensor | None = None

    def sample(
        self,
        n_samples: int,
        horizon: int,
        *,
        initial: Tensor | None = None,
        return_latents: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Generate observations and optionally latents."""
        device = "cpu"
        z = torch.zeros(n_samples, horizon + self.max_lag, self.latent_dim)
        if initial is not None:
            z[:, : self.max_lag] = initial
        else:
            z[:, : self.max_lag] = torch.randn(n_samples, self.max_lag, self.latent_dim)

        for t in range(self.max_lag, horizon + self.max_lag):
            parents = z[:, t - self.max_lag : t]
            context = (
                self.context_sampler(n_samples, 1) if self.context_sampler else None
            )
            for i, f in enumerate(self.mechanisms):
                eps = self._noise(n_samples, 1, device).squeeze(1)
                z[:, t, i] = f(parents, eps, context)

        z_obs = z[:, self.max_lag :]
        eta = torch.randn(n_samples, horizon, self.observed_dim, device=device)
        x = self.decoder(z_obs, eta)
        if return_latents:
            return x, z_obs
        return x

    def _noise(self, n: int, horizon: int, device: str) -> Tensor:
        if self.noise_sampler is not None:
            return self.noise_sampler(n, horizon, self.latent_dim)
        return torch.randn(n, horizon, self.latent_dim, device=device)
