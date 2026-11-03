"""Latent structural SCM generator for the CaDRe branch (I.E).

Reference: Fu et al., *Learning General Causal Structures with Hidden Dynamic
Process for Climate Analysis*, arXiv:2501.12500 (ICML 2026) — BibTeX
``fuLearningGeneralCausal2026``.

This module is an independent reimplementation from the paper's equations
(Eq. 1 and Appendix C.1.1, Eqs. 51--54), **not** a port of the official
``MinghaoFu/CaDRe`` code (MIT).  The simulated system is

.. math::

    z_t &= A_{\\mathrm{inst}} z_t + A_{\\mathrm{lag}} z_{t-1}
           + b_{u} + \\epsilon_t, \\qquad \\epsilon_t \\sim \\mathcal N(0, \\sigma_u^2 I), \\\\
    e_t &= f(z_t), \\qquad x_t = (I - B)^{-1} e_t,

with a strictly lower-triangular instantaneous latent matrix
:math:`A_{\\mathrm{inst}}` (a DAG in a known causal order), a sparse lagged
matrix :math:`A_{\\mathrm{lag}}` rescaled to spectral radius at most ``0.85``,
a strictly lower-triangular observed matrix :math:`B` (the observed DAG), and
regime-specific bias/noise indexed by the one-hot context ``u``.

Simplifications relative to the paper:
* one latent lag (first-order Markov), not the general ``L``-lag process;
* the latent mechanism is linear in ``z`` (the paper applies a leaky ReLU);
* regime effects enter through a bias and a noise scale rather than through
  fully regime-specific transition matrices;
* the observed mixing ``f`` is either linear or ``tanh`` elementwise, and the
  causal injection uses a fixed sparse lower-triangular ``B`` instead of the
  paper's time-varying cosine weights.

The generator is deterministic given ``seed``: two instances constructed with
the same seed and hyper-parameters produce identical structures and samples.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["LatentStructuralGenerator"]


class LatentStructuralGenerator(BaseDataGenerator):
    r"""Synthetic latent SCM used by the CaDRe experiments.

    Args:
        latent_dim: number ``d`` of latent causal factors (default 3).
        observed_dim: number ``D`` of observed variables (default 6).
        horizon: trajectory length ``T`` returned by :meth:`sample` (default 16).
        n_regimes: number ``R`` of one-hot regimes ``u`` (default 4).
        mixing: ``"linear"`` or ``"tanh"`` observation mixing.
        noise_std: base latent innovation scale.
        instant_sparsity: edge probability of ``A_inst`` (strict lower triangle).
        lag_sparsity: edge probability of the off-diagonal part of ``A_lag``.
        observed_sparsity: edge probability of ``B`` (strict lower triangle).
        max_radius: spectral-radius cap for ``A_lag`` (default 0.85).
        burn_in: discarded warm-up steps used to approach the stationary law.
        init_std: standard deviation of the initial latent state.
        seed: seed for every random draw (structures and trajectories).
    """

    def __init__(
        self,
        *,
        latent_dim: int = 3,
        observed_dim: int = 6,
        horizon: int = 16,
        n_regimes: int = 4,
        mixing: str = "linear",
        noise_std: float = 0.2,
        instant_sparsity: float = 0.5,
        lag_sparsity: float = 0.4,
        observed_sparsity: float = 0.35,
        max_radius: float = 0.85,
        burn_in: int = 16,
        init_std: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        if latent_dim < 1 or latent_dim > observed_dim:
            raise ValueError("Require 1 <= latent_dim <= observed_dim.")
        if mixing not in {"linear", "tanh"}:
            raise ValueError("mixing must be 'linear' or 'tanh'.")
        if not 0.0 < max_radius < 1.0:
            raise ValueError("max_radius must lie in (0, 1).")
        if n_regimes < 1:
            raise ValueError("n_regimes must be positive.")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive.")
        if init_std < 0:
            raise ValueError("init_std must be non-negative.")
        if not 0.0 <= instant_sparsity <= 1.0:
            raise ValueError("instant_sparsity must lie in [0, 1].")
        if not 0.0 <= lag_sparsity <= 1.0:
            raise ValueError("lag_sparsity must lie in [0, 1].")
        if not 0.0 <= observed_sparsity <= 1.0:
            raise ValueError("observed_sparsity must lie in [0, 1].")

        self.latent_dim = latent_dim
        self.n_regimes = n_regimes
        self.mixing_type = mixing
        self.noise_std = noise_std
        self.max_radius = max_radius
        self.burn_in = max(int(burn_in), 0)
        self.init_std = init_std
        self.rng = torch.Generator().manual_seed(seed)

        self.latent_instantaneous_adjacency = self._sample_instantaneous(instant_sparsity)
        self.latent_lag_adjacency = self._sample_lagged(lag_sparsity, max_radius)
        self.observed_adjacency = self._sample_observed(observed_sparsity)
        self.spectral_radius = float(torch.linalg.eigvals(self.latent_lag_adjacency).abs().max())

        # Regime-dependent intercept and innovation scale.  The context vector
        # returned in ``Batch.context["u"]`` is the one-hot encoding of u.
        self.regime_bias = 0.3 * torch.randn(n_regimes, latent_dim, generator=self.rng)
        self.regime_noise = 0.5 + torch.rand(n_regimes, generator=self.rng)

        # Observation mixing e_t = f(z_t).
        self.mixing_weight = torch.randn(observed_dim, latent_dim, generator=self.rng) / math.sqrt(
            latent_dim
        )
        self.mixing_bias = 0.1 * torch.randn(observed_dim, generator=self.rng)

    # ------------------------------------------------------------------
    # structure sampling
    # ------------------------------------------------------------------
    def _sample_instantaneous(self, sparsity: float) -> Tensor:
        """Sparse strictly lower-triangular ``A_inst`` (a DAG)."""
        d = self.latent_dim
        strict_lower = torch.tril(torch.ones(d, d), diagonal=-1)
        mask = (torch.rand(d, d, generator=self.rng) < sparsity).float() * strict_lower
        weights = 0.2 + 0.4 * torch.rand(d, d, generator=self.rng)
        adjacency = weights * mask
        if d > 1 and float(adjacency.abs().sum()) == 0.0:
            adjacency[1, 0] = 0.4
        return adjacency

    def _sample_lagged(self, sparsity: float, max_radius: float) -> Tensor:
        """Sparse ``A_lag`` with persistence, capped at ``max_radius``."""
        d = self.latent_dim
        mask = (torch.rand(d, d, generator=self.rng) < sparsity).float()
        mask.fill_diagonal_(1.0)
        off_mask = mask * (1.0 - torch.eye(d))
        weights = 0.1 + 0.25 * torch.rand(d, d, generator=self.rng) * off_mask
        diagonal = 0.35 + 0.35 * torch.rand(d, generator=self.rng)
        adjacency = weights + torch.diag(diagonal)
        if d > 1 and float(off_mask.sum()) == 0.0:
            adjacency[1, 0] = 0.2
        radius = float(torch.linalg.eigvals(adjacency).abs().max())
        if radius > max_radius:
            adjacency = adjacency * (max_radius / radius)
        # The instantaneous triangular solve also amplifies persistence.  Keep
        # the effective transition (I - A_inst)^-1 A_lag contractive as well.
        effective = torch.linalg.solve(
            torch.eye(d) - self.latent_instantaneous_adjacency, adjacency
        )
        effective_radius = float(torch.linalg.eigvals(effective).abs().max())
        if effective_radius > 0.95:
            adjacency = adjacency * (0.95 / effective_radius)
        return adjacency

    def _sample_observed(self, sparsity: float) -> Tensor:
        """Sparse strictly lower-triangular observed DAG ``B``."""
        d = self.observed_dim
        strict_lower = torch.tril(torch.ones(d, d), diagonal=-1)
        mask = (torch.rand(d, d, generator=self.rng) < sparsity).float() * strict_lower
        weights = 0.1 + 0.3 * torch.rand(d, d, generator=self.rng)
        adjacency = weights * mask
        if d > 1 and float(adjacency.abs().sum()) == 0.0:
            adjacency[1, 0] = 0.2
        return adjacency

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _mix(self, z: Tensor) -> Tensor:
        e = z @ self.mixing_weight.T + self.mixing_bias
        if self.mixing_type == "tanh":
            return torch.tanh(e)
        return e

    def _observed_from_sources(self, e: Tensor) -> Tensor:
        """Solve ``x = (I - B)^{-1} e`` by forward substitution."""
        x = e.clone()
        for i in range(self.observed_dim):
            if i == 0:
                continue
            x[:, i] = e[:, i] + x[:, :i] @ self.observed_adjacency[i, :i]
        return x

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        """Draw ``n_samples`` trajectories of length ``horizon``.

        Returns a :class:`~gcmts.typing.Batch` with ``x`` of shape
        ``(n_samples, T, D)``, ``z`` of shape ``(n_samples, T, d)``, the lagged
        latent graph in ``adjacency`` (``(d, d, 1)``), and the instantaneous
        and observed graphs in
        ``context["instantaneous_adjacency"]`` / ``context["observed_adjacency"]``.
        """
        horizon = self.horizon if horizon is None else int(horizon)
        if n_samples < 1 or horizon < 1:
            raise ValueError("n_samples and horizon must be positive.")
        d = self.latent_dim
        total = horizon + self.burn_in
        regime = torch.randint(0, self.n_regimes, (n_samples,), generator=self.rng)
        bias = self.regime_bias[regime]  # (N, d)
        noise_scale = (self.noise_std * self.regime_noise[regime]).unsqueeze(-1)

        z = torch.zeros(n_samples, total, d)
        z[:, 0] = self.init_std * torch.randn(n_samples, d, generator=self.rng)
        # A_inst is strictly lower triangular, so the instantaneous system is
        # solved exactly by forward substitution in the causal order.
        instant = self.latent_instantaneous_adjacency
        lag = self.latent_lag_adjacency
        for t in range(1, total):
            eps = torch.randn(n_samples, d, generator=self.rng)
            noise = eps * noise_scale
            state = bias + z[:, t - 1] @ lag.T + noise
            for i in range(1, d):
                state[:, i] = (
                    bias[:, i] + z[:, t - 1] @ lag[i] + noise[:, i] + state[:, :i] @ instant[i, :i]
                )
            z[:, t] = state

        z_obs = z[:, self.burn_in :]
        e = self._mix(z_obs.reshape(-1, d))
        x = self._observed_from_sources(e).reshape(n_samples, horizon, self.observed_dim)
        context = {
            "u": torch.nn.functional.one_hot(regime, self.n_regimes).float(),
            "instantaneous_adjacency": self.latent_instantaneous_adjacency.clone(),
            "observed_adjacency": self.observed_adjacency.clone(),
        }
        return Batch(
            x=x,
            z=z_obs,
            context=context,
            adjacency=self.latent_lag_adjacency.unsqueeze(-1).clone(),
            instantaneous_adjacency=self.latent_instantaneous_adjacency.clone(),
        )
