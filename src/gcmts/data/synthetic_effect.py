r"""Synthetic benchmarks for counterfactual / treatment-effect estimation.

The reference benchmark is a bank of damped harmonic oscillators, whose
closed-form dynamics admit \emph{exact} counterfactual trajectories when the
exogenous noise and initial state are held fixed (Section~\ref{sec:datasets}).
This is what makes Level-3 (counterfactual) metrics such as PEHE well defined.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["HarmonicOscillatorGenerator"]


class HarmonicOscillatorGenerator(BaseDataGenerator):
    r"""Bank of damped harmonic oscillators with a scalar forcing treatment.

    Each of the ``n_units`` independent units has position and velocity
    :math:`(x_i, v_i)` evolving under

    .. math::
        \dot x_i = v_i, \qquad
        \dot v_i = -\frac{k_i}{m_i} x_i - \frac{c_i}{m_i} v_i + \frac{a_t}{m_i},

    discretised semi-implicitly with step ``dt`` and additive process noise.
    Observations are a fixed linear mixing ``x = W s + \eta``; the exogenous
    noise ``(s_0, \epsilon, \eta)`` is shared between the factual and the
    counterfactual world, so the counterfactual trajectory is exact.

    Treatment is a binary, constant-over-time force ``a \in \{0, 1\}``; the
    benchmark therefore provides both potential outcomes ``Y(0), Y(1)`` for every
    trajectory.
    """

    def __init__(
        self,
        *,
        n_units: int = 3,
        observed_dim: int = 6,
        horizon: int = 24,
        dt: float = 0.1,
        process_noise: float = 0.02,
        observation_noise: float = 0.02,
        outcome_dim: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.n_units = n_units
        self.latent_dim = 2 * n_units
        self.dt = dt
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.outcome_dim = outcome_dim
        self.rng = torch.Generator().manual_seed(seed or 0)

        # Per-unit physical parameters (mass, damping, stiffness), kept stable.
        self.mass = 1.0 + 0.5 * torch.rand(n_units, generator=self.rng)
        self.damping = 0.4 + 0.6 * torch.rand(n_units, generator=self.rng)
        self.stiffness = 1.0 + 1.0 * torch.rand(n_units, generator=self.rng)
        self.mixing = torch.randn(observed_dim, self.latent_dim, generator=self.rng)

    def _step(
        self,
        state: Tensor,
        action: Tensor,
        noise: Tensor,
        stiffness: Tensor,
    ) -> Tensor:
        """One semi-implicit Euler step.

        Args:
            state: ``(B, 2n)`` interleaved ``[x_1, v_1, ..., x_n, v_n]``.
            action: ``(B,)`` scalar forcing (already scaled by the sample gain).
            noise: ``(B, 2n)`` process noise for this step.
            stiffness: ``(B, n)`` per-sample per-unit stiffness.
        """
        pos = state[:, 0::2]
        vel = state[:, 1::2]
        force = -stiffness * pos - self.damping * vel + action.unsqueeze(-1)
        new_vel = vel + self.dt * force / self.mass
        new_pos = pos + self.dt * new_vel
        stepped = torch.stack([new_pos, new_vel], dim=-1).reshape(state.shape)
        return stepped + noise

    def _roll(
        self,
        initial: Tensor,
        actions: Tensor,
        process_noise: Tensor,
        observation_noise: Tensor,
        stiffness: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Roll a trajectory; return latent states ``(B, T, 2n)`` and observations."""
        batch, horizon = actions.shape
        states = [initial]
        state = initial
        for t in range(horizon - 1):
            state = self._step(state, actions[:, t], process_noise[:, t], stiffness)
            states.append(state)
        latent = torch.stack(states, dim=1)
        observed = latent.reshape(-1, self.latent_dim) @ self.mixing.T
        observed = observed.reshape(batch, horizon, self.observed_dim)
        observed = observed + observation_noise
        return latent, observed

    def _outcome(self, observed: Tensor) -> Tensor:
        return observed[:, -1, self.outcome_dim]

    def potential_outcomes(self, n_samples: int) -> dict[str, Tensor]:
        """Return both potential outcomes and trajectories for ``a=0`` and ``a=1``."""
        horizon = self.horizon
        initial = torch.randn(n_samples, self.latent_dim, generator=self.rng)
        process_noise = self.process_noise * torch.randn(
            n_samples, horizon, self.latent_dim, generator=self.rng
        )
        observation_noise = self.observation_noise * torch.randn(
            n_samples, horizon, self.observed_dim, generator=self.rng
        )
        # Per-sample stiffness and forcing gain make the treatment effect
        # heterogeneous; both are identifiable from the factual trajectory.
        stiffness = (
            self.stiffness * (1.0 + 0.4 * torch.randn(n_samples, self.n_units, generator=self.rng))
        ).clamp(min=0.3)
        gain = 0.5 + torch.rand(n_samples, 1, generator=self.rng)
        zeros = torch.zeros(n_samples, horizon)
        ones = gain.expand(n_samples, horizon)
        latent0, observed0 = self._roll(
            initial, zeros, process_noise, observation_noise, stiffness
        )
        latent1, observed1 = self._roll(
            initial, ones, process_noise, observation_noise, stiffness
        )
        return {
            "y0": self._outcome(observed0),
            "y1": self._outcome(observed1),
            "observed0": observed0,
            "observed1": observed1,
            "latent0": latent0,
            "latent1": latent1,
        }

    def sample(
        self,
        n_samples: int,
        horizon: int | None = None,
        *,
        factual_action: Tensor | None = None,
        **kwargs: Any,
    ) -> Batch:
        r"""Return a factual batch and the exact counterfactual with the treatment off.

        The factual treatment switches on at a random time ``t_0 ~ U{0,..,T-1}``
        and stays on, so different trajectories exhibit different treatment
        durations; this within-trajectory variation is what makes the effect
        identifiable from factual data alone. The counterfactual keeps the same
        exogenous noise but sets the treatment off throughout.
        """
        horizon = horizon or self.horizon
        initial = torch.randn(n_samples, self.latent_dim, generator=self.rng)
        process_noise = self.process_noise * torch.randn(
            n_samples, horizon, self.latent_dim, generator=self.rng
        )
        observation_noise = self.observation_noise * torch.randn(
            n_samples, horizon, self.observed_dim, generator=self.rng
        )
        stiffness = (
            self.stiffness
            * (1.0 + 0.4 * torch.randn(n_samples, self.n_units, generator=self.rng))
        ).clamp(min=0.3)
        gain = 0.5 + torch.rand(n_samples, 1, generator=self.rng)

        if factual_action is None:
            switch = torch.randint(0, horizon, (n_samples,), generator=self.rng)
            times = torch.arange(horizon).unsqueeze(0)
            action = (times >= switch.unsqueeze(1)).float() * gain
        else:
            action = factual_action.float()
            if action.ndim == 2:
                action = action.unsqueeze(1).expand(-1, horizon, -1)[:, :, 0]
        counter_action = torch.zeros_like(action)

        latent_f, observed_f = self._roll(
            initial, action, process_noise, observation_noise, stiffness
        )
        latent_c, observed_c = self._roll(
            initial, counter_action, process_noise, observation_noise, stiffness
        )
        return Batch(
            x=observed_f,
            z=latent_f,
            context={"A": action.unsqueeze(-1)},
            counterfactual=observed_c,
            outcome=self._outcome(observed_f),
        )
