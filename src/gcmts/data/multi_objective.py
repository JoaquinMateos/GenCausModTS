"""Multi-objective MDP generator for decision-making benchmarks (III.B)."""

from __future__ import annotations

from typing import Any

import torch

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["MultiObjectiveMDPGenerator"]


class MultiObjectiveMDPGenerator(BaseDataGenerator):
    """Linear-Gaussian MDP with vector rewards and logged actions.

    Args:
        state_dim: state dimension.
        action_dim: action dimension.
        horizon: trajectory length.
        n_objectives: number of reward objectives.
        noise_std: process/observation-noise std.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        state_dim: int = 4,
        action_dim: int = 2,
        horizon: int = 12,
        n_objectives: int = 2,
        noise_std: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=state_dim, horizon=horizon)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.dynamics = torch.randn(state_dim, state_dim, generator=self.rng) * 0.3
        self.control = torch.randn(state_dim, action_dim, generator=self.rng)
        self.reward_weights = torch.randn(n_objectives, state_dim, generator=self.rng)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        states = torch.zeros(n_samples, horizon, self.state_dim)
        actions = torch.zeros(n_samples, horizon, self.action_dim)
        rewards = torch.zeros(n_samples, horizon, self.n_objectives)
        state = torch.randn(n_samples, self.state_dim, generator=self.rng)
        for t in range(horizon):
            action = torch.tanh(0.5 * torch.randn(n_samples, self.action_dim, generator=self.rng))
            action = action + 0.3 * torch.randn(n_samples, self.action_dim, generator=self.rng)
            state = torch.tanh(state @ self.dynamics.T + action @ self.control.T)
            state = state + self.noise_std * torch.randn(
                n_samples, self.state_dim, generator=self.rng
            )
            states[:, t] = state
            actions[:, t] = action
            rewards[:, t] = (
                torch.einsum("so,bo->bs", self.reward_weights, state) + 0.1 * action[:, 0:1]
            )
        return Batch(
            x=states,
            context={"action": actions, "reward": rewards},
        )
