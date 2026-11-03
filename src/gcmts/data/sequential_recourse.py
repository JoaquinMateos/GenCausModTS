"""Sequential recourse environment for SCARF (decision-making, III.B).

Compact implementation of the synthetic process of Gumucio & Zhang (2026),
*Algorithmic recourse in sequential decision-making for long-term fairness*,
Frontiers in Big Data 9:1750906 (DOI 10.3389/fdata.2026.1750906).

The state is ``x_t = (S, X_I, X_M)`` with ``d = 6``:
coordinate 0 is the binary, immutable sensitive attribute ``S``; coordinates
1--3 are directly actionable (``X_I``); coordinates 4--5 are mediated
(``X_M``). A group-blind classifier ``y_t = 1[beta'x_t + beta_0 >= 0]``
excludes ``S`` from its input. The factual dynamics are

    x_{t+1} = A x_t + c y_t e_I + b_S S + eps_t,

with the exogenous ``eps_t`` stored so counterfactuals can be replayed
exactly. An intervention is ``x_I += delta`` and ``x_M += W delta`` while
``S`` is left unchanged and ``|delta|_inf <= step_cap``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["SequentialRecourseGenerator"]


class SequentialRecourseGenerator(BaseDataGenerator):
    """Linear-Gaussian sequential recourse environment.

    Args:
        state_dim: state dimension ``d`` (default 6).
        horizon: trajectory length ``T`` (default 8).
        action_dim: number of directly actionable coordinates (default 3).
        sensitive_prob: probability of ``S = 1``.
        noise_std: process-noise standard deviation.
        step_cap: maximum absolute intervention per coordinate and step.
        budget: default cumulative ``L1`` intervention budget per individual.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        state_dim: int = 6,
        horizon: int = 8,
        action_dim: int = 3,
        sensitive_prob: float = 0.5,
        noise_std: float = 0.1,
        step_cap: float = 0.3,
        budget: float = 2.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=state_dim, horizon=horizon)
        if not 0 < action_dim < state_dim - 1:
            raise ValueError("action_dim must leave at least one mediated coordinate.")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sensitive_prob = sensitive_prob
        self.noise_std = noise_std
        self.step_cap = step_cap
        self.budget = budget
        self.rng = torch.Generator().manual_seed(seed or 0)

        self.actionable_indices = torch.arange(1, action_dim + 1, dtype=torch.long)
        self.mediated_indices = torch.arange(action_dim + 1, state_dim, dtype=torch.long)
        actionable_mask = torch.zeros(state_dim, dtype=torch.bool)
        actionable_mask[self.actionable_indices] = True
        self.actionable_mask = actionable_mask

        # Stable linear recurrence; the sensitive row is frozen so S is immutable.
        dynamics = 0.18 * torch.randn(state_dim, state_dim, generator=self.rng)
        radius = float(torch.linalg.eigvals(dynamics).abs().max())
        if radius > 0.7:
            dynamics = dynamics * (0.7 / radius)
        dynamics[0, :] = 0.0
        self.dynamics = dynamics
        self.control = 0.35
        self.intervention_effect = actionable_mask.float()
        group_shift = torch.zeros(state_dim)
        group_shift[1:] = -0.15 + 0.05 * torch.randn(state_dim - 1, generator=self.rng)
        self.group_shift = group_shift

        # Group-blind decision rule: beta[0] = 0, so S is never used.
        beta = 0.6 * torch.randn(state_dim, generator=self.rng)
        beta[0] = 0.0
        self.beta = beta
        self.beta0 = -0.1

        # Initial law conditional on S (group 1 is mildly disadvantaged).
        self.init_mean = 0.2 * torch.randn(state_dim - 1, generator=self.rng)
        self.init_shift = -0.35 * torch.ones(state_dim - 1)
        self.init_std = 0.5

        # Additive mediated propagation: x_M += W delta with S and I unchanged.
        propagation = torch.zeros(state_dim, state_dim)
        propagation[self.actionable_indices, self.actionable_indices] = 1.0
        propagation[self.mediated_indices.unsqueeze(1), self.actionable_indices] = (
            0.15 * torch.randn(len(self.mediated_indices), action_dim, generator=self.rng)
        )
        self.propagation = propagation
        self.cost_weights = actionable_mask.float()

    def decision(self, x: Tensor) -> Tensor:
        """Group-blind decision ``1[beta'x + beta_0 >= 0]``."""
        return (x @ self.beta + self.beta0 >= 0).to(x.dtype)

    def _initial_state(self, sensitive: Tensor) -> Tensor:
        noise = torch.randn(sensitive.shape[0], self.state_dim - 1, generator=self.rng)
        features = self.init_mean + sensitive.unsqueeze(-1) * self.init_shift
        features = features + self.init_std * noise
        return torch.cat([sensitive.unsqueeze(-1), features], dim=-1)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        """Draw factual trajectories and store every exogenous innovation."""
        horizon = horizon or self.horizon
        device = self.dynamics.device
        sensitive = (torch.rand(n_samples, generator=self.rng) < self.sensitive_prob).to(
            device=device, dtype=torch.float32
        )
        x = torch.empty(n_samples, horizon, self.state_dim, device=device)
        noise = torch.zeros(n_samples, horizon, self.state_dim, device=device)
        decisions = torch.zeros(n_samples, horizon, device=device)
        x[:, 0] = self._initial_state(sensitive)

        for t in range(horizon - 1):
            y = self.decision(x[:, t])
            eps = self.noise_std * torch.randn(n_samples, self.state_dim, generator=self.rng)
            eps[:, 0] = 0.0  # S is immutable
            innovation = (
                x[:, t] @ self.dynamics.T
                + self.control * y.unsqueeze(-1) * self.intervention_effect
                + sensitive.unsqueeze(-1) * self.group_shift
                + eps
            )
            innovation[:, 0] = sensitive  # keep S exactly fixed
            decisions[:, t] = y
            noise[:, t] = eps
            x[:, t + 1] = innovation
        decisions[:, -1] = self.decision(x[:, -1])

        context = {
            "sensitive": sensitive,
            "decision": decisions,
            "actionable_mask": self.actionable_mask.clone(),
            "propagation": self.propagation.clone(),
            "budget": torch.as_tensor(self.budget, device=device),
            "step_cap": torch.as_tensor(self.step_cap, device=device),
            "noise": noise,
            "cost_weights": self.cost_weights.clone(),
        }
        return Batch(x=x, context=context)

    def _full_delta(
        self,
        delta: Tensor,
        batch_size: int,
        horizon: int,
        actionable_mask: Tensor | None = None,
        step_cap: Tensor | None = None,
    ) -> Tensor:
        """Normalise an intervention to ``(B, T, d)`` with non-actionables zero."""
        delta = torch.as_tensor(delta, dtype=self.propagation.dtype, device=self.propagation.device)
        if delta.ndim == 2:
            delta = delta.unsqueeze(1).expand(-1, horizon, -1)
        if delta.ndim != 3:
            raise ValueError("delta must have shape (B, d), (B, T, d) or (B, T, |I|).")
        if delta.shape[0] != batch_size:
            raise ValueError("delta batch size does not match the trajectory batch.")
        if delta.shape[-1] == self.action_dim:
            full = torch.zeros(
                batch_size, horizon, self.state_dim, dtype=delta.dtype, device=delta.device
            )
            index = self.actionable_indices.to(delta.device).view(1, 1, -1)
            delta = full.scatter(-1, index.expand(batch_size, horizon, -1), delta)
        elif delta.shape[-1] != self.state_dim:
            raise ValueError("delta's last dimension must be d or the actionable dimension.")
        if delta.shape[1] != horizon:
            raise ValueError("delta horizon does not match the trajectory horizon.")
        mask = self.actionable_mask if actionable_mask is None else actionable_mask
        delta = delta * torch.as_tensor(mask, dtype=delta.dtype, device=delta.device).reshape(-1)
        cap = self.step_cap if step_cap is None else step_cap
        cap = torch.as_tensor(cap, dtype=delta.dtype, device=delta.device).reshape(-1).mean()
        return torch.clamp(delta, -cap, cap)

    def _project_budget(
        self, delta: Tensor, budget: Tensor, cost_weights: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Scale each intervention so its weighted L1 cost respects ``budget``."""
        cost = (delta.abs() * cost_weights).sum(dim=(1, 2))
        budget = torch.as_tensor(budget, dtype=delta.dtype, device=delta.device).reshape(-1)
        if budget.numel() == 1:
            budget = budget.expand(delta.shape[0])
        scale = torch.clamp(budget / cost.clamp_min(1e-8), min=0.0, max=1.0)
        return delta * scale[:, None, None], cost * scale

    def counterfactual(
        self,
        batch: Batch,
        delta: Tensor,
        *,
        project_budget: bool = True,
    ) -> Batch:
        """Replay the sequence under ``do(x_I += delta)`` with the stored noise.

        The returned :class:`Batch` contains the counterfactual trajectory in
        ``x`` and the counterfactual decisions under ``context['decision']``.
        With ``delta=0`` and ``project_budget=False`` the result equals the
        factual trajectory up to floating-point round-off.
        """
        if batch.context is None or "noise" not in batch.context:
            raise ValueError("The batch must carry the generator's stored 'noise' context.")
        context = batch.context
        noise = context["noise"]
        batch_size, horizon, _ = batch.x.shape
        sensitive = torch.as_tensor(
            context.get("sensitive", batch.x[:, 0, 0]),
            dtype=batch.x.dtype,
            device=batch.x.device,
        ).reshape(batch_size)
        delta = self._full_delta(
            delta,
            batch_size,
            horizon,
            actionable_mask=context.get("actionable_mask", self.actionable_mask),
            step_cap=context.get("step_cap", torch.as_tensor(self.step_cap)),
        )
        if project_budget:
            budget = context.get("budget", torch.as_tensor(self.budget))
            weights = context.get("cost_weights", self.cost_weights)
            delta, _ = self._project_budget(delta, budget, weights)
        propagation = context.get("propagation", self.propagation).to(delta.device, delta.dtype)

        counterfactual = torch.empty_like(batch.x)
        decisions = torch.zeros(batch_size, horizon, device=batch.x.device)
        current = batch.x[:, 0]
        for t in range(horizon):
            intervened = current + delta[:, t] @ propagation.T
            intervened = torch.cat([sensitive.unsqueeze(-1), intervened[:, 1:]], dim=-1)
            y = self.decision(intervened)
            counterfactual[:, t] = intervened
            decisions[:, t] = y
            if t < horizon - 1:
                current = (
                    intervened @ self.dynamics.T
                    + self.control * y.unsqueeze(-1) * self.intervention_effect
                    + sensitive.unsqueeze(-1) * self.group_shift
                    + noise[:, t]
                )
                current = torch.cat([sensitive.unsqueeze(-1), current[:, 1:]], dim=-1)
        new_context = dict(context)
        new_context["decision"] = decisions
        new_context["intervention"] = delta
        return Batch(x=counterfactual, context=new_context)
