# Adapted-source provenance:
# Paper: Gumucio and Zhang, Algorithmic Recourse in Sequential Decision Making, 2026.
# Official implementation: none used; this is a paper-based adaptation.
# Status: compact surrogate.

"""SCARF: sequential causal algorithmic recourse for long-term fairness.

Reference: Gumucio & Zhang, *Algorithmic recourse in sequential decision-making
for long-term fairness*, Frontiers in Big Data 9:1750906 (2026),
DOI 10.3389/fdata.2026.1750906 (BibTeX key
``gumucioAlgorithmicRecourseSequential2026``).

This is a compact, faithful re-implementation of the paper's four components:
a group-blind MLP classifier, a GRU temporal-dynamics model, a graph-masked
linear propagation module for counterfactual features, and a GRU intervention
generator with hard global-budget projection. The objective combines the
improvement loss, a demographic-parity (DP) gap, a budget hinge, and
transition/propagation fit. The paper's long-term Sinkhorn transport term is
optional (``long_term_weight=0`` by default); the DP gap is the short-term
fairness term used in the compact version. The implementation also replaces
the paper's RCGAN/VACA adversarial modules with the lighter GRU/linear
propagation surrogates, which is why no Sinkhorn/optimal-transport discriminator
is needed while training.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gcmts.core.backbones import MLP
from gcmts.decision_making.base import BaseDecisionMaker
from gcmts.typing import Batch

__all__ = ["SCARF"]


class SCARF(BaseDecisionMaker):
    """Sequential recourse generator for long-term fairness.

    Args:
        state_dim: dimension ``d`` of the profile state.
        action_dim: number of directly actionable coordinates ``|I|``.
        hidden_dim: width of the GRU and MLP modules.
        horizon: default intervention horizon.
        step_cap: maximum absolute intervention per coordinate and step.
        budget: default cumulative weighted-L1 budget.
        cost_weights: per-coordinate intervention costs (defaults to 1 on I).
        long_term_weight: optional weight of the Sinkhorn long-term term.
        sinkhorn_eps: entropic regularisation for the optional Sinkhorn term.
        sinkhorn_iters: number of Sinkhorn iterations.
        loss_weights: overrides for the individual loss weights.
    """

    def __init__(
        self,
        *,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 64,
        horizon: int = 8,
        step_cap: float = 0.3,
        budget: float = 2.0,
        cost_weights: Tensor | None = None,
        long_term_weight: float = 0.0,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 20,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        if not 0 < action_dim < state_dim - 1:
            raise ValueError("action_dim must leave at least one mediated coordinate.")
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.step_cap = step_cap
        self.budget = budget
        self.long_term_weight = long_term_weight
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters

        self.actionable_indices = torch.arange(1, action_dim + 1, dtype=torch.long)
        self.mediated_indices = torch.arange(action_dim + 1, state_dim, dtype=torch.long)
        self.register_buffer("_action_index", self.actionable_indices)
        self._action_index: Tensor
        mask = torch.zeros(state_dim, dtype=torch.bool)
        mask[self.actionable_indices] = True
        self.register_buffer("actionable_mask", mask)
        self.actionable_mask: Tensor
        self.register_buffer(
            "cost_weights",
            torch.ones(state_dim) if cost_weights is None else cost_weights.float(),
        )
        self.cost_weights: Tensor

        # Graph-masked propagation: identity on I and learned I -> M edges.
        propagation_mask = torch.zeros(state_dim, state_dim)
        propagation_mask[self.mediated_indices.unsqueeze(1), self.actionable_indices] = 1.0
        self.register_buffer("propagation_mask", propagation_mask)
        self.propagation_mask: Tensor
        direct = torch.zeros(state_dim, state_dim)
        direct[self.actionable_indices, self.actionable_indices] = 1.0
        self.register_buffer("direct_propagation", direct)
        self.direct_propagation: Tensor
        self.propagation_raw = nn.Parameter(torch.zeros(state_dim, state_dim))

        self.classifier = MLP(state_dim - 1, 1, hidden_dims=(hidden_dim, hidden_dim))
        self.transition_gru = nn.GRUCell(state_dim + 2, hidden_dim)
        self.transition_head = nn.Linear(hidden_dim, state_dim)
        self.intervention_gru = nn.GRUCell(state_dim + action_dim + 1, hidden_dim)
        self.intervention_head = nn.Linear(hidden_dim, action_dim)

        self.loss_weights = {
            "improvement": 1.0,
            "fairness": 1.0,
            "budget": 1.0,
            "transition": 1.0,
            "propagation": 1.0,
            "long_term": long_term_weight,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def propagation_matrix(self) -> Tensor:
        """Learned graph-masked propagation ``P`` with ``x_cf = x + P delta``."""
        return self.direct_propagation + self.propagation_raw * self.propagation_mask

    def probability(self, x: Tensor) -> Tensor:
        """Group-blind favourable-decision probability for states ``(..., d)``."""
        logits = self.classifier(x[..., 1:]).squeeze(-1)
        return torch.sigmoid(logits)

    def decision(self, x: Tensor) -> Tensor:
        """Binary decision ``1[p(x) >= 0.5]`` (the classifier never sees S)."""
        return (self.probability(x) >= 0.5).to(x.dtype)

    def _keep_sensitive(self, x: Tensor, sensitive: Tensor) -> Tensor:
        return torch.cat([sensitive.unsqueeze(-1), x[..., 1:]], dim=-1)

    def _transition_step(
        self, x: Tensor, y: Tensor, sensitive: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        inputs = torch.cat([x, y.unsqueeze(-1), sensitive.unsqueeze(-1)], dim=-1)
        hidden = self.transition_gru(inputs, hidden)
        next_state = self.transition_head(hidden)
        return self._keep_sensitive(next_state, sensitive), hidden

    def _factual_next(self, states: Tensor, sensitive: Tensor) -> Tensor:
        batch_size, horizon, _ = states.shape
        hidden = states.new_zeros(batch_size, self.hidden_dim)
        predictions: list[Tensor] = []
        for t in range(horizon - 1):
            y = self.decision(states[:, t])
            next_state, hidden = self._transition_step(states[:, t], y, sensitive, hidden)
            predictions.append(next_state)
        if not predictions:
            return states.new_zeros(batch_size, 0, self.state_dim)
        return torch.stack(predictions, dim=1)

    def _rollout(
        self, x0: Tensor, sensitive: Tensor, deltas: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Roll out counterfactual states using the learned transition model."""
        batch_size, horizon, _ = deltas.shape
        propagation = self.propagation_matrix()
        hidden = x0.new_zeros(batch_size, self.hidden_dim)
        current = x0
        states: list[Tensor] = []
        probabilities: list[Tensor] = []
        decisions: list[Tensor] = []
        for t in range(horizon):
            intervened = current + deltas[:, t] @ propagation.T
            intervened = self._keep_sensitive(intervened, sensitive)
            prob = self.probability(intervened)
            y = (prob >= 0.5).to(intervened.dtype)
            states.append(intervened)
            probabilities.append(prob)
            decisions.append(y)
            if t < horizon - 1:
                current, hidden = self._transition_step(intervened, y, sensitive, hidden)
        return (
            torch.stack(states, dim=1),
            torch.stack(probabilities, dim=1),
            torch.stack(decisions, dim=1),
        )

    def _generate_interventions(
        self, x0: Tensor, sensitive: Tensor, step_cap: float, horizon: int
    ) -> Tensor:
        """Autoregressive GRU generation of ``delta`` on the actionable coordinates."""
        batch_size = x0.shape[0]
        hidden = x0.new_zeros(batch_size, self.hidden_dim)
        previous = x0.new_zeros(batch_size, self.action_dim)
        deltas: list[Tensor] = []
        for _ in range(horizon):
            inputs = torch.cat([x0, previous, sensitive.unsqueeze(-1)], dim=-1)
            hidden = self.intervention_gru(inputs, hidden)
            raw = torch.tanh(self.intervention_head(hidden)) * step_cap
            deltas.append(raw)
            previous = raw
        raw_sequence = torch.stack(deltas, dim=1)
        full = x0.new_zeros(batch_size, horizon, self.state_dim)
        index = self._action_index.to(x0.device).view(1, 1, -1)
        return full.scatter(-1, index.expand(batch_size, horizon, -1), raw_sequence)

    # ------------------------------------------------------------------
    # Context helpers and budget projection
    # ------------------------------------------------------------------
    @staticmethod
    def _context_value(batch: Batch, key: str, default: Tensor) -> Tensor:
        if batch.context is not None and key in batch.context:
            return batch.context[key]
        return default

    def _sensitive(self, batch: Batch) -> Tensor:
        default = batch.x[:, 0, 0]
        value = self._context_value(batch, "sensitive", default)
        return torch.as_tensor(value, dtype=batch.x.dtype, device=batch.x.device).reshape(
            batch.x.shape[0]
        )

    def _cost_weights(self, batch: Batch) -> Tensor:
        value = self._context_value(batch, "cost_weights", self.cost_weights)
        weights = torch.as_tensor(value, dtype=batch.x.dtype, device=batch.x.device).reshape(-1)
        if weights.numel() == 1:
            weights = weights.expand(self.state_dim)
        return weights

    def _budget(self, batch: Batch) -> Tensor:
        value = self._context_value(batch, "budget", torch.as_tensor(self.budget))
        budget = torch.as_tensor(value, dtype=batch.x.dtype, device=batch.x.device).reshape(-1)
        return budget.expand(batch.x.shape[0]) if budget.numel() == 1 else budget

    @staticmethod
    def _project_budget(
        delta: Tensor, budget: Tensor, cost_weights: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Hard global projection: scale ``delta`` down if its cost exceeds budget."""
        cost = (delta.abs() * cost_weights).sum(dim=(1, 2))
        scale = torch.clamp(budget / cost.clamp_min(1e-8), min=0.0, max=1.0)
        projected = delta * scale[:, None, None]
        return projected, cost, cost * scale

    # ------------------------------------------------------------------
    # Base-model contract
    # ------------------------------------------------------------------
    def forward(self, batch: Batch) -> dict[str, Tensor]:
        states = batch.x
        batch_size, horizon, _ = states.shape
        x0 = states[:, 0]
        sensitive = self._sensitive(batch)
        step_cap = self.step_cap
        if batch.context is not None and "step_cap" in batch.context:
            value = torch.as_tensor(batch.context["step_cap"], dtype=torch.float32)
            step_cap = float(value.detach().reshape(-1).mean())
        budget = self._budget(batch)
        cost_weights = self._cost_weights(batch)

        raw_intervention = self._generate_interventions(x0, sensitive, step_cap, horizon)
        intervention, cost_raw, cost = self._project_budget(raw_intervention, budget, cost_weights)
        counterfactual, probability, decision = self._rollout(x0, sensitive, intervention)
        return {
            "intervention": intervention,
            "intervention_raw": raw_intervention,
            "counterfactual": counterfactual,
            "probability": probability,
            "decision": decision,
            "factual": states,
            "sensitive": sensitive,
            "next_state": self._factual_next(states, sensitive),
            "propagation": self.propagation_matrix(),
            "direct_propagation": self.direct_propagation,
            "cost": cost,
            "cost_raw": cost_raw,
            "budget": budget,
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        probability = outputs["probability"]
        sensitive = outputs["sensitive"]
        improvement = ((probability - 1.0) ** 2).mean()
        fairness = self._dp_gap(probability, sensitive)
        budget_hinge = torch.relu(outputs["cost_raw"] - outputs["budget"]).pow(2).mean()

        if batch.x.shape[1] > 1:
            transition = F.mse_loss(outputs["next_state"], batch.x[:, 1:])
        else:
            transition = probability.new_zeros(())

        target_propagation = None
        if batch.context is not None and "propagation" in batch.context:
            target_propagation = batch.context["propagation"].to(
                outputs["propagation"].device, outputs["propagation"].dtype
            )
        if target_propagation is None:
            propagation = probability.new_zeros(())
        else:
            propagation = F.mse_loss(outputs["propagation"], target_propagation)

        if self.loss_weights["long_term"] > 0.0:
            long_term = self._long_term_gap(outputs["counterfactual"], sensitive)
        else:
            long_term = probability.new_zeros(())

        terms = {
            "improvement": improvement,
            "fairness": fairness,
            "budget": budget_hinge,
            "transition": transition,
            "propagation": propagation,
            "long_term": long_term,
        }
        total = probability.new_zeros(())
        for name, value in terms.items():
            total = total + self.loss_weights[name] * value
        return {"loss": total, **terms}

    @torch.no_grad()
    def act(self, states: Tensor, preference: Tensor | None = None) -> Tensor:
        """Return the recourse (counterfactual) trajectory for a state history."""
        if states.ndim == 2:
            x0 = states
            horizon = self.horizon
        elif states.ndim == 3:
            x0 = states[:, 0]
            horizon = states.shape[1]
        else:
            raise ValueError("states must have shape (B, d) or (B, T, d).")
        sensitive = x0[:, 0]
        raw_intervention = self._generate_interventions(x0, sensitive, self.step_cap, horizon)
        budget = torch.as_tensor(self.budget, dtype=x0.dtype, device=x0.device).expand(x0.shape[0])
        intervention, _, _ = self._project_budget(
            raw_intervention, budget, self.cost_weights.to(x0.device, x0.dtype)
        )
        counterfactual, _, _ = self._rollout(x0, sensitive, intervention)
        return counterfactual

    @torch.no_grad()
    def recourse(self, batch: Batch) -> dict[str, Tensor]:
        """Return the intervention, counterfactual trajectory and decisions."""
        outputs = self.forward(batch)
        return {
            "intervention": outputs["intervention"],
            "counterfactual": outputs["counterfactual"],
            "decision": outputs["decision"],
            "probability": outputs["probability"],
            "cost": outputs["cost"],
        }

    # ------------------------------------------------------------------
    # Fairness terms
    # ------------------------------------------------------------------
    @staticmethod
    def _dp_gap(probability: Tensor, sensitive: Tensor) -> Tensor:
        """Mean absolute demographic-parity gap over time steps."""
        group_one = probability[sensitive > 0.5]
        group_zero = probability[sensitive <= 0.5]
        if group_one.numel() == 0 or group_zero.numel() == 0:
            return probability.new_zeros(())
        return (group_one.mean(dim=0) - group_zero.mean(dim=0)).abs().mean()

    def _long_term_gap(self, counterfactual: Tensor, sensitive: Tensor) -> Tensor:
        """Optional entropic-OT gap between group profiles at the final step."""
        features = counterfactual[:, -1, 1:]
        group_one = features[sensitive > 0.5]
        group_zero = features[sensitive <= 0.5]
        if group_one.numel() == 0 or group_zero.numel() == 0:
            return features.new_zeros(())
        return self._sinkhorn2(
            group_one,
            group_zero,
            epsilon=self.sinkhorn_eps,
            n_iters=self.sinkhorn_iters,
        )

    @staticmethod
    def _sinkhorn2(source: Tensor, target: Tensor, *, epsilon: float, n_iters: int) -> Tensor:
        """Entropic Sinkhorn transport cost between two feature clouds."""
        cost = torch.cdist(source, target).pow(2)
        log_kernel = -cost / epsilon
        log_mu = source.new_full((source.shape[0],), -math.log(source.shape[0]))
        log_nu = target.new_full((target.shape[0],), -math.log(target.shape[0]))
        u = source.new_zeros(source.shape[0])
        v = target.new_zeros(target.shape[0])
        for _ in range(n_iters):
            u = log_mu - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)
        plan = torch.exp(log_kernel + u.unsqueeze(1) + v.unsqueeze(0))
        return (plan * cost).sum()

    def identifiability_statement(self) -> str:
        return (
            "SCARF assumes the profile's causal graph and the additive linear "
            "propagation x_M += W delta are known (the classifier is group-blind). "
            "Under those assumptions the counterfactual trajectory is identified "
            "by abduction-action-prediction with the stored exogenous noise; "
            "fairness (DP gap / Sinkhorn) is measured, not causally identified."
        )
