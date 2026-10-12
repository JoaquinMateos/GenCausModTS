"""Binary SCM generator for CausalGAN-style interventional benchmarks.

``K`` binary labels follow a random DAG in topological order,

    l_i = 1[ u_i < sigma( sum_{j in Pa(i)} w_ji l_j + b_i ) ],

and observations are a fixed mixing of the label vector replicated over time
with additive noise. Interventions ``do(l_treatment = v)`` overwrite the target
label and re-propagate it to descendants while holding every noise term fixed,
so the interventional trajectories are exact.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["BinarySCMGenerator"]


class BinarySCMGenerator(BaseDataGenerator):
    """Binary-label SCM with a known DAG and an intervention target.

    Args:
        n_labels: number ``K`` of binary causal labels.
        observed_dim: observation dimension ``D``.
        horizon: sequence length ``T``.
        edge_prob: probability of a DAG edge ``j -> i`` for ``j < i``.
        treatment_index: intervention-target label (default ``K-1``).
        noise_std: observation-noise std.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        n_labels: int = 3,
        observed_dim: int = 6,
        horizon: int = 8,
        edge_prob: float = 0.5,
        treatment_index: int = -1,
        noise_std: float = 0.05,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.n_labels = n_labels
        self.treatment_index = treatment_index if treatment_index >= 0 else n_labels - 1
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        lower = torch.tril(torch.ones(n_labels, n_labels), diagonal=-1)
        self.adjacency = (torch.rand(n_labels, n_labels, generator=self.rng) < edge_prob) * lower
        self.adjacency = self.adjacency.float()
        self.weights = torch.randn(n_labels, n_labels, generator=self.rng) * self.adjacency
        self.bias = 0.2 * torch.randn(n_labels, generator=self.rng)
        self.mixing = torch.randn(observed_dim, n_labels, generator=self.rng)

    def _mechanisms(self, noise: Tensor, override: Tensor | None = None) -> Tensor:
        batch = noise.shape[0]
        labels = torch.zeros(batch, self.n_labels)
        for i in range(self.n_labels):
            if override is not None and bool(override[i] > 0.5):
                continue
            logits = labels @ self.weights[:, i] + self.bias[i]
            labels[:, i] = (noise[:, i] < torch.sigmoid(logits)).float()
        if override is not None:
            for i in range(self.n_labels):
                if bool(override[i] > 0.5):
                    labels[:, i] = 0.0
        return labels

    def sample(
        self,
        n_samples: int,
        horizon: int | None = None,
        *,
        treatment_value: float | None = None,
        **kwargs: Any,
    ) -> Batch:
        horizon = horizon or self.horizon
        label_noise = torch.rand(n_samples, self.n_labels, generator=self.rng)
        labels = self._mechanisms(label_noise)

        override = torch.zeros(self.n_labels)
        override[self.treatment_index] = 1.0
        counter_labels = labels.clone()
        counter_labels[:, self.treatment_index] = (
            0.0 if treatment_value is None else treatment_value
        )
        # Re-propagate the intervention to descendants using the shared noise.
        for i in range(self.treatment_index + 1, self.n_labels):
            if self.adjacency[:, i].sum() > 0:
                logits = counter_labels @ self.weights[:, i] + self.bias[i]
                counter_labels[:, i] = (label_noise[:, i] < torch.sigmoid(logits)).float()

        time_noise = self.noise_std * torch.randn(
            n_samples, horizon, self.observed_dim, generator=self.rng
        )
        base = labels @ self.mixing.T
        counter_base = counter_labels @ self.mixing.T
        x = base.unsqueeze(1) + time_noise
        counterfactual = counter_base.unsqueeze(1) + time_noise
        return Batch(
            x=x,
            context={
                "L": labels,
                "A": labels[:, self.treatment_index : self.treatment_index + 1],
                "adjacency": self.adjacency,
            },
            counterfactual=counterfactual,
            outcome=x[:, -1, 0],
        )
