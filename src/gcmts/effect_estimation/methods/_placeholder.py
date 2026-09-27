"""Placeholder concrete methods for effect estimation."""

from __future__ import annotations

import torch
from torch import Tensor

from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

__all__ = ["DummyEffectEstimator"]


class DummyEffectEstimator(BaseEffectEstimator):
    """Minimal functional stub that copies the factual trajectory."""

    def __init__(self, *, observed_dim: int, treatment_dim: int = 1) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        return {"noise": torch.zeros_like(batch.x)}

    def action(
        self,
        noise: dict[str, Tensor],
        intervention: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        return {**noise, **intervention}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        return context["noise"][:, :horizon]

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        return batch.x

    def counterfactual(
        self,
        batch: Batch,
        counterfactual_action: dict[str, Tensor],
    ) -> CounterfactualOutput:
        return CounterfactualOutput(
            factual=batch.x,
            counterfactual=batch.x,
            noise=self.abduction(batch),
            intervention=counterfactual_action,
        )

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        return {"x": batch.x}

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        return {"loss": torch.tensor(0.0)}

    def identifiability_statement(self) -> str:
        return "Dummy estimator: no causal guarantees."
