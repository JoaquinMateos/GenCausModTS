"""Base class for the explanation & attribution branch (III.C).

Explanation returns an attribution of a prediction to input features or
concepts under intervention. Because attributions are relative to a classifier
rather than to a ground-truth mechanism, they are external-validity evidence
(rule R2), not proof; the standard checks are faithfulness, intervention
efficacy and stability.
"""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from gcmts.core.base import BaseModel
from gcmts.typing import Batch, CausalLevel

__all__ = ["BaseExplainer"]


class BaseExplainer(BaseModel):
    """Abstract base for causal explanation / attribution methods."""

    def __init__(
        self,
        *,
        n_channels: int,
        horizon: int,
        n_concepts: int,
        n_classes: int = 2,
    ) -> None:
        super().__init__(observed_dim=n_channels, latent_dim=None)
        self.n_channels = n_channels
        self.horizon = horizon
        self.n_concepts = n_concepts
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, batch: Batch) -> dict[str, Tensor]:
        """Return classifier logits and imputation losses/targets."""

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        """Return the total loss (key ``'loss'``)."""

    @abstractmethod
    def explain(self, batch: Batch) -> dict[str, Tensor]:
        """Return per-concept causal attributions (ITEs)."""

    @abstractmethod
    def identifiability_statement(self) -> str:
        """Describe the (weak) guarantee attached to the explanation."""

    def causal_level(self) -> str:
        return CausalLevel.COUNTERFACTUAL

    def get_latent_adjacency(self) -> Tensor | None:
        return None
