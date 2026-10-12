"""Base class for the decision-making branch (III.B).

Decision-making returns a policy, an action sequence or a recourse plan. It is
downstream of the other paradigms and carries the weakest causal guarantee: its
methods generate decisions under causal assumptions but generally prove no
decision-theoretic identifiability result. Evaluation therefore keeps decision
quality (hypervolume, regret, utility) separate from causal validity.
"""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from gcmts.core.base import BaseModel
from gcmts.typing import Batch, CausalLevel

__all__ = ["BaseDecisionMaker"]


class BaseDecisionMaker(BaseModel):
    """Abstract base for causal decision-making methods."""

    def __init__(self, *, state_dim: int, action_dim: int, n_objectives: int = 2) -> None:
        super().__init__(observed_dim=state_dim, latent_dim=None)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives

    @abstractmethod
    def forward(self, batch: Batch) -> dict[str, Tensor]:
        """Return latents, reconstructed actions and denoising targets."""

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        """Return the total loss (key ``'loss'``)."""

    @abstractmethod
    def act(self, states: Tensor, preference: Tensor | None = None) -> Tensor:
        """Generate an action sequence for the given state history."""

    @abstractmethod
    def identifiability_statement(self) -> str:
        """Describe the (weak) guarantee attached to the returned decision."""

    def causal_level(self) -> str:
        return CausalLevel.INTERVENTION

    def get_latent_adjacency(self) -> Tensor | None:
        return None
