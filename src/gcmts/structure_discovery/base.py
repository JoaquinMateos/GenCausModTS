"""Base class for the structure-discovery branch (III.A).

Structure discovery returns a graph or a propagation path over already-observed
variables rather than a latent model or an effect. Following the review's
object-returned criterion, a discovery method is evaluated on structural
recovery (path-F1, source accuracy, structural-violation rate, temporal
deviation) against a benchmark that exposes the true graph and propagation
paths.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from torch import Tensor

from gcmts.core.base import BaseModel
from gcmts.typing import Batch, CausalLevel

__all__ = ["BaseStructureDiscovery"]


class BaseStructureDiscovery(BaseModel):
    """Abstract base for graph-constrained generative discovery methods."""

    def __init__(self, *, n_nodes: int, max_path_len: int = 5) -> None:
        super().__init__(observed_dim=n_nodes, latent_dim=None)
        self.n_nodes = n_nodes
        self.max_path_len = max_path_len

    @abstractmethod
    def forward(self, batch: Batch) -> dict[str, Tensor]:
        """Return reconstruction, node embeddings and path logits."""

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        """Return the total loss (key ``'loss'``)."""

    @abstractmethod
    def sample_paths(self, batch: Batch, n_samples: int = 1) -> Any:
        """Return the highest-likelihood propagation paths for each instance."""

    @abstractmethod
    def identifiability_statement(self) -> str:
        """Describe what is recovered and under which graph assumptions."""

    def causal_level(self) -> str:
        return CausalLevel.ASSOCIATION

    def get_latent_adjacency(self) -> Tensor | None:
        return None
