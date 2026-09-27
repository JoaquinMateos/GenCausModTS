"""Base dataset and generator abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gcmts.typing import Batch

__all__ = ["BaseDataGenerator"]


class BaseDataGenerator(ABC):
    """Abstract base for synthetic or real-world data generators."""

    def __init__(self, *, observed_dim: int, horizon: int) -> None:
        self.observed_dim = observed_dim
        self.horizon = horizon

    @abstractmethod
    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        """Return a :class:`Batch` with ``x`` of shape ``(n_samples, horizon, D)``."""

    def __iter__(self) -> Any:
        """Optional iterable interface for dataloaders."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
