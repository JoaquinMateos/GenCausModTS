"""Core abstractions shared by all three paradigms."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gcmts.typing import Batch

logger = logging.getLogger(__name__)

__all__ = ["BaseModel", "BaseTrainer"]


class BaseModel(ABC, nn.Module):
    """Abstract base class for every method in the library.

    Subclasses must implement the method-specific forward contract and document
    the causal level they target (L1/L2/L3) plus the identifiability ambiguity
    class. This class intentionally stays framework-agnostic: it subclasses
    ``torch.nn.Module`` but does not prescribe a training loop (see
    :class:`BaseTrainer`).
    """

    def __init__(self, *, observed_dim: int, latent_dim: int | None = None) -> None:
        super().__init__()
        self.observed_dim = observed_dim
        self._latent_dim = latent_dim

    @property
    def latent_dim(self) -> int | None:
        """Latent dimension; subclasses may override this property."""
        return self._latent_dim

    @abstractmethod
    def forward(self, batch: Batch) -> Any:
        """Run a forward pass and return all tensors needed for loss/evaluation.

        The return type is method-specific (e.g. a
        :class:`gcmts.typing.CausalRepresentationOutput`); ``loss`` must accept
        exactly this object.
        """

    @abstractmethod
    def loss(self, outputs: Any, batch: Batch) -> dict[str, Tensor]:
        """Compute a dictionary of scalar loss terms.

        The key ``'loss'`` must contain the total differentiable objective.
        """

    @abstractmethod
    def identifiability_statement(self) -> str:
        """Return a human-readable statement of assumptions and ambiguity class."""

    def causal_level(self) -> str:
        """Return the highest Pearl level this model can answer.

        Defaults to L1; override when the model supports interventions or
        counterfactuals.
        """
        from gcmts.typing import CausalLevel

        return CausalLevel.ASSOCIATION

    def get_latent_adjacency(self) -> Tensor | None:
        """Return the estimated latent adjacency tensor if available.

        Shape: ``(latent_dim, latent_dim, max_lag)`` where the last axis indexes
        lags ``1..max_lag``.
        """
        return None

    def save(self, path: str | Path) -> None:
        """Serialize model weights and hyper-parameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "hparams": self._hparams()}, path)

    def load(self, path: str | Path, *, strict: bool = True) -> None:
        """Load model weights and hyper-parameters."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["state_dict"], strict=strict)

    def _hparams(self) -> dict[str, Any]:
        """Hyper-parameters to persist with the checkpoint."""
        return {
            "observed_dim": self.observed_dim,
            "latent_dim": self.latent_dim,
        }


class BaseTrainer(ABC):
    """Framework-agnostic trainer contract.

    Concrete implementations may wrap ``lightning.Trainer``, a plain PyTorch
    loop, or an ``optuna`` hyper-parameter search. The contract only requires
    that ``fit`` accepts a :class:`BaseModel` and a data generator / loader and
    returns a history dictionary.
    """

    def __init__(self, *, max_epochs: int = 100, device: str = "auto") -> None:
        self.max_epochs = max_epochs
        self.device = device

    @abstractmethod
    def fit(
        self,
        model: BaseModel,
        train_data: Any,
        val_data: Any | None = None,
    ) -> dict[str, list[float]]:
        """Train ``model`` and return a history mapping metric -> values."""

    @abstractmethod
    def predict(self, model: BaseModel, data: Any) -> Tensor | dict[str, Tensor]:
        """Run inference on ``data``."""

    def resolve_device(self, model: BaseModel) -> torch.device:
        """Public helper returning the device this trainer will use."""
        return self._device_for(model)

    def _device_for(self, model: BaseModel) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        cuda_ok = False
        try:
            cuda_ok = torch.cuda.is_available()
        except RuntimeError as error:  # e.g. driver/toolkit mismatch
            logger.warning("CUDA is built into torch but could not initialise: %s", error)
        if cuda_ok:
            return torch.device("cuda")
        if torch.version.cuda is not None:
            logger.warning(
                "CUDA runtime %s is present but unavailable; falling back to CPU. "
                "Check the NVIDIA driver is new enough for this torch build.",
                torch.version.cuda,
            )
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except RuntimeError:
            pass
        return torch.device("cpu")
