"""Static–Dynamic Disentanglement for time series.

The goal is to split the latent space into invariant (static) and varying
(dynamic) causal factors for robust generalisation across domains or regimes.

Key references:
* SYNC — ``heLearningTimeAwareCausal2025``
* DAG-VAE — ``spulerDisentanglingRegionalImpacts2026``
* GCIM — ``zhaoGenerativeCausalInterpretation2023``
* CaDRe — ``minghaofuLearningGeneralCausal2025``
* UDA — ``zijianliNonstationaryTimeSeries2024``
"""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from gcmts.core.base import BaseModel
from gcmts.typing import Batch, DisentangledFactors

__all__ = ["BaseStaticDynamicDisentangler"]


class BaseStaticDynamicDisentangler(BaseModel):
    """Abstract base for static-dynamic disentanglers.

    The model decomposes observations into four groups of latent factors:

    * ``z_stc`` — static causal (invariant across time and domain)
    * ``z_dyc`` — dynamic causal (evolves over time, drives the target)
    * ``z_sts`` — static spurious (invariant but non-causal)
    * ``z_dys`` — dynamic spurious (time-varying but non-causal)

    Subclasses choose whether to model all four groups explicitly or only the
    static/dynamic split.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_static_dim: int,
        latent_dynamic_dim: int,
        latent_spurious_static_dim: int = 0,
        latent_spurious_dynamic_dim: int = 0,
    ) -> None:
        super().__init__(observed_dim=observed_dim, latent_dim=None)
        self.latent_static_dim = latent_static_dim
        self.latent_dynamic_dim = latent_dynamic_dim
        self.latent_spurious_static_dim = latent_spurious_static_dim
        self.latent_spurious_dynamic_dim = latent_spurious_dynamic_dim

    @property
    def latent_dim(self) -> int:
        return (
            self.latent_static_dim
            + self.latent_dynamic_dim
            + self.latent_spurious_static_dim
            + self.latent_spurious_dynamic_dim
        )

    @abstractmethod
    def encode_static(self, x: Tensor) -> Tensor:
        """Return ``(B, T, d_s)`` static factors (shared across the sequence)."""

    @abstractmethod
    def encode_dynamic(self, x: Tensor) -> Tensor:
        """Return ``(B, T, d_d)`` dynamic factors."""

    @abstractmethod
    def decode(self, factors: DisentangledFactors) -> Tensor:
        """Reconstruct observations from the disentangled factors."""

    @abstractmethod
    def disentangle(self, x: Tensor) -> DisentangledFactors:
        """Return all factor groups for input ``x``."""

    @abstractmethod
    def predict(
        self,
        x: Tensor,
        target_domain: Tensor | None = None,
    ) -> Tensor:
        """Predict under a new domain or regime.

        Args:
            x: input observations
            target_domain: optional domain index / embedding for domain adaptation
        """

    @abstractmethod
    def ood_score(self, x: Tensor, target_domain: Tensor | None = None) -> Tensor:
        """Return a scalar score quantifying distributional shift."""

    def forward(self, batch: Batch) -> DisentangledFactors:
        return self.disentangle(batch.x)

    @abstractmethod
    def identifiability_statement(self) -> str:
        """State the static/dynamic identifiability assumptions and ambiguity class."""
