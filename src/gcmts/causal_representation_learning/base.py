"""Identifiable Causal Representation Learning for time series.

This module groups methods whose goal is to invert the observation process
``x_t = g(z_t, η_t)`` and recover the latent causal variables and/or temporal
mechanisms up to the appropriate ambiguity class.

Key references (BibTeX keys from the review):
* LEAP — ``weiranyaoLearningTemporallyCausal2021``
* TDRL — ``yaoTemporallyDisentangledRepresentation2022``
* NCTRL — ``songTemporallyDisentangledRepresentation2023``
* CtrlNS — ``songCausalTemporalRepresentation2024``
* IDOL — ``liIdentificationTemporallyCausal2026``
* DMM — ``ruichucaiCausalViewTime2025``
* Slow Flows — ``edouardpineauTimeSeriesSource2020``
* SNICA — ``hermannihalvaDisentanglingIdentifiableFeatures2021``
* CITRIS / iCITRIS — ``lippeCITRISCausalIdentifiability2022``,
  ``lippeCausalRepresentationLearning2023``
* MOSAIC — ``shichengfanMOSAICModuleDiscovery2026``
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from gcmts.core.base import BaseModel
from gcmts.typing import Batch, CausalRepresentationOutput

__all__ = ["BaseCausalRepresentationLearner"]


def _last_step_context(
    context: dict[str, Tensor] | None, time: int
) -> dict[str, Tensor] | None:
    """Reduce a per-timestep context to its last step for autoregressive rollout."""
    if context is None:
        return None
    return {
        key: value[:, -1:] if (value.ndim == 3 and value.shape[1] == time) else value
        for key, value in context.items()
    }


class BaseCausalRepresentationLearner(BaseModel):
    """Abstract base for identifiable causal representation learners.

    The expected data flow is:

        ``x_{1:T} -> encode -> z_{1:T} -> transition/mechanism -> z_{2:T+1}``
        ``z_t -> decode -> x_t``

    Subclasses decide whether to model the encoder/decoder (VAE/iVAE/flow),
    the latent transition (temporal SCM prior), or both.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        max_lag: int = 1,
    ) -> None:
        super().__init__(observed_dim=observed_dim, latent_dim=latent_dim)
        self.max_lag = max_lag

    @abstractmethod
    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        """Map observations ``(B, T, D)`` to latents ``(B, T, d)``.

        ``context`` may carry auxiliary variables (regimes ``u``, intervention
        targets ``I``) for conditional methods; unconditional methods ignore it.
        """

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """Map latents ``(B, T, d)`` back to observations ``(B, T, D)``."""

    @abstractmethod
    def transition(
        self,
        z_past: Tensor,
        context: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Apply one-step latent mechanism ``p(z_t | z_{t-p:t-1}, u_t)``.

        Args:
            z_past: ``(B, p, d)`` lagged latents
            context: optional regime / intervention / domain context
        """

    @abstractmethod
    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        """Return latents, reconstructions, adjacency, and log-probabilities."""

    def predict_next_latents(
        self,
        z: Tensor,
        steps: int,
        context: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Auto-regressively forecast latents for ``steps`` time points."""
        history = z
        preds = []
        for _ in range(steps):
            z_next = self.transition(history[:, -self.max_lag :], context)
            preds.append(z_next)
            history = torch.cat([history, z_next.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    def predict_next_observations(
        self,
        x: Tensor,
        steps: int,
        context: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Forecast observations by encoding, transitioning, and decoding."""
        z = self.encode(x, context)
        roll_context = _last_step_context(context, x.shape[1] if x.ndim == 3 else 1)
        z_future = self.predict_next_latents(z, steps, roll_context)
        return self.decode(z_future)

    @abstractmethod
    def get_latent_adjacency(self) -> Tensor | None:
        """Return estimated adjacency ``(d, d, p)`` if the method discovers it."""

    @abstractmethod
    def identifiability_statement(self) -> str:
        """State assumptions and ambiguity class, e.g. permutation + invertible."""
