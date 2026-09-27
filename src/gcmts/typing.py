"""Shared types and lightweight data structures for ``gcmts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch
from torch import Tensor

__all__ = [
    "ArrayLike",
    "TensorDict",
    "CausalLevel",
    "AmbiguityClass",
    "Batch",
    "CausalRepresentationOutput",
    "CounterfactualOutput",
    "DisentangledFactors",
    "DataGenerator",
]

type ArrayLike = torch.Tensor | list[float]
type TensorDict = dict[str, Tensor]


class CausalLevel:
    """Pearl's causal hierarchy levels for time series."""

    ASSOCIATION = "L1_association"
    INTERVENTION = "L2_intervention"
    COUNTERFACTUAL = "L3_counterfactual"


class AmbiguityClass:
    """Common identifiability ambiguity classes."""

    PERM_COMPONENTWISE = "permutation_and_componentwise_invertible"
    PERM_SCALE = "permutation_and_scaling"
    PERM_ISOMETRY = "permutation_and_isometry"
    LINEAR_DEMIXING = "linear_demixing"
    MARKOV_EQUIVALENCE = "markov_equivalence_class"
    UNKNOWN = "unknown"


@dataclass
class Batch:
    """A generic batch produced by data loaders/generators."""

    x: Tensor  # observations (B, T, D)
    z: Tensor | None = None  # ground-truth latents (B, T, d), when available
    context: TensorDict | None = None  # regimes, interventions, domains
    mask: Tensor | None = None  # missingness / padding mask (B, T)

    def __post_init__(self) -> None:
        if self.x.ndim not in {2, 3}:
            raise ValueError("Batch.x must have shape (B, T, D) or (T, D).")


@dataclass
class CausalRepresentationOutput:
    """Return type for causal representation learners."""

    z: Tensor  # latents (B, T, d)
    x_recon: Tensor | None = None  # reconstruction (B, T, D)
    adjacency: Tensor | None = None  # (d, d, p) lagged adjacency
    log_prob: Tensor | None = None  # model log-likelihood
    extras: TensorDict | None = None  # method-specific tensors


@dataclass
class CounterfactualOutput:
    """Return type for effect estimators."""

    factual: Tensor  # original trajectory (B, T, D)
    counterfactual: Tensor  # counterfactual trajectory (B, T, D)
    noise: TensorDict | None = None  # inferred exogenous noise
    intervention: TensorDict | None = None  # do-specification used


@dataclass
class DisentangledFactors:
    """Return type for static-dynamic disentanglers.

    Following the review's four-way decomposition:
    * ``z_stc`` — static causal factors
    * ``z_dyc`` — dynamic causal factors
    * ``z_sts`` — static spurious factors
    * ``z_dys`` — dynamic spurious factors
    """

    z_stc: Tensor
    z_dyc: Tensor
    z_sts: Tensor | None = None
    z_dys: Tensor | None = None
    domain_index: Tensor | None = None


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class DataGenerator(Protocol[T_co]):
    """Protocol for dataset / trajectory generators."""

    def sample(self, n_samples: int, horizon: int, **kwargs: Any) -> T_co:
        """Generate ``n_samples`` trajectories of length ``horizon``."""
        ...
