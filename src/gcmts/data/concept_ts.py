"""Concept time-series generator for explanation benchmarks (III.C).

A class label determines a per-concept mean signature; observations are the
concatenation of the concept segments with additive noise, so the concepts that
carry the class-discriminative signal are known ground-truth causal effects for
faithfulness evaluation.
"""

from __future__ import annotations

from typing import Any

import torch

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["ConceptTimeSeriesGenerator"]


class ConceptTimeSeriesGenerator(BaseDataGenerator):
    """Concept-segmented time series with a binary class label.

    Args:
        n_channels: number of channels ``D``.
        horizon: sequence length ``T``.
        n_concepts: number ``C`` of contiguous concept segments.
        causal_concepts: indices of concepts carrying the class signal.
        signal: signal magnitude for the causal concepts.
        noise_std: observation-noise std.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        n_channels: int = 4,
        horizon: int = 16,
        n_concepts: int = 4,
        causal_concepts: tuple[int, ...] = (0, 2),
        signal: float = 1.5,
        noise_std: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=n_channels, horizon=horizon)
        self.n_channels = n_channels
        self.n_concepts = n_concepts
        self.causal_concepts = set(causal_concepts)
        self.signal = signal
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        width = max(1, horizon // n_concepts)
        ids = (torch.arange(horizon) // width).clamp(max=n_concepts - 1)
        self.concept_ids = ids.long()

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        labels = torch.randint(0, 2, (n_samples,), generator=self.rng)
        x = self.noise_std * torch.randn(
            n_samples, horizon, self.n_channels, generator=self.rng
        )
        for concept in self.causal_concepts:
            mask = (self.concept_ids == concept).unsqueeze(0).unsqueeze(-1).float()
            effect = labels.float().reshape(-1, 1, 1) * mask * self.signal
            x = x + effect
        return Batch(x=x, context={"y": labels})
