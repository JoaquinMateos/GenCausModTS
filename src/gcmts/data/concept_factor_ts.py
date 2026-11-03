"""Concept-factor time series for the Becker et al. explanation surrogate (III.C).

The generator instantiates the compact data-generating process used by
:class:`gcmts.explanation.methods.becker_attribution.BeckerAttribution`:

* ``C = 4`` latent factors ``z`` are sampled i.i.d. standard normal;
* the ``c``-th factor is active on a contiguous time segment and enters the
  observations through a fixed loading vector ``m_c``:
  ``x_t = sum_c 1[t in segment_c] * tanh(z_c) * m_c + noise``;
* the binary label follows
  ``y ~ Bernoulli(sigmoid(sum_c w_c tanh(z_c)))`` with ``w = (2, 0, -1.5, 0)``,
  so factors ``0`` and ``2`` are causal and factors ``1`` and ``3`` do not
  enter the label mechanism. Factor ``1`` is additionally made predictive of
  ``y`` (a confounded nuisance) while keeping ``w_1 = 0``;
* each sample is paired with a sample that differs in exactly one factor,
  which supplies the Ada-GVAE pairing step.

Because the label mechanism is known, the context exposes two ground-truth
attribution heads: the gradient head ``w_c (1 - tanh(z_c)^2)`` and the
mean-ablation head ``w_c tanh(z_c)`` (the label logit difference obtained by
setting factor ``c`` to its zero mean).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["ConceptFactorTimeSeriesGenerator"]

# Fixed loadings for the default (C=4, D=6) configuration. Each factor occupies
# a distinct pair of channels, with a small overlap on the last row.
_FIXED_LOADINGS: tuple[tuple[float, ...], ...] = (
    (1.0, 0.6, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.6, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0, 0.6),
    (0.6, 0.0, 0.0, 0.0, 0.0, 1.0),
)

# Label-mechanism weights: only factors 0 and 2 are causal.
_DEFAULT_WEIGHTS = (2.0, 0.0, -1.5, 0.0)


class ConceptFactorTimeSeriesGenerator(BaseDataGenerator):
    """Segment-wise concept-factor time series with known attribution targets.

    Args:
        n_channels: observation dimension ``D`` (default 6).
        horizon: sequence length ``T`` (default 12).
        n_factors: number of latent factors ``C`` (default 4).
        noise_std: standard deviation of the additive observation noise.
        confound_strength: correlation of the nuisance factor ``1`` with the
            label; set to zero to make it independent.
        seed: RNG seed. Two generators with the same seed produce identical
            samples.

    The returned :class:`~gcmts.typing.Batch` has ``x`` of shape
    ``(B, T, D)``, ground-truth latents ``z`` of shape ``(B, T, C)``, binary
    ``outcome`` of shape ``(B,)`` and context keys ``z`` ``(B, C)``, ``w``
    ``(C,)``, ``causal_concepts``, ``true_attribution``, ``true_ablation``
    and ``pair_x``.
    """

    def __init__(
        self,
        *,
        n_channels: int = 6,
        horizon: int = 12,
        n_factors: int = 4,
        noise_std: float = 0.1,
        confound_strength: float = 0.8,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=n_channels, horizon=horizon)
        if n_factors < 1:
            raise ValueError("n_factors must be positive.")
        if not 0.0 <= confound_strength < 1.0:
            raise ValueError("confound_strength must lie in [0, 1).")
        self.n_channels = n_channels
        self.n_factors = n_factors
        self.noise_std = noise_std
        self.confound_strength = confound_strength
        self.rng = torch.Generator().manual_seed(seed if seed is not None else 0)
        self.loadings = self._make_loadings(n_channels, n_factors)
        weights = torch.zeros(n_factors)
        default = torch.tensor(_DEFAULT_WEIGHTS)
        weights[: min(n_factors, default.numel())] = default[: min(n_factors, default.numel())]
        self.weights = weights
        self.causal_concepts = torch.nonzero(weights != 0).reshape(-1).long()
        self.nuisance_factor = 1 if n_factors > 1 else None

    @staticmethod
    def _make_loadings(n_channels: int, n_factors: int) -> Tensor:
        if (n_factors, n_channels) == (4, 6):
            return torch.tensor(_FIXED_LOADINGS)
        loadings = torch.zeros(n_factors, n_channels)
        for factor in range(n_factors):
            first = factor % n_channels
            second = (factor + 1) % n_channels
            loadings[factor, first] = 1.0
            if second != first:
                loadings[factor, second] = 0.6
        return loadings

    def concept_ids(self, horizon: int | None = None) -> Tensor:
        """Contiguous segment id (length ``horizon``) for each time step."""
        horizon = horizon or self.horizon
        width = max(1, horizon // self.n_factors)
        return (torch.arange(horizon) // width).clamp(max=self.n_factors - 1).long()

    def _render(self, z: Tensor, horizon: int) -> Tensor:
        """Map factors ``(B, C)`` to segment-wise observations ``(B, T, D)``."""
        concept = self.concept_ids(horizon)
        segment = F.one_hot(concept, self.n_factors).float()  # (T, C)
        contribution = z.tanh().unsqueeze(-1) * self.loadings.unsqueeze(0)  # (B, C, D)
        signal = torch.einsum("tc,bcd->btd", segment, contribution)
        noise = self.noise_std * torch.randn(
            z.shape[0], horizon, self.n_channels, generator=self.rng
        )
        return signal + noise

    def _attribution_targets(self, z: Tensor) -> tuple[Tensor, Tensor]:
        tanh = z.tanh()
        gradient = self.weights.unsqueeze(0) * (1.0 - tanh**2)
        ablation = self.weights.unsqueeze(0) * tanh
        return gradient, ablation

    def sample(
        self,
        n_samples: int,
        horizon: int | None = None,
        *,
        z: Tensor | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Draw ``n_samples`` trajectories; optionally condition on explicit ``z``."""
        del kwargs
        horizon = horizon or self.horizon
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if z is None:
            factors = torch.randn(n_samples, self.n_factors, generator=self.rng)
        else:
            factors = z.detach().clone().float()
            if factors.shape != (n_samples, self.n_factors):
                raise ValueError(
                    f"Expected z of shape ({n_samples}, {self.n_factors}), "
                    f"got {tuple(factors.shape)}."
                )

        logits = (factors.tanh() * self.weights.unsqueeze(0)).sum(dim=-1)
        labels = torch.bernoulli(torch.sigmoid(logits), generator=self.rng).long()

        nuisance = self.nuisance_factor
        if z is None and nuisance is not None and self.confound_strength > 0.0:
            # Make the nuisance factor predictive of the label without changing
            # the label mechanism (w_1 = 0).
            eps = torch.randn(n_samples, generator=self.rng)
            factors = factors.clone()
            factors[:, nuisance] = (
                self.confound_strength * (2.0 * labels.float() - 1.0)
                + (1.0 - self.confound_strength**2) ** 0.5 * eps
            )

        # Paired sample: resample one factor per sample, keep the rest fixed.
        pair_factors = factors.clone()
        pair_index = torch.randint(0, self.n_factors, (n_samples,), generator=self.rng)
        fresh = torch.randn(n_samples, generator=self.rng)
        pair_factors[torch.arange(n_samples), pair_index] = fresh

        x = self._render(factors, horizon)
        pair_x = self._render(pair_factors, horizon)
        true_attribution, true_ablation = self._attribution_targets(factors)
        context = {
            "z": factors,
            "w": self.weights.clone(),
            "causal_concepts": self.causal_concepts.clone(),
            "true_attribution": true_attribution,
            "true_ablation": true_ablation,
            "pair_x": pair_x,
            "y": labels,
        }
        z_series = factors.unsqueeze(1).expand(-1, horizon, -1).contiguous()
        return Batch(x=x, z=z_series, context=context, outcome=labels)
