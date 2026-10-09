r"""Voronoi / Causal Pinball-style instantaneous + lagged interventional SCM.

Faithful low-dimensional adaptation of the Voronoi and Causal Pinball
generators of Lippe et al. (2022, ``lippeCITRISCausalIdentifiability2022``;
``github.com/phlippe/CITRIS``, ``data_generation/data_generation_voronoi.py``
and ``data_generation_pinball.py``). The reference generators render polygons
and pinball frames with matplotlib; the *causal process* they share is:

* ``K`` bounded causal variables with a causal order,
* **instantaneous** (same time step) acyclic edges and **lagged** edges,
* single-target interventions applied at each time step (with a null
  intervention probability ``2/(K+2)``), with the intervention target recorded,
* a fixed invertible or non-linear mixing of the factors to observations.

We expose the low-dimensional factors plus the ground-truth graph so that
identifiability (MCC, R^2, SHD) and intervention-recovery metrics (triplet,
two-sample) can be computed without the rendering step. One extra *shared*
factor is appended, never intervened, matching the shared group that CITRIS
reserves for intervention-independent information.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.data.synthetic import _frozen_mlp, random_invertible_mixing
from gcmts.typing import Batch

__all__ = ["VoronoiGenerator"]


def soft_limit(value: Tensor, low: float, high: float) -> Tensor:
    """Smooth bounded reparameterisation (tanh rescaled to ``[low, high]``)."""
    return ((torch.tanh(value) + 1.0) / 2.0) * (high - low) + low


class VoronoiGenerator(BaseDataGenerator):
    r"""Interventional SCM with instantaneous and lagged edges.

    Args:
        n_vars: number ``K`` of causal variables (the shared factor is extra).
        horizon: sequence length ``T``.
        max_lag: number of lagged parent steps (only ``1`` is supported by the
            adjacency layout; instantaneous edges live at lag ``0``).
        instantaneous_sparsity / lagged_sparsity: edge probabilities.
        intv_std: standard deviation of the fresh noise drawn for an intervened
            variable.
        no_intervention_prob: probability of a time step without intervention;
            defaults to ``2 / (K + 2)`` as in the CITRIS implementation.
        mixing: ``"invertible"`` requires ``observed_dim == latent_dim`` and uses
            the affine-coupling flow of :mod:`gcmts.data.synthetic`;
            ``"nonlinear"`` uses a fixed MLP.
        standardize: z-score the factors (a component-wise invertible transform).
    """

    def __init__(
        self,
        *,
        n_vars: int = 5,
        horizon: int = 16,
        observed_dim: int | None = None,
        instantaneous_sparsity: float = 0.35,
        lagged_sparsity: float = 0.35,
        intv_std: float = 1.0,
        no_intervention_prob: float | None = None,
        mechanism_scale: float = 0.7,
        noise_std: float = 0.05,
        mixing: str = "invertible",
        hidden: int = 64,
        standardize: bool = True,
        seed: int | None = None,
    ) -> None:
        if mixing not in {"invertible", "nonlinear"}:
            raise ValueError("mixing must be 'invertible' or 'nonlinear'.")
        self.n_vars = n_vars
        self.latent_dim = n_vars + 1  # causal variables + shared factor
        observed_dim = observed_dim or self.latent_dim
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        if mixing == "invertible" and observed_dim != self.latent_dim:
            raise ValueError("invertible mixing requires observed_dim == latent_dim.")
        self.mixing_type = mixing
        self.max_lag = 1
        self.intv_std = intv_std
        self.no_intervention_prob = (
            2.0 / (n_vars + 2) if no_intervention_prob is None else no_intervention_prob
        )
        self.mechanism_scale = mechanism_scale
        self.noise_std = noise_std
        self.standardize = standardize
        self.rng = torch.Generator().manual_seed(seed or 0)

        # Instantaneous edges strictly in the causal order (acyclic by
        # construction); lagged edges are unstructured.
        inst = torch.rand(self.latent_dim, self.latent_dim, generator=self.rng)
        inst_draw = torch.rand(self.latent_dim, self.latent_dim, generator=self.rng)
        inst_mask = (inst_draw < instantaneous_sparsity).float()
        self.instantaneous = inst * inst_mask * torch.triu(
            torch.ones(self.latent_dim, self.latent_dim), diagonal=1
        )
        lagged = torch.rand(self.latent_dim, self.latent_dim, generator=self.rng)
        lag_draw = torch.rand(self.latent_dim, self.latent_dim, generator=self.rng)
        lag_mask = (lag_draw < lagged_sparsity).float()
        self.lagged = lagged * lag_mask
        self.adjacency = torch.stack([self.instantaneous, self.lagged], dim=-1)

        self.mechanisms = torch.nn.ModuleList(
            _frozen_mlp(2 * self.latent_dim, 1, hidden=hidden, generator=self.rng)
            for _ in range(self.latent_dim)
        )
        if mixing == "invertible":
            self.mixing: torch.nn.Module = random_invertible_mixing(
                self.latent_dim, generator=self.rng
            )
        else:
            self.mixing = _frozen_mlp(
                self.latent_dim, observed_dim, hidden=hidden, generator=self.rng
            )

    def _step(self, previous: Tensor, current: Tensor, targets: Tensor) -> Tensor:
        """Advance one time step; ``current`` is filled in causal order."""
        n_samples = previous.shape[0]
        for i in range(self.latent_dim):
            if i < self.n_vars:
                intervened = targets[:, i].bool()
                parents = torch.cat(
                    [
                        previous * self.lagged[i].unsqueeze(0),
                        current * self.instantaneous[i].unsqueeze(0),
                    ],
                    dim=-1,
                )
                mean = self.mechanism_scale * cast(
                    Tensor, self.mechanisms[i](parents)
                ).squeeze(-1)
                fresh = soft_limit(
                    self.intv_std
                    * torch.randn(n_samples, generator=self.rng),
                    -2.0,
                    2.0,
                )
                sampled = soft_limit(
                    mean + self.noise_std * torch.randn(n_samples, generator=self.rng),
                    -2.0,
                    2.0,
                )
                current[:, i] = torch.where(intervened, fresh, sampled)
            else:
                # Shared factor: never intervened, lagged parents only.
                parents = torch.cat(
                    [
                        previous * self.lagged[i].unsqueeze(0),
                        current * self.instantaneous[i].unsqueeze(0),
                    ],
                    dim=-1,
                )
                mean = self.mechanism_scale * cast(
                    Tensor, self.mechanisms[i](parents)
                ).squeeze(-1)
                current[:, i] = soft_limit(
                    mean + self.noise_std * torch.randn(n_samples, generator=self.rng),
                    -2.0,
                    2.0,
                )
        return current

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        z = torch.zeros(n_samples, horizon, self.latent_dim)
        z[:, 0] = torch.randn(n_samples, self.latent_dim, generator=self.rng)
        targets = torch.zeros(n_samples, horizon, self.n_vars)
        for t in range(1, horizon):
            no_intervention = torch.rand(n_samples, generator=self.rng) < self.no_intervention_prob
            target_index = torch.randint(0, self.n_vars, (n_samples,), generator=self.rng)
            for i in range(self.n_vars):
                targets[(~no_intervention) & (target_index == i), t, i] = 1.0
            z[:, t] = self._step(z[:, t - 1], torch.zeros_like(z[:, t]), targets[:, t])
        z_obs = z
        if self.standardize:
            z_obs = (z_obs - z_obs.mean(dim=(0, 1), keepdim=True)) / (
                z_obs.std(dim=(0, 1), keepdim=True) + 1e-6
            )
        with torch.no_grad():
            x = cast(
                Tensor, self.mixing(z_obs.reshape(-1, self.latent_dim))
            ).reshape(n_samples, horizon, self.observed_dim)
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(
            x=x,
            z=z_obs,
            context={"I": targets},
            adjacency=self.adjacency,
        )
