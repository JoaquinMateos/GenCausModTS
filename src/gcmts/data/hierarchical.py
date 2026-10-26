"""Hierarchical temporal generator for CHiLD-style benchmarks.

Produces ``L`` latent layers in which each component has time-delayed parents
and, for all but the top layer, hierarchical (higher-layer) parents:

    z^l_{t,i} = f^l_i( z^l_{t-1}, z^{l+1}_t, eps^l_{t,i} ),
    z^L_{t,i} = f^L_i( z^L_{t-1}, eps^L_{t,i} ),

with only the bottom layer mixing to observations ``x_t = g(z^1_t)``. The
returned ``Batch.z`` concatenates the layers so that MCC can be computed against
the learned hierarchy.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.data.synthetic import _frozen_mlp, random_invertible_mixing
from gcmts.typing import Batch

__all__ = ["HierarchicalTemporalGenerator"]


class HierarchicalTemporalGenerator(BaseDataGenerator):
    """Two-or-more-layer hierarchical temporal process.

    Args:
        width: per-layer latent dimension ``n``.
        n_layers: number ``L`` of layers.
        horizon: sequence length ``T``.
        observed_dim: observation dimension; defaults to ``L * n`` for an
            invertible mixing map.
        mixing: ``"invertible"`` (requires ``observed_dim == L*n``) or a linear
            projection.
        mechanism_scale / noise_std: mechanism gain and process-noise std.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        width: int = 3,
        n_layers: int = 2,
        horizon: int = 12,
        observed_dim: int | None = None,
        mixing: str = "invertible",
        mechanism_scale: float = 0.7,
        noise_std: float = 0.1,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        self.width = width
        self.n_layers = n_layers
        self.latent_dim = width * n_layers
        observed_dim = observed_dim or width
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        if mixing == "invertible" and observed_dim != width:
            raise ValueError(
                "invertible mixing requires observed_dim == width (the bottom layer)."
            )
        self.mixing_type = mixing
        self.max_lag = 1
        self.mechanism_scale = mechanism_scale
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.mechanisms = torch.nn.ModuleList(
            _frozen_mlp(
                width if level == n_layers - 1 else 2 * width,
                width,
                hidden=hidden,
                generator=self.rng,
            )
            for level in range(n_layers)
        )
        if mixing == "invertible":
            self.mixing: torch.nn.Module = random_invertible_mixing(width, generator=self.rng)
        else:
            self.mixing = torch.nn.Linear(width, observed_dim, bias=False)
            with torch.no_grad():
                self.mixing.weight.copy_(
                    torch.randn(observed_dim, width, generator=self.rng)
                )

    def _advance(self, layers: list[Tensor]) -> list[Tensor]:
        """Advance one step; ``layers`` is ordered bottom (index 0) to top."""
        new: list[Tensor] = [torch.empty_like(layers[0]) for _ in layers]
        for level in range(self.n_layers - 1, -1, -1):
            lagged = layers[level]
            if level == self.n_layers - 1:
                parents = lagged
            else:
                # The higher layer at the *same* time step is already available
                # because we advance top-down; using ``layers[level+1]`` here
                # would make the hierarchical edge lagged instead of
                # contemporaneous, contradicting the generative model in the
                # docstring and the CHiLD benchmark.
                parents = torch.cat([lagged, new[level + 1]], dim=-1)
            mean = self.mechanism_scale * self.mechanisms[level](parents)
            noise = self.noise_std * torch.randn(
                lagged.shape[0], self.width, generator=self.rng
            )
            new[level] = torch.tanh(mean + noise)
        return new

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        layers = [
            torch.randn(n_samples, self.width, generator=self.rng)
            for _ in range(self.n_layers)
        ]
        trajectory = [layers]
        for _ in range(1, horizon):
            layers = self._advance(layers)
            trajectory.append(layers)
        stacked = [
            torch.stack([step[level] for step in trajectory], dim=1)
            for level in range(self.n_layers)
        ]
        z_flat = torch.cat(stacked, dim=-1)  # (B, T, L*n)
        bottom = stacked[0]
        with torch.no_grad():
            x = self.mixing(bottom.reshape(-1, self.width)).reshape(
                n_samples, horizon, self.observed_dim
            )
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z_flat)
