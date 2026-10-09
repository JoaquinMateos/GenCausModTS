r"""Temporal Causal3DIdent-style image benchmark with ground-truth factors.

The reference Temporal Causal3DIdent dataset (von K\"ugelgen et al., 2021;
Lippe et al., 2023, ``lippeCausalRepresentationLearning2023``) renders a 3D
teapot with Blender and intervenes on its visual factors. A full Blender
pipeline is out of scope for a Python benchmark, so this module implements a
lightweight, fully differentiable *surrogate renderer*: a rotated coloured
object under a coloured spotlight on a coloured background. The causal process
--- lagged mechanisms with recorded single-target interventions --- matches the
reference, and every factor is exposed as ground truth so that MCC, R^2, DCI,
SHD and intervention metrics apply.

Images are stored flattened in ``Batch.x`` with shape ``(B, T, C*H*W)``; the
``image_shape`` attribute and :meth:`Causal3DIdentImageGenerator.to_image`
recover the ``(B, T, C, H, W)`` layout expected by
:class:`gcmts.core.backbones.CNNEncoder`.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.data.synthetic import _frozen_mlp
from gcmts.typing import Batch

__all__ = ["Causal3DIdentImageGenerator"]

_FACTOR_NAMES = (
    "object_x",
    "object_y",
    "object_rotation",
    "object_scale",
    "object_hue",
    "spotlight_x",
)


def hsv_to_rgb(hue: Tensor, saturation: float, value: float) -> Tensor:
    """Vectorised HSV -> RGB conversion for a scalar hue in ``[0, 1]``."""
    h = hue * 6.0
    i = torch.floor(h)
    f = h - i
    p = torch.full_like(h, value * (1.0 - saturation))
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    i = i.long() % 6
    conditions = [
        (i == 0, torch.stack([torch.full_like(h, value), t, p], dim=-1)),
        (i == 1, torch.stack([q, torch.full_like(h, value), p], dim=-1)),
        (i == 2, torch.stack([p, torch.full_like(h, value), t], dim=-1)),
        (i == 3, torch.stack([p, q, torch.full_like(h, value)], dim=-1)),
        (i == 4, torch.stack([t, p, torch.full_like(h, value)], dim=-1)),
        (i == 5, torch.stack([torch.full_like(h, value), p, q], dim=-1)),
    ]
    out = conditions[0][1]
    for index in range(1, 6):
        out = torch.where((i == index).unsqueeze(-1), conditions[index][1], out)
    return out


class _Renderer:
    """Differentiable renderer of a rotated object under coloured lighting."""

    def __init__(self, *, image_size: int = 32, temperature: float = 0.02) -> None:
        self.image_size = image_size
        self.temperature = temperature
        axis = torch.linspace(-1.0, 1.0, image_size)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_grid = (grid_x, grid_y)

    def __call__(self, factors: Tensor) -> Tensor:
        """Render ``(N, F)`` factors into ``(N, 3, H, W)`` RGB images."""
        grid_x, grid_y = self.register_grid
        (obj_x, obj_y, rotation, scale, hue, spot_x) = factors.unbind(dim=-1)

        cos = torch.cos(rotation)[:, None, None]
        sin = torch.sin(rotation)[:, None, None]
        dx = grid_x.unsqueeze(0) - obj_x[:, None, None]
        dy = grid_y.unsqueeze(0) - obj_y[:, None, None]
        u = cos * dx + sin * dy
        v = -sin * dx + cos * dy
        half = (0.35 * torch.sigmoid(scale))[:, None, None]
        mask = torch.sigmoid((half - (u.abs() + v.abs())) / self.temperature)

        object_color = hsv_to_rgb(hue, 0.85, 0.9).reshape(-1, 1, 1, 3)
        background_hue = (hue * 0.5 + 0.35) % 1.0
        background = hsv_to_rgb(background_hue, 0.25, 0.35).reshape(-1, 1, 1, 3)
        spotlight = torch.exp(
            -((grid_x.unsqueeze(0) - spot_x[:, None, None]) ** 2) / (2.0 * 0.6**2)
        )
        brightness = (0.4 + 0.6 * spotlight).unsqueeze(-1)
        image = background * (1.0 - mask.unsqueeze(-1)) + object_color * mask.unsqueeze(-1)
        image = image * brightness
        return image.permute(0, 3, 1, 2).contiguous()


class Causal3DIdentImageGenerator(BaseDataGenerator):
    r"""Temporal Causal3DIdent surrogate: rendered images, known factors, interventions.

    Six scalar factors (object position/rotation/scale/hue, spotlight position)
    evolve through a random lagged acyclic mechanism; at each time step a
    single factor may be intervened upon (fresh independent noise) and the
    binary target vector is recorded in ``context["I"]``. Images are rendered
    by :class:`_Renderer` and flattened into ``Batch.x``.
    """

    factor_names = _FACTOR_NAMES

    def __init__(
        self,
        *,
        horizon: int = 8,
        image_size: int = 32,
        adjacency_sparsity: float = 0.4,
        intervene_prob: float = 0.3,
        mechanism_scale: float = 0.7,
        noise_std: float = 0.1,
        observation_noise: float = 0.02,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        n_factors = len(_FACTOR_NAMES)
        self.n_factors = n_factors
        self.image_size = image_size
        super().__init__(observed_dim=3 * image_size * image_size, horizon=horizon)
        self.intervene_prob = intervene_prob
        self.observation_noise = observation_noise
        self.rng = torch.Generator().manual_seed(seed or 0)
        adj = torch.rand(n_factors, n_factors, generator=self.rng)
        mask = (torch.rand(n_factors, n_factors, generator=self.rng) < adjacency_sparsity).float()
        self.adjacency = (adj * mask * torch.triu(torch.ones(n_factors, n_factors), diagonal=1))
        self.adjacency = self.adjacency.unsqueeze(-1)
        self.mechanisms = torch.nn.ModuleList(
            _frozen_mlp(n_factors, 1, hidden=hidden, generator=self.rng)
            for _ in range(n_factors)
        )
        self.mechanism_scale = mechanism_scale
        self.noise_std = noise_std
        self.renderer = _Renderer(image_size=image_size)

    def _step(self, previous: Tensor, targets: Tensor) -> Tensor:
        n_samples = previous.shape[0]
        current = torch.zeros_like(previous)
        adjacency = self.adjacency[:, :, 0]
        for i in range(self.n_factors):
            parents = previous * adjacency[i].unsqueeze(0)
            mean = self.mechanism_scale * cast(
                Tensor, self.mechanisms[i](parents)
            ).squeeze(-1)
            sampled = mean + self.noise_std * torch.randn(n_samples, generator=self.rng)
            fresh = torch.randn(n_samples, generator=self.rng)
            current[:, i] = torch.where(targets[:, i].bool(), fresh, sampled)
        return current

    def _normalise_factors(self, factors: Tensor) -> Tensor:
        """Map factors to the renderer's expected ranges (component-wise invertible)."""
        normalised = factors.clone()
        for index in range(self.n_factors):
            column = normalised[..., index]
            column = (column - column.mean()) / (column.std() + 1e-6)
            normalised[..., index] = 0.5 * torch.tanh(column) + 0.5
        return normalised

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        z = torch.zeros(n_samples, horizon, self.n_factors)
        z[:, 0] = torch.randn(n_samples, self.n_factors, generator=self.rng)
        targets = torch.zeros(n_samples, horizon, self.n_factors)
        for t in range(1, horizon):
            draw = torch.rand(n_samples, generator=self.rng) < self.intervene_prob
            target_index = torch.randint(0, self.n_factors, (n_samples,), generator=self.rng)
            for i in range(self.n_factors):
                targets[(draw) & (target_index == i), t, i] = 1.0
            z[:, t] = self._step(z[:, t - 1], targets[:, t])
        factors = self._normalise_factors(z)
        flat = factors.reshape(-1, self.n_factors)
        with torch.no_grad():
            images = self.renderer(flat)
        images = images + self.observation_noise * torch.randn(
            images.shape, generator=self.rng
        )
        flattened = images.reshape(n_samples, horizon, -1)
        return Batch(x=flattened, z=factors, context={"I": targets})

    def to_image(self, x: Tensor) -> Tensor:
        """Reshape flattened observations ``(B, T, C*H*W)`` to ``(B, T, C, H, W)``."""
        batch, time, _ = x.shape
        side = self.image_size
        return x.reshape(batch, time, 3, side, side)
