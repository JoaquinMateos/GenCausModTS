r"""Compositional image benchmark for counterfactual generative networks.

The benchmark follows Sauer & Geiger, *Counterfactual Generative Networks*
(ICLR 2021, arXiv:2101.06046), which factorises an image into three independent
mechanisms: a binary **shape** mask, a **texture** (the object's appearance) and
a **background**.  The reference implementation renders real images and
segments them with a pretrained U2-Net; this module instead uses a *fully
analytic surrogate renderer* of small RGB primitives, so the ground-truth mask,
foreground and background are available exactly and the benchmark runs on CPU
without external assets.

A latent class ``c`` is drawn uniformly.  Each mechanism label equals ``c`` with
probability ``class_prob`` (default 0.9) and is otherwise drawn uniformly from
the remaining classes, so the mechanisms are strongly but not perfectly
class-correlated, as in the paper.  The factual image is composed as
``mask * foreground + (1 - mask) * background``.  Paired counterfactuals swap a
single mechanism label while holding the nuisance noise (object jitter and
observation noise) fixed.

Images are stored flattened in ``Batch.x`` with shape ``(B, T, 3*H*W)``;
:meth:`CompositionalImageGenerator.to_image` recovers ``(B, T, 3, H, W)``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["CompositionalImageGenerator"]


def _hsv_to_rgb(hue: Tensor, saturation: float, value: float) -> Tensor:
    """Vectorised HSV -> RGB conversion for hues in ``[0, 1]``; returns ``(N, 3)``."""
    h = hue * 6.0
    i = torch.floor(h).long() % 6
    f = h - torch.floor(h)
    p = torch.full_like(h, value * (1.0 - saturation))
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    full = torch.full_like(h, value)
    choices = [
        torch.stack([full, t, p], dim=-1),
        torch.stack([q, full, p], dim=-1),
        torch.stack([p, full, t], dim=-1),
        torch.stack([p, q, full], dim=-1),
        torch.stack([t, p, full], dim=-1),
        torch.stack([full, p, q], dim=-1),
    ]
    out = choices[0]
    for index in range(1, 6):
        out = torch.where((i == index).unsqueeze(-1), choices[index], out)
    return out


def _shape_templates(n_classes: int, image_size: int, temperature: float = 0.02) -> Tensor:
    """Soft masks ``(n_classes, 1, H, W)`` for six primitive shape families."""
    axis = (torch.arange(image_size, dtype=torch.float32) + 0.5) / image_size * 2.0 - 1.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    ax, ay = xx.abs(), yy.abs()
    radius = torch.sqrt(xx**2 + yy**2)
    signed = [
        torch.maximum(ax, ay) - 0.55,  # square
        radius - 0.55,  # disc
        ax + ay - 0.80,  # diamond
        torch.maximum(yy - (0.55 - 0.85 * ax), -0.55 - yy),  # triangle
        torch.minimum(
            torch.maximum(ax - 0.18, ay - 0.62),
            torch.maximum(ay - 0.18, ax - 0.62),
        ),  # cross
        torch.maximum(0.22 - radius, radius - 0.65),  # ring
    ]
    templates = [torch.sigmoid(-signed[k % len(signed)] / temperature) for k in range(n_classes)]
    return torch.stack([template.unsqueeze(0) for template in templates])


class CompositionalImageGenerator(BaseDataGenerator):
    r"""Class-correlated shape/texture/background images with paired counterfactuals.

    Args:
        image_size: side ``H = W`` of the square RGB images.
        horizon: number of frames per sequence (the object is static apart from
            a small per-frame jitter).
        n_classes: number of shape, texture and background classes.
        class_prob: probability that a mechanism label equals the latent class.
        jitter: amplitude of the per-frame object translation.
        observation_noise: std of additive pixel noise shared by factual and
            counterfactual pairs.
        seed: RNG seed; two generators with the same seed return identical data.

    The returned ``Batch`` contains:

    * ``x`` — flattened images ``(B, T, 3*H*W)`` in ``[0, 1]``;
    * ``counterfactual`` — the texture-swap counterfactual, same shape;
    * ``context["mask"]`` — the factual soft mask ``(B, T, H, W)``;
    * ``context["foreground"]`` / ``context["background"]`` — ``(B, T, 3, H, W)``;
    * ``context["shape_label"]``, ``"texture_label"``, ``"background_label"`` —
      factual mechanism labels ``(B,)`` and ``context["class"]`` the latent class;
    * ``context["cf_shape"]``, ``"cf_texture"``, ``"cf_background"`` — flattened
      paired counterfactuals and their ``cf_*_label`` swapped labels.
    """

    def __init__(
        self,
        *,
        image_size: int = 16,
        horizon: int = 4,
        n_classes: int = 6,
        class_prob: float = 0.9,
        jitter: float = 0.06,
        observation_noise: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if image_size < 8:
            raise ValueError("image_size must be at least 8.")
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2 to define swaps.")
        if not 0.0 <= class_prob <= 1.0:
            raise ValueError("class_prob must lie in [0, 1].")
        super().__init__(observed_dim=3 * image_size * image_size, horizon=horizon)
        self.image_size = image_size
        self.n_classes = n_classes
        self.class_prob = class_prob
        self.jitter = jitter
        self.observation_noise = observation_noise
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.image_shape = (3, image_size, image_size)

        self.shape_templates = _shape_templates(n_classes, image_size)
        hue = torch.arange(n_classes, dtype=torch.float32) / n_classes
        self.texture_colours = _hsv_to_rgb(hue, 0.85, 0.95)
        self.background_colours = _hsv_to_rgb((hue + 0.5) % 1.0, 0.45, 0.45)
        axis = (torch.arange(image_size, dtype=torch.float32) + 0.5) / image_size * 2.0 - 1.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        patterns = []
        for k in range(n_classes):
            angle = math.pi * k / n_classes
            coordinate = math.cos(angle) * xx + math.sin(angle) * yy
            patterns.append(0.8 + 0.2 * torch.sin((2.0 + k % 3) * math.pi * coordinate))
        self.texture_patterns = torch.stack([p.unsqueeze(0) for p in patterns])
        self._background_shading = (0.85 + 0.15 * (yy + 1.0) / 2.0).unsqueeze(0).unsqueeze(0)

    # -- rendering -----------------------------------------------------------

    def _shift(self, masks: Tensor, offsets: Tensor) -> Tensor:
        """Translate ``(N, 1, H, W)`` masks by per-sample ``(dx, dy)`` offsets."""
        theta = torch.zeros(masks.shape[0], 2, 3, device=masks.device, dtype=masks.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = -offsets[:, 0]
        theta[:, 1, 2] = -offsets[:, 1]
        grid = F.affine_grid(theta, masks.shape, align_corners=False)
        return F.grid_sample(masks, grid, align_corners=False, padding_mode="zeros")

    def _render(
        self,
        shape_label: Tensor,
        texture_label: Tensor,
        background_label: Tensor,
        offsets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Render images ``(N, 3, H, W)`` plus their mask and foreground/background."""
        n = shape_label.shape[0]
        mask = self._shift(self.shape_templates[shape_label.long()], offsets)
        foreground = (
            self.texture_colours[texture_label.long()].view(n, 3, 1, 1)
            * self.texture_patterns[texture_label.long()]
        )
        background = (
            self.background_colours[background_label.long()].view(n, 3, 1, 1)
            * self._background_shading
        )
        image = mask * foreground + (1.0 - mask) * background
        return image, mask[:, 0], foreground, background

    # -- sampling ------------------------------------------------------------

    def _correlated_labels(self, class_index: Tensor) -> Tensor:
        batch = class_index.shape[0]
        keep = torch.rand(batch, generator=self.rng) < self.class_prob
        alternative = torch.randint(0, self.n_classes - 1, (batch,), generator=self.rng)
        alternative = alternative + (alternative >= class_index).long()
        return torch.where(keep, class_index, alternative)

    def _swap(self, labels: Tensor) -> Tensor:
        step = torch.randint(1, self.n_classes, labels.shape, generator=self.rng)
        return (labels + step) % self.n_classes

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        batch = n_samples
        image_size = self.image_size

        class_index = torch.randint(0, self.n_classes, (batch,), generator=self.rng)
        labels = {
            "shape": self._correlated_labels(class_index),
            "texture": self._correlated_labels(class_index),
            "background": self._correlated_labels(class_index),
        }
        swapped = {name: self._swap(value) for name, value in labels.items()}
        offsets = (torch.rand(batch * horizon, 2, generator=self.rng) - 0.5) * 2.0 * self.jitter
        observation_noise = self.observation_noise * torch.randn(
            batch * horizon, 3, image_size, image_size, generator=self.rng
        )
        repeated = {name: value.repeat_interleave(horizon) for name, value in labels.items()}

        factual = self._render(
            repeated["shape"], repeated["texture"], repeated["background"], offsets
        )
        counterfactual_images: dict[str, Tensor] = {}
        for mechanism in ("shape", "texture", "background"):
            swapped_labels = dict(repeated)
            swapped_labels[mechanism] = swapped[mechanism].repeat_interleave(horizon)
            counterfactual_images[mechanism] = self._render(
                swapped_labels["shape"],
                swapped_labels["texture"],
                swapped_labels["background"],
                offsets,
            )[0]

        def pack(image: Tensor) -> Tensor:
            noisy = (image + observation_noise).clamp(0.0, 1.0)
            return noisy.reshape(batch, horizon, -1)

        x = pack(factual[0])
        cf_images = {name: pack(image) for name, image in counterfactual_images.items()}
        context = {
            "mask": factual[1].reshape(batch, horizon, image_size, image_size),
            "foreground": factual[2].reshape(batch, horizon, 3, image_size, image_size),
            "background": factual[3].reshape(batch, horizon, 3, image_size, image_size),
            "class": class_index,
            "shape_label": labels["shape"],
            "texture_label": labels["texture"],
            "background_label": labels["background"],
            "cf_shape_label": swapped["shape"],
            "cf_texture_label": swapped["texture"],
            "cf_background_label": swapped["background"],
            "labels": torch.stack(
                [labels["shape"], labels["texture"], labels["background"]], dim=-1
            ),
        }
        for mechanism, image in cf_images.items():
            context[f"cf_{mechanism}"] = image
        return Batch(x=x, context=context, counterfactual=cf_images["texture"])

    def to_image(self, x: Tensor) -> Tensor:
        """Reshape flattened observations ``(B, T, C*H*W)`` to ``(B, T, C, H, W)``."""
        return x.reshape(*x.shape[:2], *self.image_shape)
