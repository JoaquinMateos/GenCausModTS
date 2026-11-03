r"""Counterfactual Generative Network (CNF) for compositional images.

Reference: Sauer & Geiger, *Counterfactual Generative Networks*, ICLR 2021,
arXiv:2101.06046; official code
https://github.com/autonomousvision/counterfactual_generative_networks (MIT).

The CNF factorises an image into three independent mechanisms,

.. math::
    x = m_s \odot f_t + (1 - m_s) \odot f_b,\quad
    m_s = \sigma(f_s(y_s, z_s)),\; f_t = f_t(y_t, z_t),\; f_b = f_b(y_b, z_b),

with observed labels :math:`y_s, y_t, y_b` and exogenous noises
:math:`z_s, z_t, z_b`.  A counterfactual swaps one label while holding all
noises fixed: abduction recovers the noises by per-sample gradient inversion,
action replaces the requested label and prediction recomposes the image.

Simplifications relative to the paper: a **surrogate renderer** (three small
MLPs over flattened 16x16 pixels instead of U-Net shape and VGG-style
texture/background generators); **no U2-Net** (labels are observed); **no
adversarial term or cGAN pseudo-ground-truth** (reconstruction plus mask
regularisers replace the paper's L1/perceptual loss against a pre-trained cGAN);
**no classifier or VGG texture term** (an unsupervised patch-consistency
penalty replaces the Gram-matrix loss); the paper's binary-mask entropy and
mean-interval penalties are kept.  A compact, CPU-friendly illustration of the
mechanism decomposition, not a reproduction of the paper's image quality.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gcmts.core.backbones import MLP
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CausalLevel, CounterfactualOutput

__all__ = ["CNF"]


class CNF(BaseEffectEstimator):
    r"""Compositional counterfactual generative network with gradient-inversion abduction.

    Args:
        observed_dim: flattened image dimension; defaults to ``3*image_size**2``
            and must agree with it when supplied.
        image_size: side of the square RGB images.
        n_classes: number of labels per mechanism.
        noise_dim: exogenous-noise dimension per mechanism.
        hidden_dims: MLP widths of the three generators.
        entropy_weight: weight of the binary mask-entropy penalty.
        mask_mean_weight: weight of the mask-mean interval penalty.
        texture_weight: weight of the patch-consistency texture penalty (0 disables it).
        mask_mean_interval: admissible ``(low, high)`` interval for the mean mask.
        inversion_steps: Adam steps used by :meth:`abduction`.
        inversion_lr: learning rate of the gradient inversion.
        inversion_seed: seed of the noise initialisation, making abduction reproducible.

    Labels are read from ``context["{mechanism}_label"]`` (``"class"`` is a
    fallback); interventions use the same keys or ``"A"`` for texture.
    """

    mechanisms = ("shape", "texture", "background")

    def __init__(
        self,
        *,
        observed_dim: int | None = None,
        image_size: int = 16,
        n_classes: int = 6,
        noise_dim: int = 4,
        hidden_dims: tuple[int, ...] = (64, 64),
        entropy_weight: float = 0.1,
        mask_mean_weight: float = 0.1,
        texture_weight: float = 0.01,
        mask_mean_interval: tuple[float, float] = (0.1, 0.9),
        inversion_steps: int = 30,
        inversion_lr: float = 0.1,
        inversion_seed: int = 0,
    ) -> None:
        expected = 3 * image_size * image_size
        if observed_dim is None:
            observed_dim = expected
        if observed_dim != expected:
            raise ValueError(
                f"observed_dim must equal 3*image_size**2 = {expected}, got {observed_dim}."
            )
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        if noise_dim < 1:
            raise ValueError("noise_dim must be positive.")
        if inversion_steps < 1:
            raise ValueError("inversion_steps must be positive.")
        low, high = mask_mean_interval
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("mask_mean_interval must satisfy 0 <= low < high <= 1.")
        super().__init__(observed_dim=observed_dim, treatment_dim=1)
        self.image_size = image_size
        self.n_classes = n_classes
        self.noise_dim = noise_dim
        self.entropy_weight = entropy_weight
        self.mask_mean_weight = mask_mean_weight
        self.texture_weight = texture_weight
        self.mask_mean_interval = mask_mean_interval
        self.inversion_steps = inversion_steps
        self.inversion_lr = inversion_lr
        self.inversion_seed = inversion_seed
        self.treatment_mechanism = "texture"

        in_dim = noise_dim + n_classes
        pixels = image_size * image_size
        self.shape_net = MLP(in_dim, pixels, hidden_dims=hidden_dims)
        self.texture_net = MLP(in_dim, 3 * pixels, hidden_dims=hidden_dims)
        self.background_net = MLP(in_dim, 3 * pixels, hidden_dims=hidden_dims)
        # A larger initial gain keeps the randomly initialised generators
        # sensitive to their noise inputs, so gradient inversion is informative
        # before any training has happened.
        with torch.no_grad():
            for net in (self.shape_net, self.texture_net, self.background_net):
                for module in net.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.mul_(4.0)

    # -- label / noise plumbing ---------------------------------------------

    def _coerce(self, value: Tensor, batch: int, device: torch.device) -> Tensor:
        """Coerce a label tensor to a ``(B,)`` long vector."""
        labels = value.to(device=device, dtype=torch.long)
        if labels.ndim >= 2:
            labels = labels.reshape(batch, -1)[:, 0]
        labels = labels.reshape(-1)
        if labels.numel() == 1:
            labels = labels.expand(batch)
        if labels.numel() != batch:
            raise ValueError(f"Expected {batch} labels, got {labels.numel()}.")
        return labels.clamp(0, self.n_classes - 1)

    def _labels(self, batch: Batch) -> dict[str, Tensor]:
        context = batch.context or {}
        batch_size = batch.x.shape[0]
        fallback = context.get("class")
        labels: dict[str, Tensor] = {}
        for mechanism in self.mechanisms:
            value = context.get(f"{mechanism}_label", fallback)
            if value is None:
                value = torch.zeros(batch_size, dtype=torch.long, device=batch.x.device)
            labels[mechanism] = self._coerce(value, batch_size, batch.x.device)
        return labels

    def _align_noise(self, noise: Tensor, time: int) -> Tensor:
        if noise.ndim == 2:
            return noise.unsqueeze(1).expand(-1, time, -1)
        if noise.shape[1] == time:
            return noise
        if noise.shape[1] > time:
            return noise[:, :time]
        padding = noise[:, -1:].expand(-1, time - noise.shape[1], -1)
        return torch.cat([noise, padding], dim=1)

    def _align_label(self, label: Tensor, batch: int, time: int) -> Tensor:
        value = self._coerce(label, batch, label.device)
        return value.unsqueeze(1).expand(batch, time)

    def _generate(
        self,
        noise: dict[str, Tensor],
        labels: dict[str, Tensor],
        batch: int,
        time: int,
    ) -> dict[str, Tensor]:
        nets = {
            "shape": self.shape_net,
            "texture": self.texture_net,
            "background": self.background_net,
        }
        raw: dict[str, Tensor] = {}
        for mechanism, net in nets.items():
            # Scale the noise so it has enough dynamic range to move the output
            # away from the label-dominated mean composition during inversion.
            z = 2.0 * self._align_noise(noise[f"noise_{mechanism}"], time)
            label = self._align_label(labels[mechanism], batch, time)
            one_hot = F.one_hot(label.reshape(-1), self.n_classes).float()
            features = torch.cat([z.reshape(batch * time, -1), one_hot], dim=-1)
            raw[mechanism] = net(features)

        side = self.image_size
        mask_logits = raw["shape"].reshape(batch, time, 1, side, side)
        mask = torch.sigmoid(mask_logits)
        foreground = torch.sigmoid(raw["texture"]).reshape(batch, time, 3, side, side)
        background = torch.sigmoid(raw["background"]).reshape(batch, time, 3, side, side)
        image = mask * foreground + (1.0 - mask) * background
        return {
            "recon": image.reshape(batch, time, -1),
            "image": image,
            "mask_logits": mask_logits,
            "mask": mask,
            "foreground": foreground,
            "background": background,
        }

    # -- BaseEffectEstimator contract ---------------------------------------

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        """Recover per-sample mechanism noise by gradient inversion of ``batch.x``.

        The noise tensors are optimised directly with Adam against the
        reconstruction MSE while the (observed) mechanism labels are held fixed.
        The returned dictionary also stores the reconstruction and the
        per-step error ``trace`` for diagnostics.
        """
        labels = self._labels(batch)
        batch_size, time = batch.x.shape[:2]
        dtype = next(self.parameters()).dtype
        generator = torch.Generator().manual_seed(self.inversion_seed)
        initial = 0.05 * torch.randn(
            batch_size,
            time,
            self.noise_dim,
            generator=generator,
        )
        noises = {
            f"noise_{mechanism}": initial.clone()
            .to(device=batch.x.device, dtype=dtype)
            .requires_grad_(True)
            for mechanism in self.mechanisms
        }
        optimiser = torch.optim.Adam(list(noises.values()), lr=self.inversion_lr)
        trace: list[float] = []
        with torch.enable_grad():
            for step in range(self.inversion_steps + 1):
                outputs = self._generate(noises, labels, batch_size, time)
                error = F.mse_loss(outputs["recon"], batch.x)
                trace.append(float(error.detach()))
                if step == self.inversion_steps:
                    break
                optimiser.zero_grad(set_to_none=True)
                error.backward()
                optimiser.step()
        # Inversion must not leave parameter gradients behind.
        self.zero_grad(set_to_none=True)
        result: dict[str, Tensor] = {name: value.detach() for name, value in noises.items()}
        result.update({mechanism: labels[mechanism] for mechanism in self.mechanisms})
        result["recon"] = outputs["recon"].detach()
        result["trace"] = torch.tensor(trace)
        return result

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Replace the requested mechanism labels, holding the noise fixed."""
        result = dict(noise)
        reference = noise["noise_shape"] if "noise_shape" in noise else next(iter(noise.values()))
        batch = reference.shape[0]
        device = reference.device
        for mechanism in self.mechanisms:
            value = intervention.get(mechanism, intervention.get(f"{mechanism}_label"))
            if value is not None:
                result[mechanism] = self._coerce(value, batch, device)
        if "A" in intervention:
            result[self.treatment_mechanism] = self._coerce(intervention["A"], batch, device)
        return result

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        """Recompose images from a context produced by :meth:`action`."""
        batch = context["noise_shape"].shape[0]
        noise = {
            f"noise_{mechanism}": self._align_noise(context[f"noise_{mechanism}"], horizon)
            for mechanism in self.mechanisms
        }
        labels = {mechanism: context[mechanism] for mechanism in self.mechanisms}
        return self._generate(noise, labels, batch, horizon)["recon"]

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        context = self.action(self.abduction(batch), intervention)
        return self.prediction(context, batch.x.shape[1])

    def counterfactual(
        self, batch: Batch, counterfactual_action: dict[str, Tensor]
    ) -> CounterfactualOutput:
        noise = self.abduction(batch)
        context = self.action(noise, counterfactual_action)
        return CounterfactualOutput(
            factual=batch.x,
            counterfactual=self.prediction(context, batch.x.shape[1]),
            noise=noise,
            intervention=counterfactual_action,
        )

    # -- training ------------------------------------------------------------

    def forward(self, batch: Batch, noise: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Generate images from fresh (or supplied) mechanism noise."""
        labels = self._labels(batch)
        batch_size, time = batch.x.shape[:2]
        if noise is None:
            dtype = next(self.parameters()).dtype
            noise = {
                f"noise_{mechanism}": torch.randn(
                    batch_size, time, self.noise_dim, device=batch.x.device, dtype=dtype
                )
                for mechanism in self.mechanisms
            }
        outputs = self._generate(noise, labels, batch_size, time)
        for mechanism in self.mechanisms:
            outputs[f"{mechanism}_label"] = labels[mechanism]
        return outputs

    def _patch_consistency(self, foreground: Tensor) -> Tensor:
        """Variance of local patch means; a VGG-free texture regulariser."""
        patch = max(1, min(4, self.image_size))
        flat = foreground.reshape(-1, *foreground.shape[-3:])
        means = F.avg_pool2d(flat, kernel_size=patch, stride=patch)
        return means.var(dim=(-1, -2), unbiased=False).mean()

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        """Reconstruction, mask entropy, mask-mean interval and texture penalties."""
        recon = (outputs["recon"] - batch.x).abs().mean()
        mask = outputs["mask"]
        entropy = -(
            mask * mask.clamp_min(1e-6).log() + (1.0 - mask) * (1.0 - mask).clamp_min(1e-6).log()
        ).mean()
        mask_mean = mask.mean(dim=(2, 3, 4))
        low, high = self.mask_mean_interval
        interval = (F.relu(low - mask_mean) + F.relu(mask_mean - high)).mean()
        texture = self._patch_consistency(outputs["foreground"])
        total = (
            recon
            + self.entropy_weight * entropy
            + self.mask_mean_weight * interval
            + self.texture_weight * texture
        )
        return {
            "loss": total,
            "recon": recon,
            "mask_entropy": entropy,
            "mask_mean": interval,
            "texture": texture,
        }

    def identifiability_statement(self) -> str:
        return (
            "CNF/CGN decomposes the image into independent shape, texture and "
            "background mechanisms; counterfactuals recompose abduced noise under a "
            "swapped label, identifying mechanism-level effects given observed "
            "labels. The surrogate MLP renderer and omitted classifier/adversarial "
            "terms bound pixel realism and class consistency."
        )

    def causal_level(self) -> str:
        return CausalLevel.COUNTERFACTUAL
