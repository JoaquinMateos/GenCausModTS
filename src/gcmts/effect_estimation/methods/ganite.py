"""GANITE: individualized treatment effects with a two-block conditional GAN.

Reference: Yoon, Jordon & van der Schaar, *GANITE: Estimation of Individualized
Treatment Effects using Generative Adversarial Nets*, ICLR 2018 — BibTeX
``yoonGANITEEstimationIndividualized2018``.

The audited implementation replaces the single conditional generator with the
two blocks of the paper:

* **counterfactual block** ``(G, D_G)``: ``G(x, t, y_f, z)`` generates a vector
  of potential outcomes for every treatment arm; ``D_G(x, y)`` predicts which
  arm supplied the factual component.  ``G`` is trained to reconstruct the
  factual arm and to fool ``D_G``.
* **ITE block** ``(I, D_I)``: ``I(x, z)`` generates the potential-outcome vector
  from features alone; ``D_I`` distinguishes the completed data set (factual
  arm from ``G``) from ``I``'s samples.  At inference ``I`` is the ITE
  estimator.

For time series the blocks are applied per timestep, which is the static ITE
formulation of the paper repeated over the trajectory; the treatment vector is
one-hot encoded over the ``2 ** treatment_dim`` arm combinations.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor, nn

from gcmts.core.backbones import MLP, GradientReversal
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["GANITE"]


class GANITE(BaseEffectEstimator):
    """Two-block GANITE estimator.

    Args:
        observed_dim: dimension of the observed vector (features plus outcome).
        treatment_dim: number of binary treatment variables.
        noise_dim: dimension of the generator noise.
        hidden_dims: MLP widths.
        adversarial_weight: weight of the counterfactual-discriminator term.
        obs_noise: observation-noise standard deviation for the supervised NLL.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        noise_dim: int = 8,
        hidden_dims: tuple[int, ...] = (128, 128),
        adversarial_weight: float = 1.0,
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        self.noise_dim = noise_dim
        self.adversarial_weight = adversarial_weight
        self.n_arms = 2 ** treatment_dim
        self.counterfactual_generator = MLP(
            2 * observed_dim + self.n_arms + noise_dim,
            self.n_arms * observed_dim,
            hidden_dims=hidden_dims,
        )
        self.counterfactual_discriminator = MLP(
            observed_dim + self.n_arms * observed_dim,
            self.n_arms,
            hidden_dims=hidden_dims,
        )
        self.ite_generator = MLP(
            observed_dim + noise_dim,
            self.n_arms * observed_dim,
            hidden_dims=hidden_dims,
        )
        self.ite_discriminator = MLP(
            observed_dim + self.n_arms * observed_dim,
            1,
            hidden_dims=hidden_dims,
        )
        self.reversal = GradientReversal(scale=1.0)
        self.register_buffer(
            "obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone()
        )
        self.obs_logvar: Tensor

    def _actions(self, batch: Batch) -> Tensor:
        action = None
        if batch.context is not None and "A" in batch.context:
            action = batch.context["A"]
        if action is None:
            action = torch.zeros(
                batch.x.shape[0], batch.x.shape[1], self.treatment_dim, device=batch.x.device
            )
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, batch.x.shape[1], -1)
        return action.float().to(batch.x.device)

    def _one_hot(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """Map a binary treatment vector to an arm index and one-hot encoding."""
        binary = (action > 0.5).long()
        weights = (2 ** torch.arange(self.treatment_dim, device=action.device)).long()
        index = (binary * weights).sum(dim=-1)
        return torch.nn.functional.one_hot(index, self.n_arms).float(), index

    def _potential(self, generated: Tensor, factual: Tensor, index: Tensor) -> Tensor:
        """Replace the factual arm's generated vector with the observed vector."""
        batch = factual.shape[0]
        potential = generated.reshape(batch, self.n_arms, self.observed_dim)
        potential[torch.arange(batch, device=factual.device), index] = factual
        return potential

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        return {"x": batch.x, "action": self._actions(batch)}

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 2:
            new_action = new_action.unsqueeze(1).expand(-1, noise["x"].shape[1], -1)
        return {"x": noise["x"], "action": new_action.float()}

    def _predict_arms(self, x: Tensor) -> Tensor:
        """Potential-outcome tensor ``(B, T, n_arms, D)`` from the ITE generator."""
        batch, time, dim = x.shape
        flat = x.reshape(batch * time, dim)
        noise = torch.zeros(flat.shape[0], self.noise_dim, device=x.device)
        generated = cast(Tensor, self.ite_generator(torch.cat([flat, noise], dim=-1))).clamp(
            -10.0, 10.0
        )
        return generated.reshape(batch, time, self.n_arms, dim)

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        arms = self._predict_arms(context["x"][:, :horizon])
        _, index = self._one_hot(context["action"][:, :horizon])
        selected = arms.gather(
            2,
            index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.observed_dim),
        ).squeeze(2)
        return selected

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        context = self.action(self.abduction(batch), intervention)
        return self.prediction(context, batch.x.shape[1])

    def counterfactual(
        self, batch: Batch, counterfactual_action: dict[str, Tensor]
    ) -> CounterfactualOutput:
        context = self.action(self.abduction(batch), counterfactual_action)
        return CounterfactualOutput(
            factual=batch.x,
            counterfactual=self.prediction(context, batch.x.shape[1]),
            noise=self.abduction(batch),
            intervention=counterfactual_action,
        )

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        batch_size, time, dim = batch.x.shape
        flat_x = batch.x.reshape(batch_size * time, dim)
        flat_action = action.reshape(batch_size * time, self.treatment_dim)
        arm, index = self._one_hot(flat_action)
        noise = torch.randn(flat_x.shape[0], self.noise_dim, device=flat_x.device)
        generated = cast(
            Tensor,
            self.counterfactual_generator(
                torch.cat([flat_x, arm, flat_x, noise], dim=-1)
            ),
        ).clamp(-10.0, 10.0)
        generated_arms = generated.reshape(flat_x.shape[0], self.n_arms, self.observed_dim)
        g_supervised = nn.functional.mse_loss(
            generated_arms[torch.arange(flat_x.shape[0], device=flat_x.device), index],
            flat_x,
        )
        potential = self._potential(generated, flat_x, index)
        completed = potential.reshape(flat_x.shape[0], -1)
        # The discriminator sees the completed vector through a gradient
        # reversal, so D_G minimises the classification loss while G maximises
        # it; D_G's own parameters receive the normal gradient.
        logits = self.counterfactual_discriminator(
            torch.cat([flat_x, self.reversal(completed)], dim=-1)
        )
        dg_loss = nn.functional.cross_entropy(logits, index)
        return {
            "potential": potential,
            "logits": logits,
            "dg_loss": dg_loss,
            "g_supervised": g_supervised,
            "action": flat_action,
            "index": index,
            "flat_x": flat_x,
            "noise": noise,
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        flat_x = outputs["flat_x"]
        index = outputs["index"]
        noise = torch.randn(flat_x.shape[0], self.noise_dim, device=flat_x.device)
        generated_ite = cast(
            Tensor,
            self.ite_generator(torch.cat([flat_x, noise], dim=-1)),
        ).clamp(-10.0, 10.0)
        generated_ite_arms = generated_ite.reshape(
            flat_x.shape[0], self.n_arms, self.observed_dim
        )
        ite_supervised = nn.functional.mse_loss(
            generated_ite_arms[torch.arange(flat_x.shape[0], device=flat_x.device), index],
            flat_x,
        )
        real = outputs["potential"].detach().reshape(flat_x.shape[0], -1)
        fake = generated_ite
        real_logits = self.ite_discriminator(torch.cat([flat_x, real], dim=-1))
        fake_logits = self.ite_discriminator(torch.cat([flat_x, fake.detach()], dim=-1))
        d_real = nn.functional.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        d_fake = nn.functional.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
        d_loss = 0.5 * (d_real + d_fake)
        g_adv = nn.functional.binary_cross_entropy_with_logits(
            self.ite_discriminator(torch.cat([flat_x, fake], dim=-1)),
            torch.ones(flat_x.shape[0], 1, device=flat_x.device),
        )
        total = (
            outputs["g_supervised"]
            + outputs["dg_loss"]
            + ite_supervised
            + d_loss
            + self.adversarial_weight * g_adv
        )
        return {
            "loss": total,
            "g_supervised": outputs["g_supervised"],
            "dg_loss": outputs["dg_loss"],
            "i_supervised": ite_supervised,
            "d_loss": d_loss,
            "g_adv": g_adv,
        }

    def identifiability_statement(self) -> str:
        return (
            "GANITE: a conditional GAN that imputes the unobserved potential "
            "outcomes; the ITE generator is identifiable only up to the "
            "assumptions of the conditional GAN (sufficient features, correct "
            "treatment arms and positivity)."
        )
