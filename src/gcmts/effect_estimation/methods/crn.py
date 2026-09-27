"""CRN: Counterfactual Recurrent Network (sequential latent-dynamics estimator).

Reference: Bica et al., *Estimating Counterfactual Treatment Outcomes over Time
through Adversarial Balancing*, NeurIPS 2020 — BibTeX
``bicaEstimatingCounterfactualTreatment2020``.

Unlike a per-step autoencoder, the treatment acts on the latent *transition*, so
the reconstruction objective cannot be satisfied without modelling how the
treatment changes the trajectory. Abduction infers the initial latent state from
the factual sequence; prediction rolls the learned transition forward under the
counterfactual treatment while holding the abducted state fixed.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor

from gcmts.core.backbones import MLP, GaussianHead, gaussian_kl, gaussian_reconstruction_nll
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["CRN"]


class CRN(BaseEffectEstimator):
    """Latent-dynamics counterfactual estimator.

    Args:
        observed_dim: observation dimension ``D``.
        treatment_dim: treatment dimension.
        latent_dim: latent state dimension.
        hidden_dims: encoder/decoder widths.
        obs_noise: observation-noise std for the Gaussian NLL.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        latent_dim: int = 6,
        horizon: int = 24,
        hidden_dims: tuple[int, ...] = (128, 128),
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        self.state_dim = latent_dim
        self.horizon = horizon
        self.encoder = GaussianHead(observed_dim * horizon, latent_dim, hidden_dims=hidden_dims)
        self.decoder = MLP(latent_dim, observed_dim, hidden_dims=hidden_dims)
        self.dynamics = MLP(latent_dim + treatment_dim, latent_dim, hidden_dims=hidden_dims)
        self.register_buffer(
            "obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone()
        )
        self.obs_logvar: Tensor

    def _actions(self, batch: Batch) -> Tensor:
        x = batch.x
        action = None
        if batch.context is not None and "A" in batch.context:
            action = batch.context["A"]
        if action is None:
            action = torch.zeros(x.shape[0], x.shape[1], self.treatment_dim, device=x.device)
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, x.shape[1], -1)
        return action.float().to(x.device)

    def _roll(self, state: Tensor, actions: Tensor) -> Tensor:
        """Roll the latent transition; return states ``(B, T, d)``."""
        horizon = actions.shape[1]
        states = [state]
        current = state
        for t in range(horizon - 1):
            step = self.dynamics(torch.cat([current, actions[:, t]], dim=-1))
            current = current + step
            states.append(current)
        return torch.stack(states, dim=1)

    def _decode(self, states: Tensor) -> Tensor:
        batch, horizon, dim = states.shape
        return cast(Tensor, self.decoder(states.reshape(batch * horizon, dim))).reshape(
            batch, horizon, self.observed_dim
        )

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        mean, logvar = self.encoder(batch.x.reshape(batch.x.shape[0], -1))
        return {"state": mean, "logvar": logvar, "action": action}

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 2:
            new_action = new_action.unsqueeze(1).expand(-1, noise["action"].shape[1], -1)
        return {"state": noise["state"], "action": new_action.float()}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        states = self._roll(context["state"], context["action"][:, :horizon])
        return self._decode(states)

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        context = self.action(self.abduction(batch), intervention)
        return self.prediction(context, batch.x.shape[1])

    def counterfactual(
        self, batch: Batch, counterfactual_action: dict[str, Tensor]
    ) -> CounterfactualOutput:
        return CounterfactualOutput(
            factual=batch.x,
            counterfactual=self.intervene(batch, counterfactual_action),
            noise=self.abduction(batch),
            intervention=counterfactual_action,
        )

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        mean, logvar = self.encoder(batch.x.reshape(batch.x.shape[0], -1))
        state = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        states = self._roll(state, action)
        return {
            "state": state,
            "mean": mean,
            "logvar": logvar,
            "recon": self._decode(states),
            "action": action,
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        kl = gaussian_kl(
            outputs["mean"],
            outputs["logvar"],
            torch.zeros_like(outputs["mean"]),
            torch.zeros_like(outputs["logvar"]),
        ).mean()
        total = recon + kl
        return {"loss": total, "recon": recon, "kl": kl}

    def identifiability_statement(self) -> str:
        return (
            "CRN: the latent state is identifiable up to the decoder's "
            "reparameterisation when the latent transition is well specified and "
            "the treatment enters it; counterfactuals preserve the abducted state."
        )
