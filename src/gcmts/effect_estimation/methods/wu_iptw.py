"""Wu-IPTW: counterfactual generative models for time-varying treatments.

Reference: Wu et al., *Counterfactual Generative Models for Time-Varying
Treatments*, KDD 2024 — BibTeX ``wuCounterfactualGenerativeModels2024``.

The counterfactual density ``f_a(y)`` is targeted through the g-formula
(Proposition 1): fit a propensity model ``f_phi(a_t | a_{t-1}, x_t)`` on the
observed data, then train a treatment-conditioned generator ``g_theta(z, a)`` on
the IPTW pseudo-population with weights

    w_phi(a, x) = 1 / prod_t f_phi(a_t | a_{t-1}, x_t).

We instantiate the generator as a conditional VAE (MSCVAE) and use a Gaussian
propensity (which reduces to the Bernoulli case for binary treatments). Weights
are truncated at the 1st/99th percentiles and normalised by their mean.
Assumptions: consistency, positivity and sequential strong ignorability; the
target is the interventional law ``P(Y(a))`` (Level 2).
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from gcmts.core.backbones import MLP, GaussianHead, gaussian_logpdf
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CausalLevel, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["WuIPTW"]


class WuIPTW(BaseEffectEstimator):
    """IPTW-reweighted conditional-VAE counterfactual estimator.

    Args:
        observed_dim: observation dimension ``D``.
        treatment_dim: treatment dimension.
        latent_dim: latent dimension ``r`` of the generator.
        hidden_dims: propensity/encoder/decoder widths.
        weight_clip: percentile for IPTW truncation (default 1%/99%).
        obs_noise: observation-noise std for the Gaussian likelihood.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        latent_dim: int = 6,
        hidden_dims: tuple[int, ...] = (128, 128),
        weight_clip: float = 1.0,
        obs_noise: float = 0.05,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        self.z_dim = latent_dim
        self.weight_clip = weight_clip
        # Propensity f_phi(a_t | a_{t-1}, x_t): Gaussian mean/logvar.
        self.propensity = GaussianHead(
            observed_dim + treatment_dim, treatment_dim, hidden_dims=hidden_dims
        )
        # Conditional VAE generator g_theta(z, a): q(z | y, a) and p(y | z, a).
        self.encoder = GaussianHead(
            observed_dim + treatment_dim, latent_dim, hidden_dims=hidden_dims
        )
        self.decoder = MLP(latent_dim + treatment_dim, observed_dim, hidden_dims=hidden_dims)
        self.register_buffer(
            "obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone()
        )
        self.obs_logvar: Tensor
        logger.debug("Initialised Wu-IPTW (D=%d, d=%d)", observed_dim, treatment_dim)

    # -- treatment handling --------------------------------------------------

    def _actions(self, batch: Batch) -> Tensor:
        x = batch.x
        action = None
        if batch.context is not None and "A" in batch.context:
            action = batch.context["A"]
        if action is None:
            action = torch.zeros(x.shape[0], x.shape[1], self.treatment_dim, device=x.device)
        if action.ndim == 2:
            action = action.unsqueeze(-1)
        return action.float().to(x.device)

    def _propensity(self, x: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Gaussian propensity parameters for ``a_t`` given ``(a_{t-1}, x_t)``."""
        previous = torch.zeros_like(action)
        previous[:, 1:] = action[:, :-1]
        features = torch.cat([x, previous], dim=-1)
        mean, logvar = self.propensity(features.reshape(-1, features.shape[-1]))
        return mean.reshape(x.shape[0], x.shape[1], -1), logvar.reshape(x.shape[0], x.shape[1], -1)

    def iptw_weights(self, batch: Batch) -> Tensor:
        """Per-trajectory IPTW weights ``w_phi(a, x)`` (truncated and normalised)."""
        action = self._actions(batch)
        mean, logvar = self._propensity(batch.x, action)
        log_density = gaussian_logpdf(action, mean, logvar)  # (B, T)
        log_weight = -log_density.sum(dim=1)
        weight = torch.exp(log_weight - log_weight.max())
        weight = weight / weight.mean().clamp(min=1e-8)
        if self.weight_clip > 0:
            lower, upper = torch.quantile(
                weight.detach(),
                torch.tensor(
                    [self.weight_clip / 100, 1 - self.weight_clip / 100],
                    device=weight.device,
                ),
            )
            weight = weight.clamp(float(lower), float(upper))
            weight = weight / weight.mean().clamp(min=1e-8)
        return weight.detach()

    # -- generator -----------------------------------------------------------

    def _decode(self, latent: Tensor, action: Tensor) -> Tensor:
        batch, time, _ = action.shape
        repeated = latent.unsqueeze(1).expand(-1, time, -1)
        features = torch.cat([repeated, action], dim=-1)
        out = self.decoder(features.reshape(batch * time, -1))
        return out.reshape(batch, time, self.observed_dim)

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        mean, logvar = self._propensity(batch.x, action)
        return {
            "action": action,
            "propensity_mean": mean,
            "propensity_logvar": logvar,
            "weight": self.iptw_weights(batch),
        }

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 2:
            new_action = new_action.unsqueeze(-1)
        horizon = noise["action"].shape[1]
        if new_action.shape[1] == 1 and horizon > 1:
            new_action = new_action.expand(-1, horizon, -1)
        return {"action": new_action.float(), "weight": noise["weight"]}

    def _sample_latent(self, x: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, time, _ = x.shape
        features = torch.cat([x, action], dim=-1)
        mean, logvar = self.encoder(features.reshape(-1, features.shape[-1]))
        mean = mean.reshape(batch, time, -1).mean(dim=1)
        logvar = logvar.reshape(batch, time, -1).mean(dim=1)
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(0.5 * logvar), mean, logvar

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        latent = context["latent"]
        action = context["action"][:, :horizon]
        return self._decode(latent, action)

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        noise = self.abduction(batch)
        context = self.action(noise, intervention)
        action = self._actions(batch)
        latent, _, _ = self._sample_latent(batch.x, action)
        return self._decode(latent, context["action"])

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
        latent, mean, logvar = self._sample_latent(batch.x, action)
        recon = self._decode(latent, action)
        mean_p, logvar_p = self._propensity(batch.x, action)
        return {
            "latent": latent,
            "encode_mean": mean,
            "encode_logvar": logvar,
            "propensity_mean": mean_p,
            "propensity_logvar": logvar_p,
            "recon": recon,
            "action": action,
            "weight": self.iptw_weights(batch),
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        # Weighted reconstruction likelihood (Proposition 1).
        weighted_params = self.obs_logvar
        per_sample = 0.5 * (
            (outputs["recon"] - batch.x) ** 2 * torch.exp(-weighted_params)
            + weighted_params
            + torch.log(torch.tensor(2 * torch.pi))
        ).mean(dim=(1, 2))
        recon = (outputs["weight"] * per_sample).sum() / outputs["weight"].sum().clamp(min=1e-8)
        kl = 0.5 * (
            torch.exp(outputs["encode_logvar"])
            + outputs["encode_mean"] ** 2
            - 1.0
            - outputs["encode_logvar"]
        ).sum(dim=1)
        kl = (outputs["weight"] * kl).sum() / outputs["weight"].sum().clamp(min=1e-8)
        # Propensity maximum-likelihood (unweighted, fitted on observed data).
        propensity_nll = -gaussian_logpdf(
            outputs["action"], outputs["propensity_mean"], outputs["propensity_logvar"]
        ).mean()
        total = recon + kl + propensity_nll
        return {"loss": total, "recon": recon, "kl": kl, "propensity_nll": propensity_nll}

    def identifiability_statement(self) -> str:
        return (
            "Wu-IPTW: under consistency, positivity and sequential strong "
            "ignorability, the IPTW-weighted conditional likelihood targets the "
            "g-formula density f_a(y) (Proposition 1); no outcome plug-in model is "
            "used, so identification is only as good as the propensity model."
        )

    def causal_level(self) -> str:
        return CausalLevel.INTERVENTION
