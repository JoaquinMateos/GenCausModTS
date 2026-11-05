# Adapted-source provenance:
# Paper: Zhou et al., Variational GAN Enhanced Causal Effect Inference, 2026.
# Official implementation: none used; this is a paper-based adaptation.
# Status: compact surrogate.

"""VGAN-CHG: variational GAN with global/regional latent confounders.

Compact surrogate for Zhou et al., *Variational GAN-Enhanced Causal Effect
Inference for Interpretable Learning in Heterogeneous IoT Systems*, IEEE TCE
72(2):2668--2685, 2026, DOI 10.1109/TCE.2026.3673867 (no public code).

Kept: the two-level Gaussian confounder (global sequence + pooled sliding
windows), the BVAE decoder for ``(x, y)``, the conditional GAN head
``G(x, t, y_f, h, z_g, z_r, 1-t)`` with a discriminator over ``(x, y)`` only,
and a propensity-weighted doubly-robust loss with clipping.  The graph
machinery is omitted; the outcome is the last coordinate and the treatment is
read from ``batch.context["A"]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from gcmts.core.backbones import (
    MLP,
    GaussianHead,
    gaussian_kl,
    gaussian_reconstruction_nll,
    reparameterise,
)
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

__all__ = ["VGANCHG"]


class VGANCHG(BaseEffectEstimator):
    """Variational GAN with global and regional latent confounders.

    ``observed_dim`` counts covariates plus the outcome channel (last).
    ``max_len`` is the sequence length stored by the global encoder (longer
    trajectories are cropped, shorter ones zero-padded).  The DR term assumes a
    binary, scalar treatment; ``propensity_clip`` bounds the propensity score.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        window: int = 3,
        latent_global: int = 8,
        latent_regional: int = 4,
        max_len: int = 32,
        hidden_dims: tuple[int, ...] = (128, 128),
        hidden_dim: int = 32,
        adversarial_weight: float = 1.0,
        dr_weight: float = 1.0,
        kl_weight: float = 1.0,
        recon_weight: float = 1.0,
        propensity_clip: float = 0.05,
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        if observed_dim < 2:
            raise ValueError("observed_dim must include at least one covariate and the outcome.")
        if window < 1:
            raise ValueError("window must be positive.")
        if not 0.0 < propensity_clip < 0.5:
            raise ValueError("propensity_clip must lie in (0, 0.5).")
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        self.covariate_dim = observed_dim - 1
        self.window = window
        self.latent_global = latent_global
        self.latent_regional = latent_regional
        self.max_len = max(max_len, window)
        self.adversarial_weight = adversarial_weight
        self.dr_weight = dr_weight
        self.recon_weight = recon_weight
        self.generator_factual_weight = 1.0
        self.kl_weight = kl_weight
        self.propensity_clip = propensity_clip

        # Two-level confounder encoders.
        self.global_encoder = GaussianHead(
            observed_dim * self.max_len, latent_global, hidden_dims=hidden_dims
        )
        self.regional_encoder = GaussianHead(
            observed_dim * window, latent_regional, hidden_dims=hidden_dims
        )
        # BVAE decoder: latents and treatment -> covariates and factual outcome.
        self.decoder = MLP(
            latent_global + latent_regional + treatment_dim,
            observed_dim,
            hidden_dims=hidden_dims,
        )
        # Propensity e(x) = P(A = 1 | x), a scalar logit per trajectory.
        self.propensity = MLP(self.covariate_dim, 1, hidden_dims=hidden_dims)
        # Shared factual representation used by the counterfactual head.
        self.hidden = MLP(
            self.covariate_dim + treatment_dim + latent_global + latent_regional,
            hidden_dim,
            hidden_dims=hidden_dims,
        )
        # Conditional generator: (x, t, h, z_g, z_r) -> y.  The factual outcome
        # is deliberately not an input: conditioning on it lets the generator
        # copy the factual value and ignore the treatment, which collapses the
        # counterfactual estimate to y_f (the failure mode we observed).
        self.generator = MLP(
            self.covariate_dim
            + treatment_dim
            + hidden_dim
            + latent_global
            + latent_regional,
            1,
            hidden_dims=hidden_dims,
        )
        # Discriminator over (x, y); the original paper does not condition on t.
        self.discriminator = MLP(observed_dim, 1, hidden_dims=hidden_dims)
        self.register_buffer("obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone())
        self.obs_logvar: Tensor

    # -- batch helpers ---------------------------------------------------
    def _actions(self, batch: Batch) -> Tensor:
        action = None if batch.context is None else batch.context.get("A")
        if action is None:
            action = torch.zeros(
                batch.x.shape[0],
                batch.x.shape[1],
                self.treatment_dim,
                device=batch.x.device,
                dtype=batch.x.dtype,
            )
        return self._broadcast_action(action, batch.x.shape[1], batch.x)

    def _broadcast_action(self, action: Tensor, time: int, ref: Tensor) -> Tensor:
        action = action.to(device=ref.device, dtype=ref.dtype)
        if action.ndim == 1:
            action = action[:, None, None]
        elif action.ndim == 2:
            if action.shape[1] == self.treatment_dim:
                action = action[:, None, :]
            else:
                action = action[..., None]
        if action.ndim != 3:
            raise ValueError("Treatment must have shape (B,), (B, A) or (B, T, A).")
        if action.shape[1] == 1 and time > 1:
            action = action.expand(-1, time, -1)
        if action.shape[1] != time:
            action = action[:, :time]
        return action.contiguous()

    def _target(self, batch: Batch) -> Tensor:
        if batch.target is None:
            return batch.x
        target = batch.target
        if target.ndim == 3:
            return target
        if target.ndim not in (1, 2):
            raise ValueError("batch.target must have shape (B,), (B, T) or (B, T, D).")
        outcome = target.unsqueeze(-1) if target.ndim == 2 else target[:, None, None]
        outcome = outcome.expand(-1, batch.x.shape[1], -1)
        return torch.cat([batch.x[..., :-1], outcome], dim=-1)

    # -- encoders / decoder ----------------------------------------------
    def _encode(self, x: Tensor) -> dict[str, Tensor]:
        batch, time, dim = x.shape
        # Global posterior over the whole (padded/cropped) sequence.
        global_input = (
            F.pad(x, (0, 0, 0, self.max_len - time))
            if time < self.max_len
            else x[:, : self.max_len]
        )
        g_mean, g_logvar = self.global_encoder(global_input.reshape(batch, -1))

        # Regional posterior: one Gaussian per sliding window, then pooled.
        regional_input = x if time >= self.window else F.pad(x, (0, 0, 0, self.window - time))
        windows = regional_input.unfold(1, self.window, 1).transpose(2, 3)
        n_windows = windows.shape[1]
        windows = windows.reshape(batch, n_windows, self.window * dim)
        r_mean, r_logvar = self.regional_encoder(
            windows.reshape(batch * n_windows, self.window * dim)
        )
        r_mean = r_mean.reshape(batch, n_windows, self.latent_regional).mean(dim=1)
        r_logvar = r_logvar.reshape(batch, n_windows, self.latent_regional).mean(dim=1)
        return {
            "z_global": reparameterise(g_mean, g_logvar),
            "z_regional": reparameterise(r_mean, r_logvar),
            "global_mean": g_mean,
            "global_logvar": g_logvar,
            "regional_mean": r_mean,
            "regional_logvar": r_logvar,
        }

    def _propensity_logit(self, x: Tensor) -> Tensor:
        summary = x[..., : self.covariate_dim].mean(dim=1)
        return self.propensity(summary).reshape(x.shape[0], 1)

    def _decode(self, z_global: Tensor, z_regional: Tensor, action: Tensor) -> Tensor:
        """Decode ``(x_hat, y_hat_f)`` for every timestep."""
        batch, time, _ = action.shape
        g = z_global.unsqueeze(1).expand(-1, time, -1)
        r = z_regional.unsqueeze(1).expand(-1, time, -1)
        features = torch.cat([g, r, action], dim=-1)
        decoded = self.decoder(features.reshape(batch * time, -1))
        return decoded.reshape(batch, time, self.observed_dim)

    def _counterfactual_outcome(
        self,
        covariates: Tensor,
        z_global: Tensor,
        z_regional: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Generate the outcome under ``action`` with the conditional GAN head.

        The head is a conditional outcome model ``G(x, a, z) -> y``; it is
        trained on the factual pairs and queried with the opposite action to
        produce the counterfactual.  The factual outcome is not an input.
        """
        batch, time, _ = covariates.shape
        g = z_global.unsqueeze(1).expand(-1, time, -1)
        r = z_regional.unsqueeze(1).expand(-1, time, -1)
        h = self.hidden(
            torch.cat([covariates, action, g, r], dim=-1).reshape(batch * time, -1)
        ).reshape(batch, time, -1)
        features = torch.cat([covariates, action, h, g, r], dim=-1)
        generated = self.generator(features.reshape(batch * time, -1))
        return generated.reshape(batch, time, 1).clamp(-10.0, 10.0)

    # -- BaseEffectEstimator --------------------------------------------
    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        x = batch.x
        action = self._actions(batch)
        latents = self._encode(x)
        z_global, z_regional = latents["z_global"], latents["z_regional"]
        recon = self._decode(z_global, z_regional, action)
        # Condition the counterfactual head on the observed factual outcome, not
        # on its reconstruction: the observed outcome is available at query time
        # for counterfactual inference, and using the reconstruction adds the
        # decoder error to every counterfactual.
        factual_outcome = self._target(batch)[..., -1:]
        return {
            **latents,
            "action": action,
            "recon": recon,
            "y_hat_f": recon[..., -1:],
            "factual_outcome": factual_outcome,
            "propensity_logit": self._propensity_logit(x),
            "x_covariates": x[..., :-1],
        }

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        time = noise["action"].shape[1]
        new_action = self._broadcast_action(intervention["A"], time, noise["action"])
        return {**noise, "action": new_action}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        time = min(horizon, context["action"].shape[1])
        action = context["action"][:, :time]
        covariates = context["x_covariates"][:, :time]
        y_cf = self._counterfactual_outcome(
            covariates,
            context["z_global"],
            context["z_regional"],
            action,
        )
        return torch.cat([covariates, y_cf], dim=-1)

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

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        x = batch.x
        target = self._target(batch)
        latents = self._encode(x)
        z_global, z_regional = latents["z_global"], latents["z_regional"]
        recon = self._decode(z_global, z_regional, action)
        propensity_logit = self._propensity_logit(x)
        # The generator is trained on the factual action (so it learns the
        # outcome surface) and queried with the opposite action for the CF.
        y_gen_factual = self._counterfactual_outcome(
            x[..., :-1], z_global, z_regional, action
        )
        y_cf = self._counterfactual_outcome(
            x[..., :-1], z_global, z_regional, 1.0 - action
        )
        return {
            **latents,
            "action": action,
            "x_input": x,
            "target": target,
            "recon": recon,
            "y_hat_f": recon[..., -1:],
            "y_cf": y_cf,
            "y_gen_factual": y_gen_factual,
            "propensity_logit": propensity_logit,
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        target = outputs["target"]
        y_obs = target[..., -1:]
        y_hat_f, y_cf = outputs["y_hat_f"], outputs["y_cf"]
        reconstruction = gaussian_reconstruction_nll(outputs["recon"], target, self.obs_logvar)
        zeros_g = torch.zeros_like(outputs["global_mean"])
        zeros_r = torch.zeros_like(outputs["regional_mean"])
        kl_global = gaussian_kl(
            outputs["global_mean"], outputs["global_logvar"], zeros_g, zeros_g
        ).mean()
        kl_regional = gaussian_kl(
            outputs["regional_mean"], outputs["regional_logvar"], zeros_r, zeros_r
        ).mean()
        kl = kl_global + kl_regional

        # Conditional GAN: D sees the detached fake pair, G sees the live one.
        covariates = outputs["x_input"][..., :-1]
        real_pair = torch.cat([covariates, y_obs], dim=-1)
        fake_pair = torch.cat([covariates, outputs["y_gen_factual"]], dim=-1)
        real_logits = self.discriminator(real_pair)
        fake_logits_d = self.discriminator(fake_pair.detach())
        d_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        d_fake = F.binary_cross_entropy_with_logits(fake_logits_d, torch.zeros_like(fake_logits_d))
        gan_d = 0.5 * (d_real + d_fake)
        gan_g = F.binary_cross_entropy_with_logits(
            self.discriminator(fake_pair), torch.ones_like(real_logits)
        )

        # Propensity-weighted doubly-robust outcome loss.
        treatment = outputs["action"][..., :1]
        propensity = torch.sigmoid(outputs["propensity_logit"]).clamp(
            self.propensity_clip, 1.0 - self.propensity_clip
        )
        propensity = propensity.unsqueeze(1).expand_as(y_obs)
        mu1 = torch.where(treatment > 0.5, y_hat_f, y_cf)
        mu0 = torch.where(treatment > 0.5, y_cf, y_hat_f)
        dr_prediction = (
            y_hat_f
            + treatment * (y_obs - mu1) / propensity
            + (1.0 - treatment) * (y_obs - mu0) / (1.0 - propensity)
        )
        dr = F.mse_loss(dr_prediction, y_obs)
        gen_factual = F.mse_loss(outputs["y_gen_factual"], y_obs)
        total = (
            self.recon_weight * reconstruction
            + float(self.kl_weight) * kl
            + self.adversarial_weight * (gan_d + gan_g)
            + self.dr_weight * dr
            + self.generator_factual_weight * gen_factual
        )
        return {
            "loss": total,
            "recon": reconstruction,
            "kl": kl,
            "kl_global": kl_global,
            "kl_regional": kl_regional,
            "gan_d": gan_d,
            "gan_g": gan_g,
            "dr": dr,
            "gen_factual": gen_factual,
        }

    def identifiability_statement(self) -> str:
        return (
            "VGAN-CHG (compact surrogate): identification is empirical only. The "
            "two-level Gaussian posteriors, the conditional GAN counterfactual "
            "head and the propensity-weighted doubly-robust loss follow the "
            "paper's construction, but no identifiability theorem is claimed; "
            "counterfactuals rely on consistency, positivity (enforced by "
            "propensity clipping) and the no-unmeasured-confounder assumption "
            "encoded by the global/regional latents."
        )
