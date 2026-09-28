"""Static-dynamic disentanglers for evolving-domain generalisation.

* :class:`SYNC` — four-way split into static/dynamic causal and spurious factors,
  with the outcome head restricted to the causal pair and a domain-adversarial
  penalty on the causal factors (He et al., 2025, BibTeX
  ``heLearningTimeAwareCausal2025``).
* :class:`ERM` — an empirical-risk-minimisation baseline that predicts from the
  full representation, illustrating why a non-causal predictor fails out of
  distribution.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor, nn

from gcmts.core.backbones import MLP, GaussianHead, GradientReversal, gaussian_reconstruction_nll
from gcmts.static_dynamic_disentanglement.base import BaseStaticDynamicDisentangler
from gcmts.typing import Batch, DisentangledFactors

logger = logging.getLogger(__name__)

__all__ = ["SYNC", "ERM"]


class _FactorExtractor(nn.Module):
    """Per-step encoder producing the four factor groups."""

    def __init__(
        self,
        *,
        observed_dim: int,
        static_dim: int,
        dynamic_dim: int,
        spurious_static_dim: int,
        spurious_dynamic_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.spurious_static_dim = spurious_static_dim
        self.spurious_dynamic_dim = spurious_dynamic_dim
        per_step = static_dim + dynamic_dim + spurious_static_dim + spurious_dynamic_dim
        self.encoder = GaussianHead(observed_dim, per_step, hidden_dims=hidden_dims)
        self.decoder = MLP(per_step, observed_dim, hidden_dims=hidden_dims)

    def forward(self, x: Tensor) -> DisentangledFactors:
        batch, time, _ = x.shape
        encoded = cast(tuple[Tensor, Tensor], self.encoder(x.reshape(-1, x.shape[-1])))
        mean = encoded[0].reshape(batch, time, -1)
        idx = 0
        z_stc = mean[:, :, idx : idx + self.static_dim].mean(dim=1, keepdim=True)
        idx += self.static_dim
        z_dyc = mean[:, :, idx : idx + self.dynamic_dim]
        idx += self.dynamic_dim
        z_sts = mean[:, :, idx : idx + self.spurious_static_dim].mean(dim=1, keepdim=True)
        idx += self.spurious_static_dim
        z_dys = mean[:, :, idx : idx + self.spurious_dynamic_dim]
        return DisentangledFactors(z_stc=z_stc, z_dyc=z_dyc, z_sts=z_sts, z_dys=z_dys)

    def decode(self, factors: DisentangledFactors) -> Tensor:
        parts = [factors.z_stc.expand(-1, factors.z_dyc.shape[1], -1), factors.z_dyc]
        if factors.z_sts is not None:
            parts.append(factors.z_sts.expand(-1, factors.z_dyc.shape[1], -1))
        if factors.z_dys is not None:
            parts.append(factors.z_dys)
        latent = torch.cat(parts, dim=-1)
        batch, time, dim = latent.shape
        decoded = cast(Tensor, self.decoder(latent.reshape(batch * time, dim)))
        return decoded.reshape(batch, time, -1)


class _BaseDisentangler(BaseStaticDynamicDisentangler):
    def __init__(
        self,
        *,
        observed_dim: int,
        latent_static_dim: int,
        latent_dynamic_dim: int,
        latent_spurious_static_dim: int,
        latent_spurious_dynamic_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(
            observed_dim=observed_dim,
            latent_static_dim=latent_static_dim,
            latent_dynamic_dim=latent_dynamic_dim,
            latent_spurious_static_dim=latent_spurious_static_dim,
            latent_spurious_dynamic_dim=latent_spurious_dynamic_dim,
        )
        self.extractor = _FactorExtractor(
            observed_dim=observed_dim,
            static_dim=latent_static_dim,
            dynamic_dim=latent_dynamic_dim,
            spurious_static_dim=latent_spurious_static_dim,
            spurious_dynamic_dim=latent_spurious_dynamic_dim,
            hidden_dims=hidden_dims,
        )
        self.register_buffer("obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone())
        self.obs_logvar: Tensor

    def encode_static(self, x: Tensor) -> Tensor:
        return cast(DisentangledFactors, self.extractor(x)).z_stc

    def encode_dynamic(self, x: Tensor) -> Tensor:
        return cast(DisentangledFactors, self.extractor(x)).z_dyc

    def disentangle(self, x: Tensor) -> DisentangledFactors:
        return cast(DisentangledFactors, self.extractor(x))

    def decode(self, factors: DisentangledFactors) -> Tensor:
        return self.extractor.decode(factors)

    def predict(self, x: Tensor, target_domain: Tensor | None = None) -> Tensor:
        return self.decode(self.extractor(x))

    def ood_score(self, x: Tensor, target_domain: Tensor | None = None) -> Tensor:
        return (self.decode(self.extractor(x)) - x).pow(2).mean(dim=(1, 2))

    def latent_factors(self, x: Tensor) -> Tensor:
        factors = self.extractor(x)
        parts = [factors.z_stc.expand(-1, factors.z_dyc.shape[1], -1), factors.z_dyc]
        if factors.z_sts is not None:
            parts.append(factors.z_sts.expand(-1, factors.z_dyc.shape[1], -1))
        if factors.z_dys is not None:
            parts.append(factors.z_dys)
        return torch.cat(parts, dim=-1)


class SYNC(_BaseDisentangler):
    """Four-way static-dynamic disentangler with a causal outcome head."""

    def __init__(
        self,
        *,
        observed_dim: int,
        static_dim: int = 2,
        dynamic_dim: int = 2,
        spurious_static_dim: int = 1,
        spurious_dynamic_dim: int = 1,
        n_domains: int = 5,
        hidden_dims: tuple[int, ...] = (128, 128),
        adversarial_weight: float = 5.0,
        outcome_weight: float = 20.0,
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(
            observed_dim=observed_dim,
            latent_static_dim=static_dim,
            latent_dynamic_dim=dynamic_dim,
            latent_spurious_static_dim=spurious_static_dim,
            latent_spurious_dynamic_dim=spurious_dynamic_dim,
            hidden_dims=hidden_dims,
            obs_noise=obs_noise,
        )
        self.n_domains = n_domains
        self.adversarial_weight = adversarial_weight
        self.outcome_weight = outcome_weight
        causal_dim = static_dim + dynamic_dim
        spurious_dim = spurious_static_dim + spurious_dynamic_dim
        self.outcome_head = MLP(causal_dim, 1, hidden_dims=hidden_dims)
        self.spurious_classifier = MLP(spurious_dim, n_domains, hidden_dims=hidden_dims)
        self.causal_classifier = MLP(causal_dim, n_domains, hidden_dims=hidden_dims)
        self.reversal = GradientReversal(scale=1.0)

    def _causal(self, factors: DisentangledFactors) -> Tensor:
        return torch.cat([factors.z_stc.mean(dim=1), factors.z_dyc.mean(dim=1)], dim=-1)

    def _spurious(self, factors: DisentangledFactors) -> Tensor:
        parts = []
        if factors.z_sts is not None:
            parts.append(factors.z_sts.mean(dim=1))
        if factors.z_dys is not None:
            parts.append(factors.z_dys.mean(dim=1))
        return torch.cat(parts, dim=-1)

    def predict_outcome(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.outcome_head(self._causal(self.extractor(x)))).squeeze(-1)

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        factors = self.extractor(batch.x)
        recon = self.extractor.decode(factors)
        causal = self._causal(factors)
        spurious = self._spurious(factors)
        return {
            "recon": recon,
            "outcome": self.outcome_head(causal).squeeze(-1),
            "spurious_logits": self.spurious_classifier(spurious),
            "causal_logits": self.causal_classifier(self.reversal(causal)),
            "factors": self.latent_factors(batch.x),
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        target = batch.outcome
        if target is None:
            raise ValueError("This method requires batch.outcome.")
        outcome = nn.functional.mse_loss(outputs["outcome"], target)
        domain = batch.domain
        if domain is None:
            domain = torch.zeros(batch.x.shape[0], dtype=torch.long, device=batch.x.device)
        spurious_ce = nn.functional.cross_entropy(outputs["spurious_logits"], domain)
        causal_ce = nn.functional.cross_entropy(outputs["causal_logits"], domain)
        total = (
            recon
            + self.outcome_weight * outcome
            + spurious_ce
            + self.adversarial_weight * causal_ce
        )
        return {
            "loss": total,
            "recon": recon,
            "outcome": outcome,
            "domain": spurious_ce,
            "adversarial": causal_ce,
        }

    def identifiability_statement(self) -> str:
        return (
            "SYNC: under mechanism drift, the optimal predictor depends only on the "
            "causal triple (static, dynamic, drift); removing domain information "
            "from the causal factors and confining the outcome to them makes the "
            "representation transfer across evolving domains."
        )


class DANN(_BaseDisentangler):
    """Domain-adversarial outcome predictor.

    The representation used for the outcome is trained to be domain-invariant
    (Ganin et al., 2015; the ``UDA`` sub-taxonomy). Because the causal factors
    are invariant by construction and the spurious ones are domain-specific,
    invariance alone discards the shortcut and transfers to unseen domains.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        static_dim: int = 2,
        dynamic_dim: int = 2,
        spurious_static_dim: int = 1,
        spurious_dynamic_dim: int = 1,
        n_domains: int = 5,
        hidden_dims: tuple[int, ...] = (128, 128),
        adversarial_weight: float = 5.0,
        outcome_weight: float = 20.0,
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(
            observed_dim=observed_dim,
            latent_static_dim=static_dim,
            latent_dynamic_dim=dynamic_dim,
            latent_spurious_static_dim=spurious_static_dim,
            latent_spurious_dynamic_dim=spurious_dynamic_dim,
            hidden_dims=hidden_dims,
            obs_noise=obs_noise,
        )
        self.n_domains = n_domains
        self.adversarial_weight = adversarial_weight
        self.outcome_weight = outcome_weight
        total = static_dim + dynamic_dim + spurious_static_dim + spurious_dynamic_dim
        self.outcome_head = MLP(total, 1, hidden_dims=hidden_dims)
        self.domain_classifier = MLP(total, n_domains, hidden_dims=hidden_dims)
        self.reversal = GradientReversal(scale=1.0)

    def _pool(self, x: Tensor) -> Tensor:
        return self.latent_factors(x).mean(dim=1)

    def predict_outcome(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.outcome_head(self._pool(x))).squeeze(-1)

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        factors = self.extractor(batch.x)
        recon = self.extractor.decode(factors)
        pooled = self.latent_factors(batch.x).mean(dim=1)
        return {
            "recon": recon,
            "outcome": self.outcome_head(pooled).squeeze(-1),
            "domain_logits": self.domain_classifier(self.reversal(pooled)),
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        target = batch.outcome
        if target is None:
            raise ValueError("This method requires batch.outcome.")
        outcome = nn.functional.mse_loss(outputs["outcome"], target)
        domain = batch.domain
        if domain is None:
            domain = torch.zeros(batch.x.shape[0], dtype=torch.long, device=batch.x.device)
        adversarial = nn.functional.cross_entropy(outputs["domain_logits"], domain)
        total = recon + self.outcome_weight * outcome + self.adversarial_weight * adversarial
        return {
            "loss": total,
            "recon": recon,
            "outcome": outcome,
            "adversarial": adversarial,
        }

    def identifiability_statement(self) -> str:
        return (
            "DANN: an outcome representation that is simultaneously predictive and "
            "domain-invariant must retain the causal factors and discard the "
            "domain-specific spurious ones, up to the invariance penalty."
        )


class VREx(_BaseDisentangler):
    """Variance-of-risk (VREx) domain-generalisation predictor.

    Reference: Krueger et al., *Out-of-Distribution Generalization via Risk
    Extrapolation*, ICML 2021. The per-domain risks of the outcome predictor are
    made uniform, so features whose relation to the target is unstable across
    domains are discarded. This is the correct inductive bias when the spurious
    *feature distribution* is domain-invariant but its *target relation* flips:
    domain-adversarial methods cannot detect such features, but risk variance can.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        static_dim: int = 2,
        dynamic_dim: int = 2,
        spurious_static_dim: int = 1,
        spurious_dynamic_dim: int = 1,
        hidden_dims: tuple[int, ...] = (128, 128),
        variance_weight: float = 100.0,
        outcome_weight: float = 20.0,
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(
            observed_dim=observed_dim,
            latent_static_dim=static_dim,
            latent_dynamic_dim=dynamic_dim,
            latent_spurious_static_dim=spurious_static_dim,
            latent_spurious_dynamic_dim=spurious_dynamic_dim,
            hidden_dims=hidden_dims,
            obs_noise=obs_noise,
        )
        self.variance_weight = variance_weight
        self.outcome_weight = outcome_weight
        total = static_dim + dynamic_dim + spurious_static_dim + spurious_dynamic_dim
        self.outcome_head = MLP(total, 1, hidden_dims=hidden_dims)

    def predict_outcome(self, x: Tensor) -> Tensor:
        pooled = self.latent_factors(x).mean(dim=1)
        return cast(Tensor, self.outcome_head(pooled)).squeeze(-1)

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        factors = self.extractor(batch.x)
        recon = self.extractor.decode(factors)
        pooled = self.latent_factors(batch.x).mean(dim=1)
        return {
            "recon": recon,
            "outcome": self.outcome_head(pooled).squeeze(-1),
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        target = batch.outcome
        if target is None:
            raise ValueError("VREx requires batch.outcome.")
        squared = (outputs["outcome"] - target) ** 2
        domain = batch.domain
        if domain is None:
            risk = squared.mean()
            variance = torch.zeros((), device=squared.device)
        else:
            domains = domain.reshape(-1)
            risks = torch.stack(
                [squared[domains == d].mean() for d in torch.unique(domains)]
            )
            risk = risks.mean()
            variance = risks.var(unbiased=False) if risks.numel() > 1 else torch.zeros(())
        total = recon + self.outcome_weight * risk + self.variance_weight * variance
        return {"loss": total, "recon": recon, "risk": risk, "variance": variance}

    def identifiability_statement(self) -> str:
        return (
            "VREx: minimising the variance of per-environment risk selects the "
            "outcome representation whose risk is invariant across environments, "
            "which excludes spurious factors with domain-varying target relations."
        )


class ERM(_BaseDisentangler):
    """Empirical-risk-minimisation baseline: predicts from the full representation."""

    def __init__(
        self,
        *,
        observed_dim: int,
        static_dim: int = 2,
        dynamic_dim: int = 2,
        spurious_static_dim: int = 1,
        spurious_dynamic_dim: int = 1,
        hidden_dims: tuple[int, ...] = (128, 128),
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(
            observed_dim=observed_dim,
            latent_static_dim=static_dim,
            latent_dynamic_dim=dynamic_dim,
            latent_spurious_static_dim=spurious_static_dim,
            latent_spurious_dynamic_dim=spurious_dynamic_dim,
            hidden_dims=hidden_dims,
            obs_noise=obs_noise,
        )
        total = static_dim + dynamic_dim + spurious_static_dim + spurious_dynamic_dim
        self.outcome_head = MLP(total, 1, hidden_dims=hidden_dims)

    def predict_outcome(self, x: Tensor) -> Tensor:
        pooled = self.latent_factors(x).mean(dim=1)
        return cast(Tensor, self.outcome_head(pooled)).squeeze(-1)

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        factors = self.extractor(batch.x)
        recon = self.extractor.decode(factors)
        pooled = self.latent_factors(batch.x).mean(dim=1)
        return {
            "recon": recon,
            "outcome": self.outcome_head(pooled).squeeze(-1),
            "factors": self.latent_factors(batch.x),
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        target = batch.outcome
        if target is None:
            raise ValueError("This method requires batch.outcome.")
        outcome = nn.functional.mse_loss(outputs["outcome"], target)
        total = recon + outcome
        return {"loss": total, "recon": recon, "outcome": outcome}

    def identifiability_statement(self) -> str:
        return "ERM baseline: no causal guarantee; expected to fail out of distribution."
