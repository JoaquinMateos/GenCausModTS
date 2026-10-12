"""CausalGAN: causal implicit generative models with adversarial training.

Reference: Kocaoglu et al., *CausalGAN: Learning Causal Implicit Generative
Models with Adversarial Training*, ICLR 2018 — BibTeX
``kocaogluCausalGANLearningCausal2017``.

Two stages are trained jointly in our adaptation:

1. a *causal controller* generating binary labels in the causal order of a given
   DAG, ``l_i = f_i(Pa(l_i), z_i)`` (node mechanisms with independent noise);
2. a conditional generator ``G(z, l)`` producing observations, with a
   real/fake discriminator and *Labeler* / *Anti-Labeler* heads that estimate the
   label posterior on real and generated samples.

An intervention ``do(l_i = v)`` overwrites the target label and re-propagates it
to its descendants through the controller mechanisms; conditioning leaves the
graph untouched. Identification follows from C being a causal implicit
generative model for the given DAG plus a class-conditional generator
(Corollary 2 / Proposition 1), assuming causal sufficiency.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gcmts.core.backbones import MLP, GaussianHead, gaussian_reconstruction_nll
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CausalLevel, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["CausalGAN"]


class CausalGAN(BaseEffectEstimator):
    """Causal implicit generative model with adversarial training.

    Args:
        observed_dim: observation dimension ``D``.
        n_labels: number ``K`` of binary labels/causal variables.
        treatment_index: index of the intervention-target label.
        noise_dim: generator exogenous-noise dimension.
        controller_noise_dim: per-node controller noise dimension.
        adversarial_weight: weight of the adversarial term.
        labeler_weight: weight of the Labeler/Anti-Labeler terms.
        hidden_dims: network widths.
        obs_noise: observation-noise std for the reconstruction NLL.
        adjacency: ``(K, K)`` DAG over labels (``adjacency[j, i] = 1`` means
            ``j`` is a parent of ``i``); may be supplied per batch instead.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        n_labels: int = 3,
        treatment_index: int = -1,
        noise_dim: int = 4,
        controller_noise_dim: int = 2,
        adversarial_weight: float = 0.1,
        labeler_weight: float = 0.1,
        hidden_dims: tuple[int, ...] = (128, 128),
        obs_noise: float = 0.05,
        adjacency: Tensor | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=1)
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        self.n_labels = n_labels
        self.treatment_index = treatment_index if treatment_index >= 0 else n_labels - 1
        self.noise_dim = noise_dim
        self.adversarial_weight = adversarial_weight
        self.labeler_weight = labeler_weight
        if adjacency is None:
            adjacency = torch.tril(torch.ones(n_labels, n_labels), diagonal=-1)
        self.register_buffer("adjacency", adjacency.float())
        self.adjacency: Tensor

        self.controller = nn.ModuleList(
            MLP(n_labels + controller_noise_dim, 1, hidden_dims=hidden_dims)
            for _ in range(n_labels)
        )
        self.encoder = GaussianHead(observed_dim, noise_dim, hidden_dims=hidden_dims)
        self.generator = MLP(noise_dim + n_labels, observed_dim, hidden_dims=hidden_dims)
        self.discriminator = MLP(observed_dim + n_labels, 1, hidden_dims=hidden_dims)
        self.labeler = MLP(observed_dim, n_labels, hidden_dims=hidden_dims)
        self.anti_labeler = MLP(observed_dim, n_labels, hidden_dims=hidden_dims)
        self.register_buffer(
            "obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone()
        )
        self.obs_logvar: Tensor
        logger.debug("Initialised CausalGAN (D=%d, K=%d)", observed_dim, n_labels)

    # -- label / treatment handling -----------------------------------------

    def _labels(self, batch: Batch) -> Tensor:
        x = batch.x
        if batch.context is not None and "L" in batch.context:
            labels = batch.context["L"].float()
            if labels.ndim == 3:
                labels = labels[:, 0]
            return labels.to(x.device)
        return torch.zeros(x.shape[0], self.n_labels, device=x.device)

    def _adjacency(self, batch: Batch) -> Tensor:
        if batch.context is not None and "adjacency" in batch.context:
            return batch.context["adjacency"].to(batch.x.device)
        return self.adjacency

    def _controller_labels(
        self,
        noise: Tensor,
        adjacency: Tensor,
        override: Tensor | None = None,
        override_value: Tensor | None = None,
    ) -> Tensor:
        """Generate labels in topological order, optionally with an intervention."""
        batch = noise.shape[0]
        labels = torch.zeros(batch, self.n_labels, device=noise.device)
        for i in range(self.n_labels):
            if override is not None and bool(override[i] > 0.5):
                labels[:, i] = (override_value[:, 0] if override_value is not None else 0.0)
                continue
            parents = labels * adjacency[:, i].unsqueeze(0)
            features = torch.cat([parents, noise[:, i]], dim=-1)
            logits = self.controller[i](features).squeeze(-1)
            labels[:, i] = torch.sigmoid(logits)
        return labels

    def _base_labels(self, batch: Batch) -> Tensor:
        return self._labels(batch)

    # -- generator -----------------------------------------------------------

    def _generate(self, noise: Tensor, labels: Tensor) -> Tensor:
        batch, time, _ = noise.shape
        repeated = labels.unsqueeze(1).expand(-1, time, -1)
        features = torch.cat([noise, repeated], dim=-1)
        out = self.generator(features.reshape(batch * time, -1))
        return out.reshape(batch, time, self.observed_dim)

    def _discriminate(self, x: Tensor, labels: Tensor) -> Tensor:
        batch, time, _ = x.shape
        repeated = labels.unsqueeze(1).expand(-1, time, -1)
        features = torch.cat([x, repeated], dim=-1)
        logits = self.discriminator(features.reshape(batch * time, -1)).reshape(batch, time)
        return logits.mean(dim=1)

    # -- BaseEffectEstimator contract ---------------------------------------

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        mean, _ = self.encoder(batch.x.reshape(-1, self.observed_dim))
        noise = mean.reshape(batch.x.shape[0], batch.x.shape[1], -1)
        return {"noise": noise, "labels": self._base_labels(batch)}

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        value = intervention["A"]
        if value.ndim == 2:
            value = value.reshape(-1, 1)
        labels = noise["labels"].clone()
        labels[:, self.treatment_index] = value[:, 0]
        return {"noise": noise["noise"], "labels": labels}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        return self._generate(context["noise"][:, :horizon], context["labels"])

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
        adjacency = self._adjacency(batch)
        labels = self._labels(batch)
        mean, logvar = self.encoder(batch.x.reshape(-1, self.observed_dim))
        noise = mean.reshape(batch.x.shape[0], batch.x.shape[1], -1)
        fake = self._generate(noise, labels)
        controller_noise = torch.randn(
            batch.x.shape[0], self.n_labels, self.controller[0].net[0].in_features - self.n_labels
        )
        generated_labels = self._controller_labels(controller_noise.to(batch.x.device), adjacency)
        return {
            "noise": noise,
            "logvar": logvar,
            "recon": fake,
            "labels": labels,
            "generated_labels": generated_labels,
            "real_logits": self._discriminate(batch.x, labels),
            "fake_logits": self._discriminate(fake, labels),
            "labeler_real": self.labeler(batch.x.reshape(-1, self.observed_dim)),
            "labeler_fake": self.labeler(fake.reshape(-1, self.observed_dim)),
            "anti_labeler_fake": self.anti_labeler(fake.reshape(-1, self.observed_dim)),
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        ones = torch.ones_like(outputs["real_logits"])
        zeros = torch.zeros_like(outputs["fake_logits"])
        generator_adv = F.binary_cross_entropy_with_logits(outputs["fake_logits"], ones)
        discriminator = 0.5 * (
            F.binary_cross_entropy_with_logits(outputs["real_logits"], ones)
            + F.binary_cross_entropy_with_logits(outputs["fake_logits"].detach(), zeros)
        )
        labels = outputs["labels"]
        time = batch.x.shape[1]
        expanded = labels.repeat_interleave(time, dim=0)
        labeler_real = F.binary_cross_entropy_with_logits(outputs["labeler_real"], expanded)
        labeler_fake = F.binary_cross_entropy_with_logits(outputs["labeler_fake"], expanded)
        anti_labeler = F.binary_cross_entropy_with_logits(
            outputs["anti_labeler_fake"], outputs["anti_labeler_fake"].sigmoid()
        )
        # Controller: match the generated label marginals to the observed ones.
        controller = F.mse_loss(outputs["generated_labels"].mean(dim=0), labels.mean(dim=0))
        labeler = 0.5 * (labeler_real + labeler_fake)
        total = (
            recon
            + self.adversarial_weight * (generator_adv + discriminator)
            + self.labeler_weight * (labeler + anti_labeler)
            + 0.1 * controller
        )
        return {
            "loss": total,
            "recon": recon,
            "gen_adv": generator_adv,
            "disc": discriminator,
            "labeler": labeler,
        }

    def identifiability_statement(self) -> str:
        return (
            "CausalGAN: given a DAG, the causal controller is a causal implicit "
            "generative model for it and a class-conditional generator yields "
            "observational and interventional laws (Corollary 2/Proposition 1), "
            "assuming causal sufficiency; the guarantee is only as good as the "
            "assumed graph."
        )

    def causal_level(self) -> str:
        return CausalLevel.INTERVENTION
