"""NCTRL, adapted from the official implementation.

Paper: Song et al., *Temporally Disentangled Representation Learning under
Unknown Nonstationarity* (NeurIPS 2023), BibTeX
``songTemporallyDisentangledRepresentation2023``.
Official source: https://github.com/xiangchensong/nctrl (MIT license).
The official ARHMM, nonparametric change prior and VAE are used; this file
adapts the batch/optimizer interface to ``gcmts``.

Fidelity notes for the shared benchmark (``observed_dim != latent_dim``):

* The paper's ARHMM models the conditional emission ``p(x_t | x_{<t}, c_t)``
  of the *observations* (Eq. 5 and Appendix B.8), and the decoded regime
  sequence ``c`` conditions the change prior on ``z``.  The vendored
  ``NCTRL`` sizes its ARHMM with ``z_dim`` only because the authors' synthetic
  experiments use ``x_dim == z_dim``; here the same official HMM is built at
  the observation width and fed observations (``mle_scaled:H``, the official
  config, i.e. features from ``x_{<t}`` only).
* ``loss`` is a pure function of ``forward``: the official step does not
  re-encode inside the loss, and resampling ``z`` there changes the objective.
* The KL term is ``log q(z|x) - log p(z)`` (Eq. 26); it is a positive
  divergence the optimiser minimises.
* The ARHMM keeps its own Adam optimiser, as in the official ``NCTRL``
  ``configure_optimizers``/``training_step``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput
from gcmts.vendor.nctrl.models.hmm import HMM
from gcmts.vendor.nctrl.models.simulation import NCTRL

logger = logging.getLogger(__name__)

__all__ = ["NCTRLOfficial"]


class NCTRLOfficial(BaseCausalRepresentationLearner):
    """Interface around the official NCTRL ARHMM model."""

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        n_regimes: int = 3,
        hidden_dim: int = 128,
        beta: float = 0.0025,
        gamma: float = 0.0075,
        lr: float = 1e-3,
        hmm_lr: float = 1e-3,
        hmm_mode: str = "mle_scaled:H",
        transition_hidden: int = 128,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 0.01,
        obs_noise: float = 1.0,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        del transition_hidden, hidden_dims, decoder_hidden, kl_weight, obs_noise
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=1,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        self.trainer_lr = float(lr)
        self.hmm_mode = str(hmm_mode)
        input_dim = self.encoder_input_dim(observed_dim)
        self.core = NCTRL(
            x_dim=input_dim,
            z_dim=latent_dim,
            lags=1,
            n_class=max(int(n_regimes), 1),
            hidden_dim=int(hidden_dim),
            lr=float(lr),
            beta=float(beta),
            gamma=float(gamma),
            hmm_mode=self.hmm_mode,
        )
        # The vendored NCTRL sizes its ARHMM with ``z_dim`` because its own
        # synthetic experiments set ``x_dim == z_dim``.  In the paper the ARHMM
        # models the conditional emission ``p(x_t | x_{<t}, c_t)`` of the
        # *observations* (Eq. 5 and Appendix B.8), and the regime sequence it
        # decodes conditions the change prior on ``z``.  Rebuild the same
        # official HMM at the observation width and feed it observations below;
        # this is the only place where ``x_dim != z_dim`` needs handling.
        self.core.hmm = HMM(
            n_class=max(int(n_regimes), 1),
            lags=1,
            x_dim=input_dim,
            hidden_dim=int(hidden_dim),
            mode=self.hmm_mode,
        )
        self.hmm_optimizer = torch.optim.Adam(self.core.hmm.parameters(), lr=float(hmm_lr))
        self.core.log = lambda *args, **kwargs: None

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        """Exclude the ARHMM parameters from the shared model optimiser.

        The official NCTRL training step gives the ARHMM its own optimiser and
        likelihood, then updates the VAE separately.  The shared trainer builds
        a single optimiser from ``model.parameters()``; without this split the
        ARHMM would receive two conflicting updates per step.
        """
        for name, parameter in self.named_parameters(recurse=recurse):
            if not name.startswith("core.hmm."):
                yield parameter

    def _obs(self, x: Tensor) -> Tensor:
        if self.image_shape is None:
            return x
        features, (batch, time) = self.observation_features(x)
        return features.reshape(batch, time, -1)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        obs = self._obs(x)
        _, mean, _, _ = self.core.net(obs)
        return mean.reshape(obs.shape[0], obs.shape[1], self.latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])
        rec = self.core.net._decode(flat)
        if self.image_shape is not None:
            return rec.reshape(z.shape[0], z.shape[1], -1)
        return rec.reshape(z.shape[0], z.shape[1], self.observed_dim)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        obs = self._obs(batch.x)
        x_recon, mean, logvar, z = self.core.net(obs)
        # Numerical guards only; the official model has no clamp here.
        logvar = torch.clamp(logvar, -10.0, 10.0)
        mean = torch.clamp(mean, -10.0, 10.0)
        z = torch.clamp(z, -10.0, 10.0)
        return CausalRepresentationOutput(
            z=z,
            x_recon=x_recon,
            extras={"mean": mean, "logvar": logvar, "obs": obs},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        """Official NCTRL objective as a function of the forward pass.

        Mirrors ``vendor/nctrl/models/simulation.py::NCTRL.training_step``:
        the ARHMM is fit by its own observed-data log-likelihood (the forward
        algorithm), then the VAE is fit by ``recon + beta * KL_normal +
        gamma * KL_laplace`` where the change prior is conditioned on the
        Viterbi regime sequence decoded from the observations.
        """
        if outputs.extras is None or outputs.x_recon is None or "obs" not in outputs.extras:
            raise ValueError("NCTRL loss requires the forward-pass extras.")
        obs = outputs.extras["obs"]
        x_recon = outputs.x_recon
        mus = outputs.extras["mean"]
        logvars = outputs.extras["logvar"]
        z_est = outputs.z

        length = obs.shape[1]
        # The ARHMM consumes observations, not latent codes (paper Eq. 5,
        # Appendix B.8).  Its likelihood is optimised by its own optimiser and
        # its posterior over regimes is inferred for the change prior.
        E_logp_x, c_est = self.core.hmm(obs)
        hmm_loss = -E_logp_x.mean()
        if torch.is_grad_enabled():
            self.hmm_optimizer.zero_grad()
            hmm_loss.backward(retain_graph=True)
            self.hmm_optimizer.step()

        embeddings = self.core.c_embeddings(c_est)
        # Same split as the official step: the first ``lags`` steps pay a
        # full reconstruction term, the remaining steps a per-step one.
        recon = self.core.reconstruction_loss(obs[:, :1], x_recon[:, :1]) + (
            self.core.reconstruction_loss(obs[:, 1:], x_recon[:, 1:]) / max(1, length - 1)
        )
        q_dist = torch.distributions.Normal(mus, torch.exp(logvars / 2).clamp(min=1e-6, max=1e6))
        log_qz = q_dist.log_prob(z_est)
        p_dist = torch.distributions.Normal(
            torch.zeros_like(mus[:, :1]), torch.ones_like(logvars[:, :1])
        )
        # Official KLD is ``log q - log p`` (Eq. 26): a positive divergence that
        # the optimiser minimises.  The previous adapter negated this term.
        log_pz_normal = torch.sum(p_dist.log_prob(z_est[:, :1]), dim=-1).sum(-1)
        log_qz_normal = torch.sum(log_qz[:, :1], dim=-1).sum(-1)
        kld_normal = (log_qz_normal - log_pz_normal).mean()
        residuals, logabsdet = self.core.transition_prior(z_est, embeddings)
        log_pz = torch.sum(self.core.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
            torch.sum(log_qz[:, 1:], dim=-1).sum(-1) - log_pz
        ).div(max(1, length - 1)).mean()
        total = recon + self.core.beta * kld_normal + self.core.gamma * kld_laplace
        return {"loss": total, "recon": recon, "hmm": hmm_loss}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        return z_past[:, -1]

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "NCTRL: under a Markov assumption on time-delayed causally related "
            "processes and nonstationary regimes with independent components, "
            "the latent factors are identifiable up to permutation and "
            "component-wise transforms without observed auxiliary variables. "
            "The official ARHMM/flow implementation is used."
        )

    def ambiguity_class(self) -> AmbiguityClass:
        return AmbiguityClass.PERM_COMPONENTWISE
