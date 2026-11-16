"""iVAE, adapted from the official implementation.

Paper: Khemakhem et al., *Variational Autoencoders and Nonlinear ICA: A
Unifying Framework* (AISTATS 2020), BibTeX
``khemakhemVariationalAutoencodersNonlinear2020``.
Official source: https://github.com/ilkhem/iVAE.
The official exponential-family prior, encoder and decoder are used; temporal
observations are flattened per timestep and the auxiliary variable is repeated
across the sequence. The vendored model bounds the encoder log-variance and the
prior log-scale to ``[-6, 6]`` as a numerical guard; the prior family, the
networks and the fixed decoder variance are unchanged from the official code.

Objective. The official model maximises the evidence lower bound

    ELBO = E_{q(z|x,u)} [ log p(x|z) + log p(z|u) - log q(z|x,u) ],

with ``p(z|u) = N(z; 0, diag(exp(2 * logl(u))))`` and a fixed Gaussian decoder
variance. Training minimises ``-ELBO``; :meth:`IVAEOfficial.loss` returns that
negative ELBO (plus separate ``recon``/``kl`` terms), because
:class:`~gcmts.core.trainer.SimpleTrainer` minimises the reported ``loss``.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput
from gcmts.vendor.ivaE.nets import Normal as NormalDist
from gcmts.vendor.ivaE.nets import iVAE as IVAECore

logger = logging.getLogger(__name__)

__all__ = ["IVAEOfficial"]

_LOG_2PI = math.log(2.0 * math.pi)


def _gaussian_logpdf(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    """Per-dimension Gaussian log-density.

    This is the official ``nets.Normal.log_pdf(..., reduce=False)`` formula
    written with scalars instead of the vendored ``Dist`` helpers, so that it is
    device- and dtype-agnostic (the vendored helpers keep their base
    distributions on the construction device).
    """
    return -0.5 * (_LOG_2PI + var.log() + (x - mean).pow(2) / var)


def _check_batch(size: int, batch: int) -> None:
    if size != batch:
        raise ValueError(f"context['u'] has {size} rows but the observation batch has {batch}.")


def _aux(
    context: dict[str, Tensor] | None,
    batch: int,
    time: int,
    dim: int,
    device: torch.device,
) -> Tensor:
    """Return the auxiliary variable ``u`` as ``(batch * time, dim)`` floats.

    The benchmark generators store regimes in two equivalent encodings:

    * an already one-hot/soft indicator of width ``dim``, either per trajectory
      ``(B, dim)`` or per timestep ``(B, T, dim)`` (NP/VAR, CartPole, ...), and
    * integer labels ``(B,)``, ``(B, 1)`` or ``(B, T)`` (physics, control and
      linear-process suites).

    Dtype (floating vs integer) decides between the two when the width happens
    to coincide with ``T``. A one-dimensional auxiliary variable cannot carry
    regime information (the official prior then has a constant input), so it is
    encoded as a constant zero column: one-hot encoding a constant label would
    either be a no-op or crash for label ``1`` with a single class.
    """
    dim = max(int(dim), 1)
    constant = torch.zeros(batch * time, dim, device=device)
    if context is None or "u" not in context:
        return constant
    u = context["u"]
    if u.ndim == 0:
        u = u.reshape(1, 1)
    if dim == 1:
        _check_batch(int(u.shape[0]), batch)
        return constant
    if u.dtype.is_floating_point and u.shape[-1] == dim:
        # Already encoded (one-hot or soft) auxiliary vector.
        if u.ndim == 2:
            u = u.unsqueeze(1).expand(-1, time, -1)
        elif u.ndim == 3:
            if u.shape[1] == 1:
                u = u.expand(-1, time, -1)
            elif u.shape[1] != time:
                raise ValueError(
                    f"context['u'] has {u.shape[1]} timesteps but the observation batch has {time}."
                )
        else:
            raise ValueError(
                "context['u'] must be (B, aux_dim), (B, T, aux_dim), (B,), "
                f"(B, 1) or (B, T); got shape {tuple(u.shape)}."
            )
        _check_batch(int(u.shape[0]), batch)
        return u.reshape(batch * time, dim).to(device=device, dtype=torch.float32)

    # Integer regime labels, possibly with a trailing singleton dimension.
    labels = u
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.expand(-1, time)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1).expand(-1, time)
    if labels.ndim != 2 or labels.shape[1] != time:
        raise ValueError(
            "context['u'] must be (B, aux_dim), (B, T, aux_dim), (B,), "
            f"(B, 1) or (B, T); got shape {tuple(u.shape)}."
        )
    if labels.dtype.is_floating_point:
        if not bool(torch.allclose(labels, labels.round())):
            raise ValueError(
                "context['u'] is floating point but neither one-hot with width "
                f"aux_dim={dim} nor integer-valued: shape {tuple(u.shape)}."
            )
        labels = labels.round()
    labels = labels.long()
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= dim):
        raise ValueError(
            f"regime labels must lie in [0, {dim}); got [{int(labels.min())}, {int(labels.max())}]."
        )
    _check_batch(int(labels.shape[0]), batch)
    return (
        F.one_hot(labels, num_classes=dim)
        .to(device=device, dtype=torch.float32)
        .reshape(batch * time, dim)
    )


class IVAEOfficial(BaseCausalRepresentationLearner):
    """Interface around the official iVAE model.

    Args:
        observed_dim: dimension ``D`` of ``x`` (``C*H*W`` for rendered images).
        latent_dim: dimension ``d`` of ``z``.
        u_dim: width of the one-hot regime encoding (0/1 = unconditional VAE).
        hidden_dim: width of every hidden layer of the official MLPs.
        n_layers: number of linear layers in the official MLPs.
        activation: official activation (``"lrelu"``, ``"xtanh"``, ``"sigmoid"``).
        decoder_var: fixed variance of the official Gaussian decoder.
        obs_noise: optional observation-noise *standard deviation*; when given it
            overrides ``decoder_var`` with ``obs_noise ** 2`` (matching the
            ``obs_noise`` argument of the repository's :class:`IVAE` adapter).
        x_scale: multiplicative rescaling applied to the encoder input.
        kl_weight: weight of the KL term (1.0 = the official ELBO). The shared
            trainer may anneal this value as a beta warm-up.
        decoder_clip: hard bound on the decoder mean in scaled observation space
            (numerical guard; the official model has none).
        image_shape / image_feature_dim: shared CNN feature encoder for rendered
            observations.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        u_dim: int = 0,
        hidden_dim: int = 100,
        n_layers: int = 3,
        activation: str = "xtanh",
        lr: float = 1e-3,
        decoder_var: float = 0.01,
        x_scale: float = 1.0,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 1.0,
        obs_noise: float | None = None,
        decoder_clip: float = 20.0,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        del hidden_dims, decoder_hidden
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=1,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        if obs_noise is not None:
            if obs_noise <= 0:
                raise ValueError("obs_noise must be positive.")
            decoder_var = float(obs_noise) ** 2
        if decoder_var <= 0:
            raise ValueError("decoder_var must be positive.")
        if x_scale <= 0:
            raise ValueError("x_scale must be positive.")
        self.n_aux = max(int(u_dim), 1)
        self.x_scale = float(x_scale)
        self.kl_weight: float = float(kl_weight)
        self.decoder_clip = float(decoder_clip)
        self.trainer_lr = float(lr)
        self.core = IVAECore(
            latent_dim=latent_dim,
            data_dim=self.encoder_input_dim(observed_dim),
            aux_dim=self.n_aux,
            n_layers=int(n_layers),
            hidden_dim=int(hidden_dim),
            activation=str(activation),
            device="cpu",
        )
        # The official module keeps the prior mean and decoder variance as plain
        # tensors: they would stay on the construction device and would not be
        # saved in the state dict. Registering them as buffers of the wrapped
        # module makes them follow ``.to()``/``.float()`` and checkpoints.
        del self.core.prior_mean
        del self.core.decoder_var
        self.core.register_buffer("prior_mean", torch.zeros(1))
        self.core.register_buffer("decoder_var", torch.tensor(float(decoder_var)))

    def _apply(self, fn, recurse: bool = True):  # type: ignore[no-untyped-def]
        """Move the wrapped module's plain distribution helpers with the model.

        The vendored iVAE keeps its three ``Dist`` helpers as plain Python
        objects, so ``nn.Module._apply`` cannot move them; without this the
        official ``sample``/``log_pdf`` helpers would draw CPU noise (or fail on
        a dtype change). This adapter evaluates log-densities directly, but the
        wrapped ``core.forward``/``core.elbo`` remain usable after a move.
        """
        module = super()._apply(fn, recurse=recurse)
        reference = next(module.parameters())
        with torch.no_grad():
            for dist in (
                module.core.prior_dist,
                module.core.decoder_dist,
                module.core.encoder_dist,
            ):
                dist.device = reference.device
                if hasattr(dist, "c"):
                    dist.c = fn(dist.c).detach()
                if isinstance(dist, NormalDist):
                    zeros = fn(torch.zeros(1)).detach()
                    ones = fn(torch.ones(1)).detach()
                    dist._dist = type(dist._dist)(zeros, ones)
        return module

    def _obs(self, x: Tensor) -> Tensor:
        if self.image_shape is None:
            return x
        features, (batch, time) = self.observation_features(x)
        return features.reshape(batch, time, -1)

    def _flat(
        self, x: Tensor, context: dict[str, Tensor] | None
    ) -> tuple[Tensor, Tensor, int, int]:
        obs = self._obs(x)
        batch, time, dim = obs.shape
        flat = obs.reshape(batch * time, dim).to(self.core.prior_mean.dtype) * self.x_scale
        u = _aux(context, batch, time, self.n_aux, flat.device).to(dtype=flat.dtype)
        return flat, u, batch, time

    def _sample(self, mean: Tensor, var: Tensor) -> Tensor:
        """Official reparameterisation ``z = g + sqrt(v) * eps``.

        ``torch.randn_like`` replaces the vendored ``Normal.sample`` so the noise
        follows the parameters' device and dtype (autocast included).
        """
        z = mean + var.sqrt() * torch.randn_like(mean)
        # Numerical guard: the official implementation leaves z unbounded, which
        # can overflow the Gaussian log-density when the encoder variance drifts.
        return z.clamp(-8.0, 8.0)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        flat, u, batch, time = self._flat(x, context)
        mean, _ = self.core.encoder_params(flat, u)
        return mean.reshape(batch, time, self.latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1]).to(self.core.prior_mean.dtype)
        rec = self.core.f(flat).clamp(-self.decoder_clip, self.decoder_clip)
        return (rec / self.x_scale).reshape(z.shape[0], z.shape[1], -1)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        flat, u, b, t = self._flat(batch.x, batch.context)
        mean, var = self.core.encoder_params(flat, u)
        z = self._sample(mean, var)
        decoder_mean, decoder_var = self.core.decoder_params(z)
        decoder_mean = decoder_mean.clamp(-self.decoder_clip, self.decoder_clip)
        prior_mean, prior_var = self.core.prior_params(u)
        # The official ELBO, kept as per-sample sums so that ``loss`` can report
        # the exact same objective it differentiates (no second sampling pass).
        log_px = _gaussian_logpdf(flat, decoder_mean, decoder_var).sum(dim=-1)
        log_q = _gaussian_logpdf(z, mean, var).sum(dim=-1)
        log_p = _gaussian_logpdf(z, prior_mean, prior_var).sum(dim=-1)
        return CausalRepresentationOutput(
            z=z.reshape(b, t, self.latent_dim),
            x_recon=(decoder_mean / self.x_scale).reshape(b, t, -1),
            log_prob=log_p.reshape(b, t),
            extras={
                "mean": mean.reshape(b, t, self.latent_dim),
                # The official encoder head is a variance; report its log.
                "logvar": var.log().reshape(b, t, self.latent_dim),
                "log_px": log_px,
                "log_q": log_q,
                "log_p": log_p,
            },
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        """Negative official ELBO, decomposed into reconstruction and KL terms.

        ``loss = recon + kl_weight * kl`` with ``recon = -E[log p(x|z)]`` and
        ``kl = E[log q(z|x,u) - log p(z|u)]``, i.e. exactly the official
        objective (up to the KL weight) with the sign expected by a minimiser.
        """
        if outputs.extras is None or "log_px" not in outputs.extras:
            raise ValueError("IVAEOfficial.forward did not return the ELBO terms.")
        log_px = outputs.extras["log_px"]
        log_q = outputs.extras["log_q"]
        log_p = outputs.extras["log_p"]
        recon = -log_px.mean()
        kl = (log_q - log_p).mean()
        total = recon + self.kl_weight * kl
        return {"loss": total, "recon": recon, "kl": kl}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        """The prior is static, so the one-step mean is the identity."""
        return z_past[:, -1]

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "iVAE: with a sufficiently variable auxiliary variable and an "
            "injective mixing map, the latents are identifiable up to "
            "permutation and component-wise transforms. The official iVAE "
            "exponential-family prior and networks are used."
        )

    def ambiguity_class(self) -> AmbiguityClass:
        return AmbiguityClass.PERM_COMPONENTWISE
