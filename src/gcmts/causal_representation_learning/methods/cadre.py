"""CaDRe: Causal Discovery and Representation learning.

Reference: Fu et al., *Learning General Causal Structures with Hidden Dynamic
Process for Climate Analysis*, arXiv:2501.12500 (ICML 2026) — BibTeX
``fuLearningGeneralCausal2026``.

This is an independent reimplementation from the paper's equations, **not** a
port of the official ``MinghaoFu/CaDRe`` code (MIT).  The estimated generative
model is the state-space VAE of Eq. 5--8:

* ``q(z|x,u)`` — Gaussian z-encoder conditioned on the regime/context ``u``;
* ``q(s|x)``   — Gaussian s-encoder for the non-stationary observation noise;
* ``p(x|z,s)`` — MLP decoder;
* ``p(z_t|z_{t-1},u)`` — a triangular residual prior with independent unit
  Gaussian innovations, ``r_i = z_{t,i} - MLP_i(z_{t,<i}, z_{t-1}, u)``;
* ``p(s_t|z_t)`` — a triangular residual prior
  ``u_i = s_{t,i} - MLP_i(z_t, s_{t,<i})``.

The ELBO is augmented with the paper's sparsity (``L_s``) and DAG
(``L_d``) penalties, Eq. 9.  The observed graph is obtained from Corollary 2,
``J_g = I - diag(J_m) J_m^{-1}`` with ``J_m = d x / d s``; the latent
instantaneous graph is the same-time Jacobian of the residual prior and the
lagged graph its derivative with respect to ``z_{t-1}``.

Simplifications relative to the paper:
* a single Markov lag and a single residual-prior MLP per latent dimension;
* the residual priors have an exactly unit diagonal Jacobian, so the
  log-determinant term in Eqs. 7--8 is zero and is omitted;
* structure penalties are evaluated on the batch-averaged Jacobians rather
  than on every sample separately;
* Gaussian residual densities replace the paper's learned flow densities;
* the decoder is a plain MLP (the base class still supplies
  ``observation_features`` for rendered-image inputs).
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import (
    MLP,
    GaussianHead,
    flatten_sequence,
    gaussian_logpdf,
    gaussian_reconstruction_nll,
    prepare_context,
    reparameterise,
    unflatten_sequence,
)
from gcmts.core.graph_ops import acyclicity
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["CaDRe"]


class CaDRe(BaseCausalRepresentationLearner):
    r"""Causal Discovery and Representation learning (CaDRe).

    Args:
        observed_dim: number ``D`` of observed variables.
        latent_dim: number ``d`` of latent causal factors.
        u_dim: width of the one-hot regime/context vector (0 = unconditional).
        hidden_dims: widths of the z-encoder and residual priors.
        s_hidden: widths of the s-encoder.
        decoder_hidden: widths of the decoder.
        prior_hidden: widths of the per-latent residual-prior MLPs.
        kl_z_weight / kl_s_weight: weights :math:`\lambda_2`, :math:`\lambda_1`
            of the ELBO KL terms (paper values ``1e-2`` and ``4e-3``).
        alpha / beta: weights of the sparsity and acyclicity penalties
            (paper values ``1e-4`` and ``5e-5``).
        obs_noise: initial observation-noise scale of the reconstruction term.
        image_shape / image_feature_dim: forwarded to the base class so that
            rendered frames are mapped through the shared CNN encoder.
    """

    kl_weight: float = 1e-2

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        u_dim: int = 0,
        hidden_dims: tuple[int, ...] = (128, 128),
        s_hidden: tuple[int, ...] = (64,),
        decoder_hidden: tuple[int, ...] = (128, 128),
        prior_hidden: tuple[int, ...] = (64,),
        kl_z_weight: float = 1e-2,
        kl_s_weight: float = 4e-3,
        alpha: float = 1e-4,
        beta: float = 5e-5,
        obs_noise: float = 1.0,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=1,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        self.u_dim = int(u_dim)
        # For vector observations s has the same width as x (the paper's d_x).
        # With rendered frames the encoder sees CNN features, so s lives in
        # feature space instead of the flattened pixel space.
        self.s_dim = int(self.encoder_input_dim(observed_dim))
        self.kl_weight = float(kl_z_weight)
        self.kl_s_weight = float(kl_s_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # q(z|x,u), q(s|x), p(x|z,s)
        self.z_encoder = GaussianHead(
            self.encoder_input_dim(observed_dim) + self.u_dim,
            latent_dim,
            hidden_dims=hidden_dims,
        )
        self.s_encoder = GaussianHead(
            self.encoder_input_dim(observed_dim),
            self.s_dim,
            hidden_dims=s_hidden,
        )
        self.decoder = MLP(latent_dim + self.s_dim, observed_dim, hidden_dims=decoder_hidden)
        # Residual prior r_i = z_i - MLP_i(z_<i, z_{t-1}, u); one MLP per
        # latent dimension, so the Jacobian wrt z_t is triangular with unit
        # diagonal.
        self.z_prior_nets = nn.ModuleList(
            MLP(
                i + latent_dim + self.u_dim,
                1,
                hidden_dims=prior_hidden,
            )
            for i in range(latent_dim)
        )
        # Residual s prior u_i = s_i - MLP_i(z, s_<i).
        self.s_prior_nets = nn.ModuleList(
            MLP(latent_dim + i, 1, hidden_dims=prior_hidden) for i in range(self.s_dim)
        )
        self.register_buffer("obs_logvar", math.log(obs_noise**2) * torch.ones(1))
        self.obs_logvar: Tensor

        self._latent_adjacency: Tensor | None = None
        self._instantaneous_adjacency: Tensor | None = None
        self._observed_adjacency: Tensor | None = None
        self._module_sensitivity: Tensor | None = None
        logger.debug("Initialised CaDRe (D=%d, d=%d, u=%d)", observed_dim, latent_dim, u_dim)

    @property
    def latent_dim(self) -> int:
        """Latent dimension (always defined for this model)."""
        if self._latent_dim is None:
            raise RuntimeError("latent_dim was not initialised.")
        return self._latent_dim

    # ------------------------------------------------------------------
    # encoders / decoder
    # ------------------------------------------------------------------
    @staticmethod
    def _condition(x: Tensor, u: Tensor | None) -> Tensor:
        return x if u is None else torch.cat([x, u], dim=-1)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        x_flat, (batch, time) = self.observation_features(x)
        u = prepare_context(context, x_flat.shape[0], n_classes=self.u_dim)
        mean, _ = self.z_encoder(self._condition(x_flat, u))
        return unflatten_sequence(mean, batch, time)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latents with the s-noise set to its mean (zero)."""
        z_flat, (batch, time) = flatten_sequence(z)
        s = torch.zeros(z_flat.shape[0], self.s_dim, device=z_flat.device, dtype=z_flat.dtype)
        return unflatten_sequence(self.decoder(torch.cat([z_flat, s], dim=-1)), batch, time)

    # ------------------------------------------------------------------
    # residual priors
    # ------------------------------------------------------------------
    def _z_residual_inputs(self, i: int, z: Tensor, z_prev: Tensor, u: Tensor | None) -> Tensor:
        parts = [z[..., :i], z_prev]
        if u is not None:
            parts.append(u)
        return torch.cat(parts, dim=-1)

    def _z_residuals(self, z: Tensor, z_prev: Tensor, u: Tensor | None) -> Tensor:
        """Residuals ``r_i = z_i - g_i(z_<i, z_{t-1}, u)``, shape ``(B, T, d)``."""
        residuals = [
            z[..., i] - self.z_prior_nets[i](self._z_residual_inputs(i, z, z_prev, u)).squeeze(-1)
            for i in range(int(self.latent_dim))
        ]
        return torch.stack(residuals, dim=-1)

    def _s_residuals(self, s: Tensor, z: Tensor) -> Tensor:
        """Residuals ``u_i = s_i - h_i(z, s_<i)``, shape ``(B, T, D)``."""
        residuals = [
            s[..., i] - self.s_prior_nets[i](torch.cat([z, s[..., :i]], dim=-1)).squeeze(-1)
            for i in range(self.s_dim)
        ]
        return torch.stack(residuals, dim=-1)

    # ------------------------------------------------------------------
    # Jacobians / structure extraction
    # ------------------------------------------------------------------
    def _residual_jacobians(
        self, z: Tensor, z_prev: Tensor, u: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Batch-averaged ``d r / d z_t`` and ``d r / d z_{t-1}``.

        Both matrices are lower/row-wise restricted by construction: row ``i``
        is zero for ``j >= i`` in the instantaneous matrix.  When gradients are
        enabled the Jacobians stay attached to the graph so the structural
        penalties train the prior networks.
        """
        differentiable = torch.is_grad_enabled() and z.requires_grad
        if differentiable:
            return self._residual_jacobians_impl(z, z_prev, u, create_graph=True)
        with torch.enable_grad():
            return self._residual_jacobians_impl(
                z.detach(),
                z_prev.detach(),
                None if u is None else u.detach(),
                create_graph=False,
            )

    def _residual_jacobians_impl(
        self, z: Tensor, z_prev: Tensor, u: Tensor | None, *, create_graph: bool
    ) -> tuple[Tensor, Tensor]:
        d = int(self.latent_dim)
        z_flat = z.reshape(-1, d)
        prev_flat = z_prev.reshape(-1, d)
        u_flat = None if u is None else u.reshape(-1, u.shape[-1])
        instant = torch.zeros(d, d, device=z.device, dtype=z.dtype)
        lagged = torch.zeros(d, d, device=z.device, dtype=z.dtype)
        for i in range(d):
            inputs = self._z_residual_inputs(i, z_flat, prev_flat, u_flat)
            if not inputs.requires_grad:
                inputs = inputs.detach().requires_grad_(True)
            output = self.z_prior_nets[i](inputs).sum()
            (grad,) = torch.autograd.grad(
                output,
                inputs,
                create_graph=create_graph,
                retain_graph=True,
            )
            if i:
                instant[i, :i] = -grad[:, :i].mean(dim=0)
            lagged[i, :] = -grad[:, i : i + d].mean(dim=0)
        return instant, lagged

    def _decoder_jacobian(self, z: Tensor, s: Tensor) -> Tensor:
        """Per-sample ``d x / d(z, s)`` of shape ``(n, D, d + D)``."""
        if torch.is_grad_enabled() and (z.requires_grad or s.requires_grad):
            return self._decoder_jacobian_impl(torch.cat([z, s], dim=-1), create_graph=True)
        with torch.enable_grad():
            inputs = torch.cat([z.detach(), s.detach()], dim=-1).requires_grad_(True)
            return self._decoder_jacobian_impl(inputs, create_graph=False)

    def _decoder_jacobian_impl(self, inputs: Tensor, *, create_graph: bool) -> Tensor:
        output = self.decoder(inputs)
        n, dim = output.shape
        eye = torch.eye(dim, device=output.device, dtype=output.dtype)
        grad_outputs = eye.unsqueeze(1).expand(dim, n, dim)
        (jacobian,) = torch.autograd.grad(
            output,
            inputs,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            create_graph=create_graph,
            retain_graph=True,
        )
        return jacobian.permute(1, 0, 2)  # (n, D, d + D)

    def _structure_summaries(
        self, z: Tensor, z_prev: Tensor, s: Tensor, u: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(J_inst, J_lag, J_g, sensitivity)`` summaries."""
        instant, lagged = self._residual_jacobians(z, z_prev, u)
        d = int(self.latent_dim)
        if self.s_dim != self.observed_dim:
            # Rendered-image inputs: J_m is rectangular, so the Corollary-2
            # observed graph is not defined in pixel space.  Keep the model
            # CNN-compatible and report zero (D, D) / (D, d) summaries.
            zero_obs = torch.zeros(
                self.observed_dim,
                self.observed_dim,
                device=z.device,
                dtype=z.dtype,
            )
            zero_sens = torch.zeros(self.observed_dim, d, device=z.device, dtype=z.dtype)
            return instant, lagged, zero_obs, zero_sens
        full = self._decoder_jacobian(z.reshape(-1, d), s.reshape(-1, self.s_dim))
        mixing = full[:, :, d:]
        diagonal = torch.diag_embed(torch.diagonal(mixing, dim1=-2, dim2=-1))
        eye = torch.eye(self.observed_dim, device=full.device, dtype=full.dtype)
        # Corollary 2: J_g = I - diag(J_m) J_m^{-1}.  Self-loops are masked,
        # as in the paper's observed-graph construction.
        observed = (eye - diagonal @ torch.linalg.inv(mixing)) * (1.0 - eye)
        sensitivity = full[:, :, :d].abs().mean(dim=0)
        return instant, lagged, observed.mean(dim=0), sensitivity

    # ------------------------------------------------------------------
    # forward / loss
    # ------------------------------------------------------------------
    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x_flat, (batch_size, time) = self.observation_features(batch.x)
        u_flat = prepare_context(batch.context, x_flat.shape[0], n_classes=self.u_dim)
        z_mean, z_logvar = self.z_encoder(self._condition(x_flat, u_flat))
        z = reparameterise(z_mean, z_logvar)
        s_mean, s_logvar = self.s_encoder(x_flat)
        s = reparameterise(s_mean, s_logvar)
        x_recon = self.decoder(torch.cat([z, s], dim=-1))

        z_seq = unflatten_sequence(z, batch_size, time)
        s_seq = unflatten_sequence(s, batch_size, time)
        z_mean_seq = unflatten_sequence(z_mean, batch_size, time)
        z_logvar_seq = unflatten_sequence(z_logvar, batch_size, time)
        s_mean_seq = unflatten_sequence(s_mean, batch_size, time)
        s_logvar_seq = unflatten_sequence(s_logvar, batch_size, time)
        u_seq = None if u_flat is None else u_flat.reshape(batch_size, time, self.u_dim)
        zeros = torch.zeros_like(z_seq[:, :1])
        z_prev = torch.cat([zeros, z_seq[:, :-1]], dim=1)

        z_residuals = self._z_residuals(z_seq, z_prev, u_seq)
        s_residuals = self._s_residuals(s_seq, z_seq)

        log_q_z = gaussian_logpdf(z_seq, z_mean_seq, z_logvar_seq)
        log_q_s = gaussian_logpdf(s_seq, s_mean_seq, s_logvar_seq)
        # Unit-variance innovation densities.  p(z_0) is standard normal; for
        # t >= 1 the triangular residual prior supplies p(z_t | z_{t-1}, u).
        log_p_z = gaussian_logpdf(
            z_residuals, torch.zeros_like(z_residuals), torch.zeros_like(z_residuals)
        )
        if time > 1:
            initial = gaussian_logpdf(
                z_seq[:, :1], torch.zeros_like(z_seq[:, :1]), torch.zeros_like(z_seq[:, :1])
            )
            log_p_z = torch.cat([initial, log_p_z[:, 1:]], dim=1)
        log_p_s = gaussian_logpdf(
            s_residuals, torch.zeros_like(s_residuals), torch.zeros_like(s_residuals)
        )

        instant, lagged, observed, sensitivity = self._structure_summaries(
            z_seq, z_prev, s_seq, u_seq
        )
        self._latent_adjacency = lagged.detach().unsqueeze(-1).clone()
        self._instantaneous_adjacency = instant.detach().clone()
        self._observed_adjacency = observed.detach().clone()
        self._module_sensitivity = sensitivity.detach().clone()

        return CausalRepresentationOutput(
            z=z_seq,
            x_recon=unflatten_sequence(x_recon, batch_size, time),
            adjacency=lagged.detach().unsqueeze(-1).clone(),
            log_prob=log_p_z.detach(),
            extras={
                "log_q_z": log_q_z,
                "log_p_z": log_p_z,
                "log_q_s": log_q_s,
                "log_p_s": log_p_s,
                "z_residuals": z_residuals,
                "s_residuals": s_residuals,
                "instantaneous_adjacency": instant,
                "lagged_adjacency": lagged,
                "observed_adjacency": observed,
                "module_sensitivity": sensitivity,
            },
        )

    @staticmethod
    def _markov_norm(jacobian: Tensor) -> Tensor:
        """``M(J) = (I + J)^T (I + J) - I`` from the paper."""
        eye = torch.eye(jacobian.shape[-1], device=jacobian.device, dtype=jacobian.dtype)
        return (eye + jacobian).transpose(-1, -2) @ (eye + jacobian) - eye

    @staticmethod
    def _scaled_acyclicity(jacobian: Tensor) -> Tensor:
        """Acyclicity on a max-scaled graph for numerical stability.

        ``graph_ops.acyclicity`` evaluates ``tr(exp(A∘A)) - d``; a decoder
        Jacobian inverse can be large at initialisation, so the graph is first
        divided by its detached maximum (when above one).  The constraint is
        still exactly zero for a DAG.
        """
        scale = jacobian.abs().max().detach().clamp(min=1.0)
        return acyclicity(jacobian / scale)

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None or outputs.extras is None:
            raise ValueError("CaDRe forward output is incomplete.")
        extras = outputs.extras
        recon = gaussian_reconstruction_nll(
            outputs.x_recon.reshape_as(batch.x), batch.x, self.obs_logvar
        )
        kl_z = (extras["log_q_z"] - extras["log_p_z"]).mean()
        kl_s = (extras["log_q_s"] - extras["log_p_s"]).mean()
        instant = extras["instantaneous_adjacency"]
        lagged = extras["lagged_adjacency"]
        observed = extras["observed_adjacency"]
        sparsity = (
            self._markov_norm(instant).abs().sum()
            + self._markov_norm(lagged).abs().sum()
            + observed.abs().sum()
        )
        acyc = self._scaled_acyclicity(instant) + self._scaled_acyclicity(observed)
        total = (
            recon
            + self.kl_weight * kl_z
            + self.kl_s_weight * kl_s
            + self.alpha * sparsity
            + self.beta * acyc
        )
        return {
            "loss": total,
            "recon": recon,
            "kl_z": kl_z,
            "kl_s": kl_s,
            "sparsity": sparsity,
            "acyclicity": acyc,
        }

    # ------------------------------------------------------------------
    # transition / graph getters
    # ------------------------------------------------------------------
    def _transition_context(self, context: dict[str, Tensor] | None, batch: int) -> Tensor | None:
        if context is None or "u" not in context:
            return None
        value = context["u"]
        if value.ndim == 3:
            value = value[:, -1]
        return prepare_context({"u": value}, batch, n_classes=self.u_dim)

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        """One-step mean of the residual mechanism, solved in causal order.

        Setting every residual to zero gives the triangular system
        ``z_i = g_i(z_<i, z_{t-1}, u)``, which is evaluated from ``i = 0`` up.
        """
        z_prev = z_past[:, -1]
        batch = z_prev.shape[0]
        u = self._transition_context(context, batch)
        values: list[Tensor] = []
        for i in range(int(self.latent_dim)):
            earlier = torch.stack(values, dim=-1) if values else z_prev.new_zeros(batch, 0)
            parts = [earlier, z_prev]
            if u is not None:
                parts.append(u)
            values.append(self.z_prior_nets[i](torch.cat(parts, dim=-1)).squeeze(-1))
        return torch.stack(values, dim=-1)

    def _zero_graph(self, *shape: int) -> Tensor:
        parameter = next(self.parameters())
        return torch.zeros(*shape, device=parameter.device, dtype=parameter.dtype)

    def get_latent_adjacency(self) -> Tensor | None:
        """Lagged latent graph ``(d, d, 1)`` from the residual prior."""
        if self._latent_adjacency is None:
            return self._zero_graph(self.latent_dim or 0, self.latent_dim or 0, 1)
        return self._latent_adjacency

    def get_instantaneous_adjacency(self) -> Tensor:
        """Same-time latent graph ``(d, d)``; rows are lower-triangular."""
        if self._instantaneous_adjacency is None:
            return self._zero_graph(self.latent_dim or 0, self.latent_dim or 0)
        return self._instantaneous_adjacency

    def get_observed_adjacency(self) -> Tensor:
        """Observed causal graph ``(D, D)`` from Corollary 2."""
        if self._observed_adjacency is None:
            return self._zero_graph(self.observed_dim, self.observed_dim)
        return self._observed_adjacency

    def get_module_sensitivity(self) -> Tensor:
        """Mean absolute decoder sensitivity ``d x / d z`` of shape ``(D, d)``."""
        if self._module_sensitivity is None:
            return self._zero_graph(self.observed_dim, self.latent_dim or 0)
        return self._module_sensitivity

    def identifiability_statement(self) -> str:
        return (
            "CaDRe: the latent factors are identifiable up to permutation and "
            "component-wise invertible transformations under contextual "
            "variability, and the latent instantaneous/lagged structures plus "
            "the observed DAG are recovered from the Jacobians of the residual "
            "priors and the decoder (Theorems 3.2, 3.5, 3.6)."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
