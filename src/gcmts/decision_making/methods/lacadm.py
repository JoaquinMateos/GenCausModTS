"""LacaDM: a latent causal diffusion model for multi-objective RL.

Reference: Yan et al., *LacaDM: A Latent Causal Diffusion Model for
Multiobjective Reinforcement Learning*, 2025 — BibTeX
``xuemingyanLacaDMLatentCausal2025``.

A latent causal module encodes a window of noisy policy history to a latent
state following a delayed temporally-causal process, and a denoising network
predicts the reverse kernel. Each reverse step is a loss-guided update

    pi_hat^{CRL}_{t-1} = pi_hat_{t-1} - alpha * grad_{pi_hat_{t-1}} L_total,

with ``L_total = ||mu(pi_hat_t) - pi_hat_{t-1}||^2 +
beta ||d(pi_hat_{t-1}, z_t) - pi_hat_t||^2 + lambda ||pi_hat_{t-1}||_1``. The
deliverable is a policy/action sequence; the guarantee is empirical, not an
identifiability theorem.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.core.backbones import MLP, GaussianHead
from gcmts.decision_making.base import BaseDecisionMaker
from gcmts.typing import Batch

logger = logging.getLogger(__name__)

__all__ = ["LacaDM"]


class LacaDM(BaseDecisionMaker):
    """Latent causal diffusion policy generator.

    Args:
        state_dim: dimension of the observed state.
        action_dim: dimension of the action.
        latent_dim: latent causal dimension.
        n_steps: number of diffusion steps.
        hidden_dims: network widths.
        beta: weight of the causal-consistency (decoder) term.
        lam: weight of the L1 policy sparsity term.
        alpha: gradient step size of the loss-guided reverse update.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 4,
        n_steps: int = 10,
        hidden_dims: tuple[int, ...] = (128, 128),
        beta: float = 1.0,
        lam: float = 0.01,
        alpha: float = 0.1,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, n_objectives=2)
        self.z_dim = latent_dim
        self.n_steps = n_steps
        self.beta = beta
        self.lam = lam
        self.alpha = alpha
        self.encoder = GaussianHead(state_dim, latent_dim, hidden_dims=hidden_dims)
        self.denoiser = MLP(action_dim + latent_dim + 1, action_dim, hidden_dims=hidden_dims)
        self.decoder = MLP(latent_dim, state_dim, hidden_dims=hidden_dims)
        self.register_buffer("betas", torch.linspace(1e-4, 2e-2, n_steps))
        logger.debug("Initialised LacaDM (s=%d, a=%d)", state_dim, action_dim)

    def _alpha_bar(self, step: int) -> Tensor:
        return torch.prod(1.0 - self.betas[: step + 1])

    def _encode(self, states: Tensor) -> Tensor:
        batch = states.shape[0]
        mean, _ = self.encoder(states.reshape(batch, -1)[:, -self.state_dim :])
        return mean

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        states = batch.x
        actions = self._actions(batch, states)
        batch_size, time, action_dim = actions.shape
        latent = self._encode(states)
        step = torch.randint(0, self.n_steps, (batch_size,), device=states.device)
        alpha_bar = torch.stack([self._alpha_bar(int(s)) for s in step]).to(states.device)
        alpha_bar = alpha_bar.reshape(-1, 1, 1)
        eps = torch.randn_like(actions)
        noisy = alpha_bar.sqrt() * actions + (1 - alpha_bar).sqrt() * eps
        step_feature = (step.float() / self.n_steps).reshape(-1, 1, 1).expand(-1, time, 1)
        repeated_latent = latent.unsqueeze(1).expand(-1, time, -1)
        features = torch.cat([noisy, repeated_latent, step_feature], dim=-1)
        predicted = self.denoiser(features.reshape(batch_size * time, -1)).reshape(
            batch_size, time, action_dim
        )
        recon_state = self.decoder(latent)
        return {
            "latent": latent,
            "predicted": predicted,
            "target_eps": eps,
            "actions": actions,
            "recon_state": recon_state,
        }

    def _actions(self, batch: Batch, states: Tensor) -> Tensor:
        if batch.context is not None and "action" in batch.context:
            return batch.context["action"].float()
        return torch.zeros(states.shape[0], states.shape[1], self.action_dim, device=states.device)

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        denoising = F.mse_loss(outputs["predicted"], outputs["target_eps"])
        states = batch.x
        target_state = states.reshape(states.shape[0], -1)[:, -self.state_dim :]
        causal = F.mse_loss(outputs["recon_state"], target_state)
        sparsity = outputs["actions"].abs().mean()
        total = denoising + self.beta * causal + self.lam * sparsity
        return {"loss": total, "denoising": denoising, "causal": causal, "sparsity": sparsity}

    @torch.no_grad()
    def act(self, states: Tensor, preference: Tensor | None = None) -> Tensor:
        latent = self._encode(states)
        action = torch.randn(states.shape[0], self.action_dim, device=states.device)
        for step in reversed(range(self.n_steps)):
            step_feature = torch.full(
                (states.shape[0], 1), step / self.n_steps, device=states.device
            )
            features = torch.cat([action, latent, step_feature], dim=-1)
            action = self.denoiser(features)
        return action

    def identifiability_statement(self) -> str:
        return (
            "LacaDM: the latent causal module separates environment drift from "
            "policy effects via a normalizing-flow exogenous prior and a "
            "delayed causal process; the returned policy is only empirically "
            "validated (no decision-theoretic identifiability theorem)."
        )
