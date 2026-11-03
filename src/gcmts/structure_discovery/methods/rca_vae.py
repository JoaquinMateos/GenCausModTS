"""RCA-VAE: regime-aware edge discovery and root-cause scoring.

Reference: Adesunkanmi, Khokhar, Trajcevski and Murad, *Root Cause Analysis of
Hydrogen Bond Separation in Spatio-Temporal Molecular Dynamics using Causal
Models*, arXiv:2508.12500 (2025).

This is a compact reimplementation of the paper's VAE-inspired causal model, not
the authors' code (none is public).  An NRI-style encoder infers a posterior over
three edge types (non-causal, pre-change mechanism, post-change mechanism) from
short temporal windows; an edge-conditioned decoder predicts ``x_{t+1}``; and a
two-component loss combines reconstruction, a KL term against an edge-type prior,
and a group-lasso/PCM penalty.

For root-cause analysis the **primary** score is the theory-faithful mechanism
change: for each node, the mean squared difference between the decoder's
conditional mean under the pre-change and post-change edge posteriors evaluated
on the *same* parent observations.  The paper-literal edge-posterior KL
(``KL(q_pre(a_ij) || q_post(a_ij))`` summed over parents) is returned as a
secondary score.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gcmts.core.backbones import MLP
from gcmts.structure_discovery.base import BaseStructureDiscovery
from gcmts.typing import Batch

__all__ = ["RCAVAE"]


class RCAVAE(BaseStructureDiscovery):
    """Regime-aware NRI/VAE for root-cause structure discovery.

    Args:
        n_nodes: number ``D`` of observed variables.
        window: encoder window length ``W`` (default ``8``).
        n_edge_types: number of edge classes; the paper uses three.
        hidden_dim: width of the node/edge message-passing MLPs.
        tau: Gumbel-softmax temperature.
        prior: prior probabilities over the three edge types.
        recon_weight / kl_weight / pcm_weight: loss weights.
        max_path_len: maximum number of hops returned by :meth:`sample_paths`.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        window: int = 8,
        n_edge_types: int = 3,
        hidden_dim: int = 64,
        tau: float = 0.5,
        prior: tuple[float, ...] = (0.9, 0.05, 0.05),
        recon_weight: float = 1.0,
        kl_weight: float = 0.1,
        pcm_weight: float = 0.001,
        max_path_len: int = 5,
    ) -> None:
        super().__init__(n_nodes=n_nodes, max_path_len=max_path_len)
        if n_edge_types != 3:
            raise ValueError("RCA-VAE uses exactly three edge types.")
        if len(prior) != n_edge_types:
            raise ValueError("prior must have one probability per edge type.")
        self.window = window
        self.n_edge_types = n_edge_types
        self.tau = tau
        self.recon_weight = recon_weight
        self.kl_weight: float = kl_weight
        self.pcm_weight = pcm_weight
        self.register_buffer("prior", torch.tensor(prior, dtype=torch.float).clamp_min(1e-6))
        self.register_buffer("_offdiag", 1.0 - torch.eye(n_nodes))
        self.prior: Tensor
        self._offdiag: Tensor
        # Encoder: f_emb, f_e^1, f_v^1, f_e^2, f_out.
        self.node_embed = MLP(window + 1, hidden_dim, hidden_dims=(hidden_dim,), activation="elu")
        self.edge_embed1 = MLP(
            2 * hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), activation="elu"
        )
        self.node_embed2 = MLP(hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), activation="elu")
        self.edge_embed2 = MLP(
            2 * hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), activation="elu"
        )
        self.edge_out = nn.Linear(hidden_dim, n_edge_types)
        # Decoder: one message MLP per causal mechanism plus f_v^d.
        self.msg_pre = MLP(2, hidden_dim, hidden_dims=(hidden_dim,), activation="relu")
        self.msg_post = MLP(2, hidden_dim, hidden_dims=(hidden_dim,), activation="relu")
        self.node_update = MLP(hidden_dim, 1, hidden_dims=(hidden_dim,), activation="relu")
        self._last_x: Tensor | None = None
        self._last_edge_probs: Tensor | None = None
        self._last_edge_probs_pre: Tensor | None = None
        self._last_edge_probs_post: Tensor | None = None

    # -- encoder / decoder ---------------------------------------------------

    def _windows(self, x: Tensor) -> Tensor:
        """Partition ``(B, T, D)`` into ``(B, n_windows, W, D)`` windows."""
        batch, horizon, dim = x.shape
        pad = (-horizon) % self.window
        if pad:
            x = torch.cat([x, x[:, -1:].expand(-1, pad, -1)], dim=1)
        return x.reshape(batch, -1, self.window, dim)

    def _pair(self, h: Tensor) -> Tensor:
        """Concatenate ``[h_i, h_j]`` for every ordered node pair ``(i, j)``."""
        nodes = h.shape[1]
        left = h.unsqueeze(2).expand(-1, nodes, nodes, -1)
        right = h.unsqueeze(1).expand(-1, nodes, nodes, -1)
        return torch.cat([left, right], dim=-1)

    def _encode_windows(self, node_features: Tensor, regime: Tensor | None = None) -> Tensor:
        if regime is not None:
            regime = regime.reshape(-1)
            node_features = torch.cat(
                [
                    node_features,
                    regime[:, None, None].expand(-1, node_features.shape[1], 1),
                ],
                dim=-1,
            )
        else:
            node_features = torch.cat(
                [node_features, torch.full_like(node_features[..., :1], 0.5)], dim=-1
            )
        hidden = self.node_embed(node_features)  # (M, D, H)
        edge1 = self.edge_embed1(self._pair(hidden))
        node2 = self.node_embed2((edge1 * self._offdiag.unsqueeze(-1)).sum(dim=1))
        edge2 = self.edge_embed2(self._pair(node2))
        return cast(Tensor, self.edge_out(edge2))

    def _pair_inputs(self, x_t: Tensor) -> Tensor:
        nodes = self.n_nodes
        left = x_t.unsqueeze(2).expand(-1, nodes, nodes)
        right = x_t.unsqueeze(1).expand(-1, nodes, nodes)
        return torch.stack([left, right], dim=-1)

    def _decode(self, x_t: Tensor, edge_weights: Tensor) -> Tensor:
        """Conditional mean ``mu_{t+1}`` given per-edge type probabilities."""
        pair = self._pair_inputs(x_t)
        messages = edge_weights[..., 1:2] * self.msg_pre(pair) + edge_weights[
            ..., 2:3
        ] * self.msg_post(pair)
        aggregate = (messages * self._offdiag.unsqueeze(-1)).sum(dim=1)
        return cast(Tensor, x_t + self.node_update(aggregate).squeeze(-1))

    def _conditional_mean(self, x_t: Tensor, weight: Tensor, edge_type: int) -> Tensor:
        """Mean under one mechanism with a fixed ``(D, D)`` causal weight matrix."""
        net = self.msg_pre if edge_type == 1 else self.msg_post
        pair = self._pair_inputs(x_t)
        messages = weight.unsqueeze(-1) * net(pair)
        aggregate = (messages * self._offdiag.unsqueeze(-1)).sum(dim=1)
        return cast(Tensor, x_t + self.node_update(aggregate).squeeze(-1))

    # -- forward / loss ------------------------------------------------------

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        x = batch.x
        batch_size, horizon, nodes = x.shape
        windows = self._windows(x)
        n_windows = windows.shape[1]
        node_features = windows.permute(0, 1, 3, 2).reshape(-1, nodes, self.window)
        window_regime = self._window_regime(batch, n_windows)
        if window_regime is not None:
            window_regime = window_regime.reshape(-1)
        window_logits = self._encode_windows(node_features, window_regime).reshape(
            batch_size, n_windows, nodes, nodes, self.n_edge_types
        )
        window_probs = F.softmax(window_logits, dim=-1)
        sample = (
            F.gumbel_softmax(window_logits, tau=self.tau, hard=False, dim=-1)
            if self.training
            else window_probs
        )
        recon = self._reconstruct(x, sample)
        pre_probs, post_probs = self._regime_posteriors(window_probs, batch)
        edge_logits = window_logits.mean(dim=1)
        edge_probs = window_probs.mean(dim=1)
        self._last_x = x.detach()
        self._last_edge_probs = window_probs.detach()
        self._last_edge_probs_pre = pre_probs.detach()
        self._last_edge_probs_post = post_probs.detach()
        return {
            "recon": recon,
            "edge_logits": edge_logits,
            "edge_probs": edge_probs,
            "window_logits": window_logits,
            "window_probs": window_probs,
            "edge_probs_pre": pre_probs,
            "edge_probs_post": post_probs,
        }

    def _reconstruct(self, x: Tensor, edge_sample: Tensor) -> Tensor:
        batch, horizon, nodes = x.shape
        if horizon < 2:
            return x.clone()
        n_windows = edge_sample.shape[1]
        window_index = torch.arange(horizon - 1, device=x.device) // self.window
        window_index = window_index.clamp_max(n_windows - 1)
        weights = edge_sample[:, window_index]  # (B, T-1, D, D, E)
        flat = batch * (horizon - 1)
        mean = self._decode(x[:, :-1].reshape(flat, nodes), weights.reshape(flat, nodes, nodes, -1))
        return torch.cat([x[:, :1], mean.reshape(batch, horizon - 1, nodes)], dim=1)

    def _window_regime(self, batch: Batch, n_windows: int) -> Tensor | None:
        """Mean post-change indicator per window, when the regime is observed."""
        regime = batch.context.get("regime") if batch.context is not None else None
        if regime is None or regime.ndim != 2:
            return None
        needed = n_windows * self.window
        flat = regime.float()
        if flat.shape[1] < needed:
            flat = torch.cat([flat, flat[:, -1:].expand(-1, needed - flat.shape[1])], dim=1)
        window = flat[:, :needed].reshape(flat.shape[0], n_windows, self.window)
        return cast(Tensor, window.mean(dim=-1))

    def _regime_posteriors(self, probs: Tensor, batch: Batch) -> tuple[Tensor, Tensor]:
        """Average edge posteriors over pre- and post-change windows per sample."""
        batch_size, n_windows = probs.shape[:2]
        regime = batch.context.get("regime") if batch.context is not None else None
        if regime is not None and regime.ndim == 2 and regime.shape[0] == batch_size:
            flat = regime.float()
            needed = n_windows * self.window
            if flat.shape[1] < needed:
                flat = torch.cat([flat, flat[:, -1:].expand(-1, needed - flat.shape[1])], dim=1)
            window_regime = flat[:, :needed].reshape(batch_size, n_windows, self.window)
            pre_mask = window_regime.mean(dim=-1) < 0.5
            post_mask = ~pre_mask
        elif n_windows == 1:
            pre_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=probs.device)
            post_mask = pre_mask
        else:
            half = max(n_windows // 2, 1)
            index = torch.arange(n_windows, device=probs.device)
            pre_mask = (index < half).expand(batch_size, -1)
            post_mask = ~pre_mask
        return self._masked_mean(probs, pre_mask), self._masked_mean(probs, post_mask)

    @staticmethod
    def _masked_mean(probs: Tensor, mask: Tensor) -> Tensor:
        weights = mask.float().view(*mask.shape, 1, 1, 1)
        mean = (probs * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        empty = (mask.sum(dim=1) == 0).view(-1, 1, 1, 1)
        return torch.where(empty, probs.mean(dim=1), mean)

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        x = batch.x
        recon = (
            F.mse_loss(outputs["recon"][:, 1:], x[:, 1:])
            if x.shape[1] > 1
            else torch.zeros((), device=x.device)
        )
        logits = outputs["window_logits"]
        log_q = F.log_softmax(logits, dim=-1)
        log_prior = torch.log(self.prior).to(logits.device)
        edge_kl = (log_q.exp() * (log_q - log_prior)).sum(dim=-1)
        offdiag = self._offdiag.to(logits.device)
        norm = float(self.n_nodes * (self.n_nodes - 1))
        kl = (edge_kl * offdiag).sum(dim=(-2, -1)).div(norm).mean()
        probs = outputs["window_probs"]
        groups = torch.sqrt(probs[..., 1].pow(2) + probs[..., 2].pow(2))
        pcm = (groups * offdiag).sum(dim=(-2, -1)).div(norm).mean()
        total = self.recon_weight * recon + self.kl_weight * kl + self.pcm_weight * pcm
        return {"loss": total, "recon": recon, "kl": kl, "pcm": pcm}

    # -- root-cause analysis -------------------------------------------------

    @staticmethod
    def _flatten_posterior(posterior: Tensor) -> Tensor:
        if posterior.ndim < 3 or posterior.shape[-1] != 3:
            raise ValueError("Expected an edge posterior of shape (..., D, D, 3).")
        return posterior.reshape(-1, posterior.shape[-3], posterior.shape[-2], 3)

    def _edge_kl(self, pre: Tensor, post: Tensor) -> Tensor:
        """Paper-literal ``KL(q_pre(a_ij) || q_post(a_ij))`` summed over parents."""
        q_pre = pre.clamp_min(1e-8)
        q_post = post.clamp_min(1e-8)
        per_edge = (q_pre * (q_pre.log() - q_post.log())).sum(dim=-1)
        return (per_edge * self._offdiag.to(per_edge.device)).sum(dim=0)

    def root_causes(
        self, pre: Tensor, post: Tensor, parents: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Score every node's mechanism change.

        Args:
            pre: pre-change edge posterior ``(..., D, D, 3)``.
            post: post-change edge posterior ``(..., D, D, 3)``.
            parents: optional parent observations ``(B, T, D)``, ``(T, D)`` or
                ``(N, D)``.  When omitted, the most recent ``forward`` batch is
                used.

        Returns:
            A dictionary with the primary ``scores`` (and its alias
            ``root_cause_scores``) plus the secondary ``kl`` (alias ``edge_kl``).
        """
        lead = tuple(pre.shape[:-3])
        if lead != post.shape[:-3]:
            pre = pre.reshape(-1, *pre.shape[-3:]).mean(dim=0)
            post = post.reshape(-1, *post.shape[-3:]).mean(dim=0)
            lead = ()
        pre_flat = self._flatten_posterior(pre)
        post_flat = self._flatten_posterior(post)
        n_scores = pre_flat.shape[0]
        nodes = self.n_nodes
        x = self._last_x if parents is None else parents
        if x is None:
            raise ValueError("parents are required before the first forward pass.")
        x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            if x.shape[0] == n_scores:
                parent_sets = [x[i].reshape(-1, nodes) for i in range(n_scores)]
            elif n_scores == 1:
                parent_sets = [x.reshape(-1, nodes)]
            else:
                parent_sets = [x.mean(dim=0).reshape(-1, nodes)] * n_scores
        elif x.ndim == 2:
            if x.shape[0] == n_scores:
                parent_sets = [x[i : i + 1] for i in range(n_scores)]
            else:
                parent_sets = [x] * n_scores
        else:
            raise ValueError("parents must have shape (B, T, D), (T, D) or (N, D).")

        scores, kls = [], []
        for index in range(n_scores):
            x_t = parent_sets[index]
            mu_pre = self._conditional_mean(x_t, pre_flat[index, ..., 1], 1)
            mu_post = self._conditional_mean(x_t, post_flat[index, ..., 2], 2)
            scores.append((mu_pre - mu_post).pow(2).mean(dim=0))
            kls.append(self._edge_kl(pre_flat[index], post_flat[index]))
        score_tensor = torch.stack(scores, dim=0)
        kl_tensor = torch.stack(kls, dim=0)
        if lead:
            score_tensor = score_tensor.reshape(*lead, nodes)
            kl_tensor = kl_tensor.reshape(*lead, nodes)
        else:
            score_tensor = score_tensor.squeeze(0)
            kl_tensor = kl_tensor.squeeze(0)
        return {
            "scores": score_tensor,
            "root_cause_scores": score_tensor,
            "kl": kl_tensor,
            "edge_kl": kl_tensor,
        }

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def sample_paths(self, batch: Batch, n_samples: int = 1) -> dict[str, Tensor]:
        outputs = self.forward(batch)
        batch_size, _, nodes = batch.x.shape
        pre, post = outputs["edge_probs_pre"], outputs["edge_probs_post"]
        scores = torch.empty(batch_size, nodes, device=batch.x.device)
        for index in range(batch_size):
            scores[index] = self.root_causes(pre[index], post[index], parents=batch.x[index])[
                "scores"
            ]
        sources = scores.argmax(dim=1)
        paths = torch.full(
            (batch_size, self.max_path_len + 1),
            -1,
            dtype=torch.long,
            device=batch.x.device,
        )
        paths[:, 0] = sources
        strength = (outputs["window_probs"][..., 1] + outputs["window_probs"][..., 2]).mean(dim=1)
        strength = strength * self._offdiag.to(strength.device)
        for index in range(batch_size):
            current = int(sources[index])
            visited = {current}
            for step in range(1, self.max_path_len + 1):
                row = strength[index, current].clone()
                row[list(visited)] = -1.0
                if float(row.max()) <= 0.0:
                    break
                current = int(row.argmax())
                paths[index, step] = current
                visited.add(current)
        return {"paths": paths, "sources": sources, "root_cause_scores": scores}

    def get_latent_adjacency(self) -> Tensor | None:
        """``(D, D, 1)`` causal graph from the argmax causal edge type."""
        if self._last_edge_probs is None:
            return torch.zeros(self.n_nodes, self.n_nodes, 1)
        probs = self._last_edge_probs.reshape(
            -1, self.n_nodes, self.n_nodes, self.n_edge_types
        ).mean(dim=0)
        adjacency = (probs.argmax(dim=-1) != 0).float()
        adjacency.fill_diagonal_(0.0)
        return adjacency.unsqueeze(-1)

    def identifiability_statement(self) -> str:
        return (
            "RCA-VAE recovers a regime-indexed edge posterior and ranks nodes by "
            "the change in their decoder conditional mean; the graph is identified "
            "only up to the observational Markov equivalence class and the assumed "
            "regime split, not by an interventional identifiability theorem."
        )
