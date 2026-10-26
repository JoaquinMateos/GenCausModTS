"""iCITRIS: causal representation learning for instantaneous and temporal effects.

Reference: Lippe et al., *Causal Representation Learning for Instantaneous and
Temporal Effects in Interactive Systems*, ICLR 2023 — BibTeX
``lippeCausalRepresentationLearning2023`` (extends CITRIS, 2022).

An invertible autoencoder maps observations to ``M`` latent components assigned
to ``K`` causal variables plus one shared group. Extending CITRIS, the
conditional transition prior also conditions on the *instantaneous* parents of
each group,

    p(z^{t+1} | z^t, I^{t+1}) =
        prod_i p(z_{psi_i}^{t+1} | z^t, z_{Pa_G(i)}^{t+1}, I_i^{t+1}),

with a learned DAG ``G`` over the groups. Minimising the information in the
shared group, maximising the entropy of the intervention-dependent factors and
enforcing acyclicity (NOTEARS) recovers the minimal causal variables up to
invertible transforms and the full instantaneous + temporal graph. Interventions
are *partially perfect*: an intervened group drops its instantaneous parents but
keeps its temporal ones.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import MLP, GradientReversal, RealNVP, gaussian_logpdf
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["iCITRIS"]


def _straight_through_bernoulli(logits: Tensor, temperature: float) -> Tensor:
    """Hard {0,1} sample with a straight-through Gumbel-sigmoid gradient."""
    soft = torch.sigmoid(logits / temperature)
    hard = (soft > 0.5).to(soft.dtype)
    return hard.detach() - soft.detach() + soft


class iCITRIS(BaseCausalRepresentationLearner):  # noqa: N801
    """Intervention-conditioned CRL with instantaneous and temporal effects.

    Args:
        observed_dim: dimension ``D`` of ``x`` (equal to ``M`` because the
            autoencoder is invertible).
        n_vars: number ``K`` of causal variables.
        var_dim: dimension of each causal variable group.
        n_shared: dimension of the shared (intervention-independent) group.
        n_flow_layers: coupling layers of the invertible autoencoder.
        hidden_dims: hidden widths of the coupling/transition/classifier nets.
        cls_weight: weight of the shared-group intervention classifier.
        recon_weight: weight of the explicit reconstruction penalty.
        dag_weight: weight of the NOTEARS acyclicity penalty.
        sparsity_weight: weight of the graph L1 sparsity penalty.
        temperature: Gumbel-sigmoid temperature for the straight-through gates.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        n_vars: int,
        var_dim: int,
        n_shared: int = 1,
        n_flow_layers: int = 6,
        hidden_dims: tuple[int, ...] = (128, 128),
        cls_weight: float = 1.0,
        recon_weight: float = 1.0,
        dag_weight: float = 1.0,
        sparsity_weight: float = 0.1,
        temperature: float = 0.5,
    ) -> None:
        latent_dim = n_vars * var_dim + n_shared
        if latent_dim != observed_dim:
            raise ValueError(
                "iCITRIS uses an invertible autoencoder, so n_vars*var_dim + "
                f"n_shared must equal observed_dim (got {latent_dim} vs {observed_dim})."
            )
        super().__init__(observed_dim=observed_dim, latent_dim=latent_dim, max_lag=1)
        self.n_vars = n_vars
        self.var_dim = var_dim
        self.n_shared = n_shared
        self.n_groups = n_vars + (1 if n_shared > 0 else 0)
        self.cls_weight = cls_weight
        self.recon_weight = recon_weight
        self.dag_weight = dag_weight
        self.sparsity_weight = sparsity_weight
        self.temperature = temperature
        self.flow = RealNVP(observed_dim, n_layers=n_flow_layers, hidden_dims=hidden_dims)

        # Instantaneous adjacency over the K causal groups: strictly lower
        # triangular in the causal order, so it is acyclic by construction.
        self.instantaneous_logits = nn.Parameter(
            -2.0
            * torch.ones(n_vars, n_vars)
            * torch.tril(torch.ones(n_vars, n_vars), diagonal=-1)
        )
        # Temporal adjacency over all groups (row = parent at t-1, col = child).
        self.temporal_logits = nn.Parameter(torch.full((self.n_groups, self.n_groups), -1.0))

        cond_dim = observed_dim + (n_vars - 1) * var_dim + 1
        self.causal_nets = nn.ModuleList(
            MLP(cond_dim, 2 * var_dim, hidden_dims=hidden_dims) for _ in range(n_vars)
        )
        self.shared_net = (
            MLP(observed_dim, 2 * n_shared, hidden_dims=hidden_dims) if n_shared > 0 else None
        )
        self.intervention_mean = nn.Parameter(torch.zeros(var_dim))
        self.intervention_logvar = nn.Parameter(torch.zeros(var_dim))
        self.classifier = (
            MLP(n_shared, n_vars, hidden_dims=hidden_dims) if n_shared > 0 else None
        )
        self.grl = GradientReversal(1.0)
        logger.debug("Initialised iCITRIS (D=%d, K=%d, var_dim=%d)", observed_dim, n_vars, var_dim)

    # -- group helpers -------------------------------------------------------

    def _groups(self) -> list[slice]:
        groups = [slice(i * self.var_dim, (i + 1) * self.var_dim) for i in range(self.n_vars)]
        if self.n_shared > 0:
            groups.append(slice(self.n_vars * self.var_dim, self.latent_dim))
        return groups

    def _gates(self) -> tuple[Tensor, Tensor]:
        inst = _straight_through_bernoulli(self.instantaneous_logits, self.temperature)
        inst = torch.tril(inst, diagonal=-1)
        temp = _straight_through_bernoulli(self.temporal_logits, self.temperature)
        return inst, temp

    # -- encode / decode -----------------------------------------------------

    @staticmethod
    def _flatten(value: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if value.ndim == 2:
            return value, (value.shape[0], 1)
        batch, time, dim = value.shape
        return value.reshape(batch * time, dim), (batch, time)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        z_flat, (batch, time) = self._flatten(x)
        z, _ = self.flow.forward(z_flat)
        return z.reshape(batch, time, -1)

    def decode(self, z: Tensor) -> Tensor:
        z_flat, (batch, time) = self._flatten(z)
        return self.flow.inverse(z_flat).reshape(batch, time, -1)

    # -- transition prior ----------------------------------------------------

    def _transition_log_prob(
        self, z_seq: Tensor, interventions: Tensor | None, inst: Tensor, temp: Tensor
    ) -> Tensor:
        """Sum of transition log-densities for ``t = 1..T-1`` per trajectory."""
        batch, time, _ = z_seq.shape
        device = z_seq.device
        if time <= 1:
            return torch.zeros(batch, device=device, dtype=z_seq.dtype)
        groups = self._groups()
        log_prob = torch.zeros(batch, time - 1, device=device, dtype=z_seq.dtype)
        for t in range(1, time):
            previous = z_seq[:, t - 1]
            current = z_seq[:, t]
            for index, group in enumerate(groups[:-1] if self.n_shared > 0 else groups):
                parents = [
                    inst[j, index] * current[:, groups[j]] for j in range(self.n_vars) if j != index
                ]
                parent_value = (
                    torch.cat(parents, dim=-1)
                    if parents
                    else torch.zeros(batch, 0, device=device, dtype=z_seq.dtype)
                )
                masked_previous = previous * self._group_expand(temp[:, index])
                target = (
                    interventions[:, t, index].unsqueeze(-1)
                    if interventions is not None
                    else torch.zeros(batch, 1, device=device, dtype=z_seq.dtype)
                )
                cond = torch.cat([masked_previous, parent_value, target], dim=-1)
                mean, logvar = self.causal_nets[index](cond).chunk(2, dim=-1)
                if interventions is not None:
                    mask = interventions[:, t, index].unsqueeze(-1)
                    mean = mask * self.intervention_mean + (1.0 - mask) * mean
                    logvar = mask * self.intervention_logvar + (1.0 - mask) * logvar
                log_prob[:, t - 1] = log_prob[:, t - 1] + gaussian_logpdf(
                    current[:, group], mean, logvar.clamp(-12.0, 4.0)
                )
            if self.n_shared > 0 and self.shared_net is not None:
                masked_previous = previous * self._group_expand(temp[:, -1])
                shared_mean, shared_logvar = self.shared_net(masked_previous).chunk(2, dim=-1)
                log_prob[:, t - 1] = log_prob[:, t - 1] + gaussian_logpdf(
                    current[:, groups[-1]],
                    shared_mean,
                    shared_logvar.clamp(-12.0, 4.0),
                )
        return log_prob.sum(dim=1)

    def _group_expand(self, group_gates: Tensor) -> Tensor:
        """Expand per-group temporal gates ``(n_groups,)`` to a latent mask ``(M,)``."""
        sizes = [self.var_dim] * self.n_vars + ([self.n_shared] if self.n_shared > 0 else [])
        return torch.cat(
            [group_gates[i].expand(size) for i, size in enumerate(sizes)],
            dim=-1,
        )

    # -- forward / loss ------------------------------------------------------

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        x = batch.x
        batch_size, time, dim = x.shape
        x_flat = x.reshape(batch_size * time, dim)
        z_flat, logdet = self.flow.forward(x_flat)
        z_seq = z_flat.reshape(batch_size, time, dim)
        interventions = None
        if batch.context is not None and "I" in batch.context:
            interventions = batch.context["I"].to(x.device)
            if interventions.ndim != 3 or interventions.shape[-1] != self.n_vars:
                raise ValueError(
                    "iCITRIS expects intervention targets of shape (B, T, n_vars); "
                    f"got {tuple(interventions.shape)} for n_vars={self.n_vars}."
                )
        inst, temp = self._gates()

        zeros = torch.zeros(dim, device=x.device, dtype=x.dtype)
        log_prob = gaussian_logpdf(z_seq[:, 0], zeros, zeros)
        log_prob = log_prob + self._transition_log_prob(z_seq, interventions, inst, temp)
        log_prob = log_prob + logdet.reshape(batch_size, time).sum(dim=1)
        nll = -log_prob.mean()

        cls_loss = torch.zeros((), device=x.device)
        if self.classifier is not None and interventions is not None and time > 1:
            shared = z_seq[:, 1:, self._groups()[-1]].reshape(
                batch_size * (time - 1), self.n_shared
            )
            logits = self.classifier(self.grl(shared)).reshape(batch_size, time - 1, self.n_vars)
            cls_loss = F.binary_cross_entropy_with_logits(logits, interventions[:, 1:])

        dag = self._notears(inst)
        sparsity = inst.abs().sum() + temp.abs().sum()
        x_recon = self.flow.inverse(z_flat).reshape(batch_size, time, dim)
        return CausalRepresentationOutput(
            z=z_seq,
            x_recon=x_recon,
            adjacency=self._expanded_adjacency(inst, temp),
            log_prob=log_prob,
            extras={
                "nll": nll,
                "cls": cls_loss,
                "dag": dag,
                "sparsity": sparsity,
                "inst": inst.detach(),
                "temp": temp.detach(),
                "logdet": logdet,
            },
        )

    @staticmethod
    def _notears(adjacency: Tensor) -> Tensor:
        """NOTEARS acyclicity ``tr(exp(A o A)) - d``; zero for a DAG."""
        d = adjacency.shape[0]
        squared = adjacency * adjacency
        return torch.trace(torch.matrix_exp(squared)) - d

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None:
            raise ValueError("iCITRIS forward output is missing the reconstruction.")
        extras = outputs.extras or {}
        nll = extras["nll"]
        recon = F.mse_loss(outputs.x_recon.reshape_as(batch.x), batch.x)
        total = (
            nll
            + self.recon_weight * recon
            + self.cls_weight * extras["cls"]
            + self.dag_weight * extras["dag"]
            + self.sparsity_weight * extras["sparsity"]
        )
        return {
            "loss": total,
            "nll": nll,
            "recon": recon,
            "cls": extras["cls"],
            "dag": extras["dag"],
            "sparsity": extras["sparsity"],
        }

    def _expanded_adjacency(self, inst: Tensor, temp: Tensor) -> Tensor:
        """Expand group gates to a ``(M, M, 2)`` latent adjacency (lag 0 / lag 1)."""
        latent_dim = self.latent_dim
        assert latent_dim is not None
        m = int(latent_dim)
        expanded = torch.zeros(m, m, 2, device=inst.device, dtype=inst.dtype)
        groups = self._groups()
        for i in range(self.n_groups):
            for j in range(self.n_groups):
                child, parent = groups[i], groups[j]
                if i < self.n_vars and j < self.n_vars and i != j:
                    value = inst[i, j] if i > j else inst[j, i]
                    expanded[parent, child, 0] = value
                expanded[parent, child, 1] = temp[j, i]
        return expanded

    def get_latent_adjacency(self) -> Tensor | None:
        inst, temp = self._gates()
        return self._expanded_adjacency(inst.detach(), temp.detach())

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        return z_past[:, -1]

    def identifiability_statement(self) -> str:
        return (
            "iCITRIS: under partially-perfect interventions with known, "
            "non-deterministically related targets, minimising the instantaneous "
            "edge count, maximising the entropy of the intervention-dependent "
            "factors and an acyclicity constraint identify the minimal causal "
            "variables up to invertible transforms and recover the full "
            "instantaneous + temporal causal graph (Thm. 3.4)."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
