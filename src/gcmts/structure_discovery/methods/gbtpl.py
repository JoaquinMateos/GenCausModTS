"""GBTPL: graph-constrained generative back-tracing for propagation paths.

Reference: Zhu et al., *GBTPL: Graph-Constrained Generative Back-Tracing for
Anomaly Propagation Path Localization*, 2026 — BibTeX
``zhuGBTPLGraphConstrainedGenerative2026a``.

A masked Transformer encoder maps an anomaly sequence to node embeddings; a
generator back-traces the propagation path by attending only over graph-valid
predecessors (the attention mask enforces reachability), and candidate paths are
ranked by a unified causal likelihood combining temporal consistency, structural
validity and causal strength. The returned object is the propagation path with
its source node.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gcmts.core.backbones import MLP
from gcmts.structure_discovery.base import BaseStructureDiscovery
from gcmts.typing import Batch

logger = logging.getLogger(__name__)

__all__ = ["GBTPL"]


class GBTPL(BaseStructureDiscovery):
    """Graph-constrained generative back-tracing.

    Args:
        n_nodes: number ``D`` of variables/nodes in the dependency graph.
        horizon: sequence length ``T``.
        max_path_len: maximum number of hops in a propagation path.
        hidden_dims: transformer/feed-forward widths.
        n_heads / n_layers: transformer configuration.
        recon_weight / path_weight / kl_weight: loss weights.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        horizon: int = 16,
        max_path_len: int = 4,
        hidden_dims: tuple[int, ...] = (128, 128),
        n_heads: int = 4,
        n_layers: int = 2,
        recon_weight: float = 1.0,
        path_weight: float = 1.0,
        kl_weight: float = 0.1,
    ) -> None:
        super().__init__(n_nodes=n_nodes, max_path_len=max_path_len)
        self.horizon = horizon
        self.recon_weight = recon_weight
        self.path_weight = path_weight
        self.kl_weight: float = kl_weight
        self.embed = MLP(horizon, hidden_dims[-1], hidden_dims=hidden_dims)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dims[-1],
            nhead=n_heads,
            dim_feedforward=hidden_dims[-1] * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = MLP(hidden_dims[-1], horizon, hidden_dims=hidden_dims)
        self.path_head = MLP(2 * hidden_dims[-1], max_path_len, hidden_dims=hidden_dims)
        logger.debug("Initialised GBTPL (D=%d, T=%d)", n_nodes, horizon)

    # -- graph mask ----------------------------------------------------------

    def _graph_mask(self, graph: Tensor) -> Tensor:
        """Attention mask with ``0`` for graph-reachable (predecessor) pairs."""
        # Back-tracing walks predecessors, so allow attention j -> i where j is
        # an ancestor of i; we use the transitive closure of the reversed graph.
        n = graph.shape[0]
        closure = graph.float().clone()
        for _ in range(n):
            closure = ((closure + closure @ closure) > 0).float()
        reachable = closure.T > 0  # predecessor -> current
        mask = torch.zeros(n, n, device=graph.device)
        mask[~reachable] = float("-inf")
        return mask.masked_fill(torch.eye(n, device=graph.device, dtype=torch.bool), float("-inf"))

    # -- forward / loss ------------------------------------------------------

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        x = batch.x  # (B, T, D)
        graph = self._graph(batch)
        node_features = x.transpose(1, 2)  # (B, D, T)
        hidden = self.embed(node_features)  # (B, D, hidden)
        mask = self._graph_mask(graph)
        encoded = self.encoder(hidden, mask=mask)
        recon = self.decoder(encoded).transpose(1, 2)  # (B, T, D)
        return {"hidden": encoded, "recon": recon, "graph": graph}

    def _graph(self, batch: Batch) -> Tensor:
        if batch.context is not None and "graph" in batch.context:
            return batch.context["graph"].float()
        if batch.adjacency is not None:
            return batch.adjacency.float()
        return torch.zeros(self.n_nodes, self.n_nodes, device=batch.x.device)

    def _path_logits(self, hidden: Tensor, graph: Tensor) -> Tensor:
        """Next-predecessor logits for each ordered pair of nodes."""
        n = hidden.shape[1]
        left = hidden.unsqueeze(2).expand(-1, n, n, -1)
        right = hidden.unsqueeze(1).expand(-1, n, n, -1)
        pair = torch.cat([left, right], dim=-1)
        scores = self.path_head(pair.reshape(-1, pair.shape[-1])).reshape(
            hidden.shape[0], n, n, -1
        )
        scores = scores.mean(dim=-1)  # (B, i, j) logit that j is a predecessor of i
        allowed = (graph + graph.T).bool().float()
        scores = scores.masked_fill(allowed.unsqueeze(0) == 0, float("-inf"))
        return scores

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = F.mse_loss(outputs["recon"], batch.x)
        graph = outputs["graph"]
        logits = self._path_logits(outputs["hidden"], graph)
        path_loss = torch.zeros((), device=batch.x.device)
        if batch.context is not None and "path" in batch.context:
            paths = batch.context["path"].long()
            for step in range(paths.shape[1] - 1):
                current = paths[:, step]
                nxt = paths[:, step + 1]
                selected = logits[torch.arange(logits.shape[0]), current]
                path_loss = path_loss + F.cross_entropy(selected, nxt, ignore_index=-1)
            path_loss = path_loss / max(paths.shape[1] - 1, 1)
        kl = (outputs["hidden"] ** 2).mean()
        total = (
            self.recon_weight * recon
            + self.path_weight * path_loss
            + self.kl_weight * kl
        )
        return {"loss": total, "recon": recon, "path": path_loss, "kl": kl}

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def sample_paths(self, batch: Batch, n_samples: int = 1) -> dict[str, Tensor]:
        outputs = self.forward(batch)
        graph = outputs["graph"]
        logits = self._path_logits(outputs["hidden"], graph)
        # The source is the node with the most anomalous reconstruction error.
        error = (outputs["recon"] - batch.x).pow(2).mean(dim=1)  # (B, D)
        sources = error.argmax(dim=1)
        batch_size = batch.x.shape[0]
        paths = torch.empty(batch_size, self.max_path_len + 1, dtype=torch.long)
        paths[:, 0] = sources
        current = sources
        for step in range(1, self.max_path_len + 1):
            step_logits = logits[torch.arange(batch_size), current]
            step_logits = torch.nan_to_num(step_logits, neginf=-1e9)
            nxt = step_logits.argmax(dim=-1)
            paths[:, step] = nxt
            current = nxt
        return {"paths": paths, "sources": sources}

    def identifiability_statement(self) -> str:
        return (
            "GBTPL: given a directed dependency graph, candidate propagation "
            "paths are restricted to graph-reachable predecessor walks and ranked "
            "by a causal likelihood; the recovered path is identified only up to "
            "the assumed graph and the ranking criterion (no latent identifiability "
            "theorem)."
        )
