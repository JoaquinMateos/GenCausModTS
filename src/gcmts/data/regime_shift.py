"""Regime-shift propagation benchmark for RCA-VAE (structure discovery, III.B).

Each trajectory follows a linear-Gaussian lag-1 SCM

.. math:: x_{t+1} = A x_t + u_t, \\qquad u_t \\sim \\mathcal N(0, \\sigma^2 I),

with a stable, row-normalised matrix ``A``.  One change point is placed in the
middle third of the sequence.  Only the rows of a small root-cause set ``S``
(one or two nodes) change after that point, so the post-change joint
distribution differs from the pre-change one only through the mechanisms of
those nodes.  The benchmark exposes the graph, the root-cause indicator, the
change point, the regime label, and a propagation path/source for structural
metrics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["RegimeShiftPropagationGenerator"]


class RegimeShiftPropagationGenerator(BaseDataGenerator):
    """Linear-Gaussian lag-1 SCM with a sparse, mechanism-level regime shift.

    Args:
        n_nodes: number ``D`` of observed variables (default ``8``).
        horizon: sequence length ``T`` (default ``32``).
        max_path_len: maximum number of hops in the exposed propagation path.
        edge_prob: probability of a lower-triangular edge ``i -> j``.
        noise_std: standard deviation of the Gaussian innovations.
        min_root_causes / max_root_causes: size range of the changed set ``S``.
        seed: RNG seed; the generator is fully deterministic given a seed.
    """

    def __init__(
        self,
        *,
        n_nodes: int = 8,
        horizon: int = 32,
        max_path_len: int = 4,
        edge_prob: float = 0.35,
        noise_std: float = 0.1,
        min_root_causes: int = 1,
        max_root_causes: int = 2,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=n_nodes, horizon=horizon)
        if not 1 <= min_root_causes <= max_root_causes <= n_nodes:
            raise ValueError("root-cause set size must lie in [1, n_nodes].")
        self.n_nodes = n_nodes
        self.max_path_len = max_path_len
        self.noise_std = noise_std
        self.min_root_causes = min_root_causes
        self.max_root_causes = max_root_causes
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.base_graph = self._sample_graph(edge_prob)
        self.pre_change_matrix = self._normalise(
            torch.rand(n_nodes, n_nodes, generator=self.rng) * self.base_graph
            + torch.diag(0.1 + 0.2 * torch.rand(n_nodes, generator=self.rng))
        )
        self.graph = (self.pre_change_matrix.abs() > 0).float()
        # A deterministic post-change mechanism per node.  The previous version
        # redrew the new row per trajectory, so the "changed" mechanism differed
        # across samples and the root cause was not identifiable; the paper
        # compares two fixed mechanisms (before/after the event).
        self.changed_rows = self._changed_rows()
        # The paper compares one system before and after the event, so the
        # changed mechanism set is global to the dataset (not redrawn per
        # trajectory); the event time may still vary per trajectory.
        n_roots = int(
            torch.randint(self.min_root_causes, self.max_root_causes + 1, (1,), generator=self.rng)
        )
        self.root_causes = torch.sort(
            torch.randperm(n_nodes, generator=self.rng)[:n_roots]
        ).values
        self.post_change_matrix = self._post_matrix(self.root_causes)

    # -- structure -----------------------------------------------------------

    def _sample_graph(self, edge_prob: float) -> Tensor:
        """Lower-triangular DAG with self-loops and at least one parent per node."""
        lower = torch.tril(torch.ones(self.n_nodes, self.n_nodes), diagonal=-1)
        graph = torch.rand(self.n_nodes, self.n_nodes, generator=self.rng) < edge_prob
        graph = graph.float() * lower + torch.eye(self.n_nodes)
        for i in range(1, self.n_nodes):
            if float(graph[i].sum()) == 1.0:  # only the self-loop
                parent = int(torch.randint(0, i, (1,), generator=self.rng))
                graph[i, parent] = 1.0
        return graph

    @staticmethod
    def _normalise(matrix: Tensor) -> Tensor:
        """Row L1-normalise to ``0.9``; this bounds the spectral radius by 0.9."""
        return 0.9 * matrix / matrix.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _changed_rows(self) -> Tensor:
        """Deterministic post-change row for every node (same across samples)."""
        rows = self.pre_change_matrix.clone()
        for node in range(self.n_nodes):
            row = rows[node].clone()
            target = (node + 1) % self.n_nodes
            if target == node:
                target = (node + 2) % self.n_nodes
            row = 0.4 * row
            row[target] += 0.6
            rows[node] = self._normalise(row)
        return rows

    def _post_matrix(self, roots: Tensor) -> Tensor:
        """Return ``A`` with only the rows in ``roots`` changing mechanism."""
        matrix = self.pre_change_matrix.clone()
        for node in roots.tolist():
            matrix[node] = self.changed_rows[node]
        return matrix

    # -- data ----------------------------------------------------------------

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        low = max(1, horizon // 3)
        high = max(low + 1, (2 * horizon) // 3)
        x = torch.empty(n_samples, horizon, self.n_nodes)
        root_causes = torch.zeros(n_samples, self.n_nodes, dtype=torch.bool)
        change_point = torch.empty(n_samples, dtype=torch.long)
        regime = torch.zeros(n_samples, horizon, dtype=torch.long)
        sources = torch.empty(n_samples, dtype=torch.long)
        paths = torch.full((n_samples, self.max_path_len + 1), -1, dtype=torch.long)
        for b in range(n_samples):
            roots = self.root_causes
            root_causes[b, roots] = True
            cp = int(torch.randint(low, high, (1,), generator=self.rng))
            change_point[b] = cp
            regime[b, cp:] = 1
            post = self.post_change_matrix
            x[b, 0] = torch.randn(self.n_nodes, generator=self.rng)
            for t in range(horizon - 1):
                matrix = post if t >= cp else self.pre_change_matrix
                x[b, t + 1] = matrix @ x[b, t] + self.noise_std * torch.randn(
                    self.n_nodes, generator=self.rng
                )
            source = int(roots[torch.randint(0, len(roots), (1,), generator=self.rng)])
            sources[b] = source
            path = self._sample_path(source)
            paths[b, : path.shape[0]] = path
        return Batch(
            x=x,
            context={
                "graph": self.graph,
                "root_causes": root_causes,
                "change_point": change_point,
                "regime": regime,
                "source": sources,
                "path": paths,
            },
        )

    def _sample_path(self, source: int) -> Tensor:
        path = [source]
        current = source
        length = int(torch.randint(1, self.max_path_len + 1, (1,), generator=self.rng))
        for _ in range(length):
            children = (self.base_graph[current] > 0).nonzero().flatten()
            children = children[children != current]
            if children.numel() == 0:
                break
            current = int(children[torch.randint(0, len(children), (1,), generator=self.rng)])
            path.append(current)
        return torch.tensor(path)
