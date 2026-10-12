"""Propagation-path generator for structure-discovery benchmarks (III.A).

A directed dependency graph over ``D`` nodes is sampled once. Each instance
starts an anomaly at a source node and propagates it along a graph-reachable
walk; node signals activate when the front reaches them. The ground-truth
propagation path and source are exposed for path-F1 and source-accuracy metrics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["PropagationGenerator"]


class PropagationGenerator(BaseDataGenerator):
    """Graph-constrained anomaly propagation.

    Args:
        n_nodes: number ``D`` of nodes.
        horizon: sequence length ``T``.
        max_path_len: maximum propagation path length.
        edge_prob: probability of a DAG edge ``j -> i`` (``j < i``).
        signal: anomaly magnitude added to activated nodes.
        noise_std: observation-noise std.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        n_nodes: int = 6,
        horizon: int = 16,
        max_path_len: int = 4,
        edge_prob: float = 0.4,
        signal: float = 2.0,
        noise_std: float = 0.2,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=n_nodes, horizon=horizon)
        self.n_nodes = n_nodes
        self.max_path_len = max_path_len
        self.signal = signal
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        lower = torch.tril(torch.ones(n_nodes, n_nodes), diagonal=-1)
        self.graph = (torch.rand(n_nodes, n_nodes, generator=self.rng) < edge_prob) * lower
        self.graph = self.graph.float()
        # Ensure every node has at least one parent except a designated root.
        for i in range(1, n_nodes):
            if self.graph[:, i].sum() == 0:
                self.graph[torch.randint(0, i, (1,), generator=self.rng), i] = 1.0

    def _sample_path(self) -> Tensor:
        children = {
            i: (self.graph[:, i] > 0).nonzero().flatten().tolist()
            for i in range(self.n_nodes)
        }
        source = int(torch.randint(0, self.n_nodes, (1,), generator=self.rng))
        path = [source]
        current = source
        length = int(torch.randint(1, self.max_path_len + 1, (1,), generator=self.rng))
        for _ in range(length):
            if not children[current]:
                break
            options = torch.tensor(children[current])
            nxt = int(options[torch.randint(0, len(options), (1,), generator=self.rng)])
            path.append(nxt)
            current = nxt
        return torch.tensor(path)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        x = self.noise_std * torch.randn(
            n_samples, horizon, self.n_nodes, generator=self.rng
        )
        paths = torch.full((n_samples, self.max_path_len + 1), -1, dtype=torch.long)
        sources = torch.zeros(n_samples, dtype=torch.long)
        for b in range(n_samples):
            path = self._sample_path()
            paths[b, : path.shape[0]] = path
            sources[b] = path[0]
            step = max(1, horizon // (path.shape[0] + 1))
            for order, node in enumerate(path.tolist()):
                start = order * step
                x[b, start:, node] += self.signal * (1.0 + 0.2 * order)
        return Batch(
            x=x,
            context={"graph": self.graph, "path": paths, "source": sources},
        )
