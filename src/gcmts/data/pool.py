"""Opt-in pooled-resampling generator (throughput mode).

.. warning::
   **Protocol-changing.** The benchmark normally draws *fresh* SCM trajectories
   on every optimisation step. This wrapper pre-generates one fixed pool and
   serves minibatches by resampling it, which removes nearly all per-step data
   generation (the current bottleneck) and can therefore raise GPU utilisation —
   but it changes the data seen at each step. Results are **not** comparable to a
   standard run and must be reported separately.

Use it only for throughput experiments, e.g. via ``run_crl_shared.py
--pool-size 65536`` (also read from ``GCMTS_POOL_SIZE``).
"""

from __future__ import annotations

from typing import Any, cast

import torch

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["PooledGenerator"]


def _index_batch(batch: Batch, indices: torch.Tensor) -> Batch:
    context = (
        {key: value[indices] for key, value in batch.context.items()}
        if batch.context
        else None
    )
    return Batch(
        x=batch.x[indices],
        z=None if batch.z is None else batch.z[indices],
        context=context,
        mask=None if batch.mask is None else batch.mask[indices],
        adjacency=batch.adjacency,
        counterfactual=None
        if batch.counterfactual is None
        else batch.counterfactual[indices],
        outcome=None if batch.outcome is None else batch.outcome[indices],
        domain=None if batch.domain is None else batch.domain[indices],
    )


class PooledGenerator(BaseDataGenerator):
    """Wrap a generator with a fixed pool resampled per minibatch.

    Args:
        generator: the underlying (fresh-sample) generator.
        pool_size: number of trajectories to pre-generate once.
        resample_seed: seed for the minibatch index sampler (independent of the
            underlying generator's own RNG).
        device: optional device to keep the materialised pool on (see :meth:`to`).
    """

    def __init__(
        self,
        generator: BaseDataGenerator,
        pool_size: int,
        *,
        resample_seed: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        self.generator: BaseDataGenerator | None = generator
        self.pool_size = max(int(pool_size), 1)
        self.observed_dim = generator.observed_dim
        self.horizon = generator.horizon
        self._pool: Batch | None = None
        self._device = torch.device(device) if device is not None else None
        self._rng = torch.Generator().manual_seed(resample_seed)

    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        *,
        resample_seed: int = 0,
        device: torch.device | str | None = None,
    ) -> PooledGenerator:
        """Wrap an already-materialised pool (e.g. loaded from disk)."""
        instance = cls.__new__(cls)
        instance.generator = None
        instance.pool_size = int(batch.x.shape[0])
        instance.observed_dim = int(batch.x.shape[-1])
        instance.horizon = int(batch.x.shape[1])
        instance._pool = batch
        instance._device = torch.device(device) if device is not None else None
        instance._rng = torch.Generator().manual_seed(resample_seed)
        instance._move_pool()
        return instance

    def to(self, device: torch.device | str) -> PooledGenerator:
        """Keep the pool resident on ``device`` (idempotent).

        The pool can be kept on a device (e.g. CUDA) so minibatching only indexes
        device memory and no per-step host-to-device copy is needed. The index
        draw still uses a CPU ``torch.Generator``, so the sampled rows — and
        therefore the numerics — are unchanged.
        """
        self._device = torch.device(device)
        self._move_pool()
        return self

    def _move_pool(self) -> None:
        if self._device is None or self._pool is None:
            return
        from gcmts.core.trainer import move_batch

        self._pool = move_batch(self._pool, self._device)

    def _ensure_pool(self) -> Batch:
        if self._pool is None:
            assert self.generator is not None
            self._pool = cast(Batch, self.generator.sample(self.pool_size))
            self._move_pool()
        return self._pool

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        pool = self._ensure_pool()
        indices = torch.randint(0, pool.x.shape[0], (n_samples,), generator=self._rng)
        return _index_batch(pool, indices)
