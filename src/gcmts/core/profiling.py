"""Lightweight phase profiler for the training loop.

When enabled, :class:`~gcmts.core.trainer.SimpleTrainer` records how much wall
time goes into each phase of an optimisation step (data generation, host-to-
device copy, forward, loss, backward, optimiser, loss-sync). This makes it
obvious whether a suite is CPU-bound (data generation dominates) or GPU-bound
(forward/backward dominate) before optimising.

The profiler synchronises the device at every phase boundary, so enable it only
for short diagnostic runs (see ``scripts/profile_training.py``), not for the
paper runs.
"""

from __future__ import annotations

import contextlib
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch

__all__ = ["PhaseProfiler"]

logger = logging.getLogger("gcmts.profiling")


class PhaseProfiler:
    """Accumulate per-phase timings across optimisation steps."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        device: torch.device | None = None,
        warmup_steps: int = 3,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.warmup_steps = max(warmup_steps, 0)
        self._sync = bool(device is not None and getattr(device, "type", "") == "cuda")
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._steps = 0

    def _barrier(self) -> None:
        if self._sync:
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._barrier()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._barrier()
            if self._steps >= self.warmup_steps:
                self._durations[name].append(time.perf_counter() - start)

    def end_step(self) -> None:
        self._steps += 1

    def summary(self) -> dict[str, dict[str, float]]:
        """``phase -> {mean, median, total, share}`` (seconds, share of total)."""
        totals = {name: sum(values) for name, values in self._durations.items()}
        grand = sum(totals.values())
        out: dict[str, dict[str, float]] = {}
        for name, values in self._durations.items():
            out[name] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "total": totals[name],
                "share": totals[name] / grand if grand else 0.0,
            }
        return out

    def report(self, *, title: str = "training step profile") -> None:
        if not self.enabled:
            return
        summary = self.summary()
        if not summary:
            logger.info(
                "%s: no samples (run longer than %d warm-up steps)",
                title,
                self.warmup_steps,
            )
            return
        steps = self._steps
        logger.info("=== %s (%d steps) ===", title, steps)
        logger.info("%-14s %10s %10s %7s", "phase", "mean ms", "median ms", "share")
        for name, stats in sorted(summary.items(), key=lambda kv: kv[1]["total"], reverse=True):
            logger.info(
                "%-14s %10.3f %10.3f %6.1f%%",
                name,
                stats["mean"] * 1e3,
                stats["median"] * 1e3,
                stats["share"] * 100,
            )
        total = sum(stats["mean"] for stats in summary.values())
        logger.info("%-14s %10.3f", "TOTAL", total * 1e3)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot (used by the profiling script)."""
        return {
            "steps": self._steps,
            "phases": {
                name: {k: float(v) for k, v in stats.items()}
                for name, stats in self.summary().items()
            },
        }
