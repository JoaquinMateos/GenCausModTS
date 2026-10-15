"""Tests for the trainer data path: prefetch, light batch, profiling, pooling."""

from __future__ import annotations

import torch

from gcmts.causal_representation_learning.methods import IVAE
from gcmts.core import SimpleTrainer
from gcmts.core.profiling import PhaseProfiler
from gcmts.data.pool import PooledGenerator
from gcmts.data.synthetic import TemporalNonlinearGenerator


def _train(**trainer_kwargs: object) -> dict[str, list[float]]:
    torch.manual_seed(0)
    generator = TemporalNonlinearGenerator(
        observed_dim=6, latent_dim=3, horizon=16, n_regimes=4, seed=0
    )
    model = IVAE(observed_dim=6, latent_dim=3, u_dim=4, obs_noise=0.1)
    trainer = SimpleTrainer(
        max_epochs=2,
        batch_size=64,
        steps_per_epoch=4,
        log_every=0,
        device="cpu",
        seed=0,
        val_size=0,
        **trainer_kwargs,
    )
    return trainer.fit(model, generator)


def _equal(a: dict[str, list[float]], b: dict[str, list[float]]) -> bool:
    if set(a) != set(b):
        return False
    return all(
        len(a[key]) == len(b[key])
        and all(x == y or (x != x and y != y) for x, y in zip(a[key], b[key], strict=True))
        for key in a
    )


def test_prefetch_and_light_batch_are_result_neutral() -> None:
    base = _train()
    assert _equal(base, _train(prefetch=True))
    assert _equal(base, _train(light_batch=True))
    assert _equal(base, _train(prefetch=True, light_batch=True))


def test_phase_profiler_records_phases() -> None:
    profiler = PhaseProfiler(enabled=True, device=torch.device("cpu"), warmup_steps=1)
    with profiler.phase("forward"):
        torch.randn(8, 8).sum()
    profiler.end_step()
    with profiler.phase("forward"):
        torch.randn(8, 8).sum()
    profiler.end_step()
    summary = profiler.summary()
    assert "forward" in summary
    assert summary["forward"]["median"] >= 0.0
    assert 0.0 <= summary["forward"]["share"] <= 1.0
    assert profiler.as_dict()["steps"] == 2


def test_pooled_generator_resamples_a_fixed_pool() -> None:
    base = TemporalNonlinearGenerator(
        observed_dim=6, latent_dim=3, horizon=16, n_regimes=2, seed=1
    )
    pooled = PooledGenerator(base, pool_size=16, resample_seed=0)
    first = pooled.sample(8)
    assert first.x.shape == (8, 16, 6)
    assert first.z is not None and first.z.shape == (8, 16, 3)
    # The pool is built once and then reused.
    pool = pooled._ensure_pool()
    assert pool.x.shape[0] == 16
    again = pooled.sample(8)
    assert again.x.shape == (8, 16, 6)
