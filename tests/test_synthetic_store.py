"""Tests for persistent synthetic datasets and the derived pool generator."""

from __future__ import annotations

from pathlib import Path

import torch

from gcmts.data.pool import PooledGenerator
from gcmts.data.synthetic import TemporalNonlinearGenerator
from gcmts.data.synthetic_store import (
    dataset_exists,
    dataset_path,
    load_dataset,
    read_manifest,
    save_dataset,
)


def _pools() -> tuple[object, object]:
    generator = TemporalNonlinearGenerator(
        observed_dim=6, latent_dim=3, horizon=16, n_regimes=2, seed=0
    )
    return generator.sample(64), generator.sample(16)


def test_save_load_roundtrip_and_manifest(tmp_path: Path) -> None:
    train, eval_batch = _pools()
    path = dataset_path(tmp_path, "npvar", 3)
    record = save_dataset(
        path, train, eval_batch,
        suite="npvar", seed=3, max_lag=1, generator="TemporalNonlinearGenerator", latent_dim=3,
    )
    assert dataset_exists(tmp_path, "npvar", 3)
    assert len(record.sha256) == 64
    assert (record.train_size, record.eval_size) == (64, 16)

    loaded_train, loaded_eval, meta = load_dataset(path)
    assert torch.equal(loaded_train.x, train.x)
    assert torch.equal(loaded_train.z, train.z)
    assert torch.equal(loaded_eval.x, eval_batch.x)
    assert meta["suite"] == "npvar" and meta["max_lag"] == 1 and meta["latent_dim"] == 3

    manifest = read_manifest(tmp_path)
    assert "npvar/seed_03" in manifest["datasets"]
    assert manifest["datasets"]["npvar/seed_03"]["train_size"] == 64


def test_manifest_accumulates_datasets(tmp_path: Path) -> None:
    train, eval_batch = _pools()
    for seed in (0, 1):
        save_dataset(
            dataset_path(tmp_path, "npvar", seed), train, eval_batch,
            suite="npvar", seed=seed, max_lag=1, generator="TemporalNonlinearGenerator",
        )
    manifest = read_manifest(tmp_path)
    assert {"npvar/seed_00", "npvar/seed_01"} <= set(manifest["datasets"])


def test_from_batch_pool_resamples_deterministically(tmp_path: Path) -> None:
    train, _ = _pools()
    pool_a = PooledGenerator.from_batch(train, resample_seed=0)
    pool_b = PooledGenerator.from_batch(train, resample_seed=0)
    sample_a = pool_a.sample(8)
    sample_b = pool_b.sample(8)
    assert sample_a.x.shape == (8, 16, 6)
    assert torch.equal(sample_a.x, sample_b.x)
    # Indices come from the pool, so every drawn row is one of its rows.
    pool_rows = set(map(tuple, train.x.reshape(train.x.shape[0], -1).tolist()))
    drawn_rows = set(map(tuple, sample_a.x.reshape(8, -1).tolist()))
    assert drawn_rows <= pool_rows


def test_pooled_generator_exposes_to() -> None:
    # Regression guard: run_crl_shared._run calls ``source.to(device)`` when the
    # training source is a pool, so the method must exist and be idempotent.
    generator = TemporalNonlinearGenerator(
        observed_dim=6, latent_dim=3, horizon=16, n_regimes=2, seed=0
    )
    pool = PooledGenerator.from_batch(generator.sample(32), resample_seed=0)
    assert hasattr(pool, "to")
    assert pool.to("cpu") is pool
    assert pool.to("cpu") is pool
    assert pool.sample(4).x.shape == (4, 16, 6)


def test_read_manifest_missing(tmp_path: Path) -> None:
    manifest = read_manifest(tmp_path / "does_not_exist")
    assert manifest["datasets"] == {}
