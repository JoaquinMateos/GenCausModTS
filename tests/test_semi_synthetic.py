"""Tests for the semi-synthetic effect benchmarks (tumour, IHDP, ACIC)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gcmts.data.semi_synthetic import (
    ACICLoader,
    IHDPGenerator,
    IHDPLoader,
    TumourGrowthSimulator,
)

IHDP_TRAIN = Path("data/semi_synthetic/ihdp_npci_1-100.train.npz")
IHDP_TEST = Path("data/semi_synthetic/ihdp_npci_1-100.test.npz")


def test_tumour_simulator_shapes_and_counterfactual() -> None:
    generator = TumourGrowthSimulator(horizon=8, seed=0)
    batch = generator.sample(16)
    assert batch.x.shape == (16, 8, 3)
    assert batch.context is not None
    assert batch.context["A"].shape == (16, 8, 2)
    assert batch.counterfactual is not None
    assert batch.counterfactual.shape == batch.x.shape
    assert batch.outcome is not None and batch.outcome.shape == (16,)
    assert torch.isfinite(batch.x).all()
    # Untreated counterfactual tumours cannot be smaller on average.
    assert batch.counterfactual[:, -1, 0].mean() >= batch.x[:, -1, 0].mean()


def test_tumour_simulator_shared_noise() -> None:
    generator = TumourGrowthSimulator(horizon=6, seed=1)
    first = generator.sample(4)
    second = generator.sample(4)
    # Independent draws must differ (shared noise only within a batch).
    assert not torch.allclose(first.x, second.x)


def test_tumour_simulator_explicit_action_overrides_policy() -> None:
    generator = TumourGrowthSimulator(horizon=5, seed=2)
    action = torch.ones(3, 5, 2)
    batch = generator.sample(3, factual_action=action)
    assert torch.allclose(batch.context["A"], action)
    assert batch.counterfactual is not None
    assert batch.counterfactual[:, -1, 0].mean() > batch.x[:, -1, 0].mean()


@pytest.mark.skipif(
    not (IHDP_TRAIN.exists() and IHDP_TEST.exists()),
    reason="IHDP archives not downloaded",
)
def test_ihdp_loader_replications() -> None:
    loader = IHDPLoader(IHDP_TRAIN, IHDP_TEST)
    assert loader.n_replications == 100
    assert loader.n_covariates == 25
    for split in ("train", "test"):
        replication = loader.replication(3, split=split)
        assert replication.x.shape[0] == replication.t.shape[0]
        assert set(np.unique(replication.t)).issubset({0, 1})
        tau = replication.mu1 - replication.mu0
        assert loader.pehe(3, tau, split=split) == pytest.approx(0.0)
        assert loader.ate_error(3, tau, split=split) == pytest.approx(0.0, abs=1e-5)


def test_acic_loader_missing_directory() -> None:
    with pytest.raises(FileNotFoundError):
        ACICLoader("data/acic/does-not-exist")


@pytest.mark.skipif(
    not (IHDP_TRAIN.exists() and IHDP_TEST.exists()),
    reason="IHDP archives not downloaded",
)
def test_ihdp_generator_batches_and_rescaling() -> None:
    loader = IHDPLoader(IHDP_TRAIN, IHDP_TEST)
    replication = loader.replication(0, split="test")
    generator = IHDPGenerator(replication, seed=0)
    batch = generator.sample(32)
    assert batch.x.shape == (32, 1, 26)
    assert batch.context is not None
    assert batch.context["A"].shape == (32, 1, 1)
    assert batch.counterfactual is not None
    assert batch.outcome is not None and batch.outcome.shape == (32,)
    full = generator.sample_all()
    assert full.x.shape[0] == replication.x.shape[0]
    # Counterfactual and factual outcomes differ only in the last coordinate.
    factual = full.x[:, 0, :-1]
    counterfactual = full.counterfactual[:, 0, :-1]
    assert torch.allclose(factual, counterfactual)
    # Rescaling recovers the original outcome units.
    raw = torch.from_numpy(replication.yf).float()
    assert torch.allclose(generator.rescale_outcome(full.x[:, 0, -1]), raw, atol=1e-4)
    assert generator.outcome_dim == 25
