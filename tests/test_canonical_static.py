"""Tests for the canonical real-covariate static loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gcmts.data.canonical_static import NewsLoader, TwinsLoader, load_jobs
from gcmts.data.semi_synthetic import ACICLoader

ROOT = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(
    not (ROOT / "static" / "twins.csv.gz").exists(), reason="Twins data not downloaded"
)
def test_twins_loader_has_both_potential_outcomes() -> None:
    loader = TwinsLoader(ROOT / "static" / "twins.csv.gz", replications=2, seed=0)
    replication = loader.load()[0]
    assert replication.x.ndim == 2
    assert replication.t.shape == (replication.x.shape[0],)
    assert replication.mu0.shape == replication.mu1.shape == replication.t.shape
    differences = replication.yf - np.where(replication.t > 0, replication.mu1, replication.mu0)
    assert np.allclose(differences, 0.0)


@pytest.mark.skipif(
    not (ROOT / "static" / "news").exists(), reason="News data not downloaded"
)
def test_news_loader_returns_potential_means() -> None:
    loader = NewsLoader(ROOT / "static" / "news", replications=1, seed=0)
    replication = loader.load()[0]
    assert replication.x.shape[0] == replication.t.shape[0]
    assert replication.mu0.shape == replication.mu1.shape
    assert np.isfinite(replication.mu0).all()


@pytest.mark.skipif(
    not (ROOT / "static" / "jobs_train.npz").exists(),
    reason="Jobs data not downloaded",
)
def test_jobs_loader_exposes_marginal_reference() -> None:
    cohort = load_jobs(ROOT / "static" / "jobs_train.npz")
    assert cohort.kind == "rct"
    assert cohort.x.shape[0] == cohort.t.shape[0] == cohort.y.shape[0]
    assert np.isfinite(cohort.reference_ate)


@pytest.mark.skipif(
    not (ROOT / "acic").exists(), reason="ACIC mirror not downloaded"
)
def test_acic_loader_standardises_outcomes() -> None:
    directory = ROOT / "acic"
    names = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if not names:
        pytest.skip("no ACIC replications")
    item = ACICLoader(directory).replication(names[0])
    assert abs(float(np.mean(item["yf"]))) < 1e-5
    assert 0.5 < float(np.std(item["yf"])) < 2.0
    assert "mu0" in item and "mu1" in item
