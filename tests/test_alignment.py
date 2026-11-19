"""Tests for the audit fixes in the CRL runner (C2, C10, C20)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from gcmts.evaluation.benchmarks import CRL_BENCHMARKS
from gcmts.typing import Batch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    training = False

    def encode(self, x: torch.Tensor, context: dict | None = None) -> torch.Tensor:
        return x


class _FakeGenerator:
    permutation = torch.tensor([1, 0])

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def sample(self, n: int) -> Batch:
        z = torch.randn(n, 3, 2)
        return Batch(x=z[..., self.permutation], z=z)


def test_structure_permutation_uses_a_held_out_draw_not_the_evaluation_batch() -> None:
    module = _load("run_crl_shared")
    used: list[int] = []

    def make_generator(seed: int):
        used.append(seed)
        return _FakeGenerator(seed)

    suite = module.Suite(
        name="fake",
        observed_dim=2,
        latent_dim=2,
        horizon=3,
        n_regimes=1,
        make_generator=make_generator,
    )
    perm, provenance = module._structure_permutation(
        _FakeModel(), suite, 3, "cpu", eval_dim=2
    )
    assert used == [3 + module.ALIGNMENT_SEED_OFFSET]
    assert torch.equal(perm.cpu(), _FakeGenerator.permutation)
    assert "held-out" in provenance


def test_structure_permutation_falls_back_to_identity_without_latents() -> None:
    module = _load("run_crl_shared")

    class NoLatents(_FakeGenerator):
        def sample(self, n: int) -> Batch:
            return Batch(x=torch.randn(n, 3, 2), z=None)

    suite = module.Suite(
        name="fake",
        observed_dim=2,
        latent_dim=2,
        horizon=3,
        n_regimes=1,
        make_generator=lambda seed: NoLatents(seed),
    )
    perm, provenance = module._structure_permutation(
        _FakeModel(), suite, 0, "cpu", eval_dim=2
    )
    assert torch.equal(perm.cpu(), torch.arange(2))
    assert provenance.startswith("identity")


def test_structure_permutation_falls_back_on_dimension_mismatch() -> None:
    module = _load("run_crl_shared")

    class WideLatents(_FakeGenerator):
        def sample(self, n: int) -> Batch:
            return Batch(x=torch.randn(n, 3, 4), z=torch.randn(n, 3, 2))

    suite = module.Suite(
        name="fake",
        observed_dim=4,
        latent_dim=2,
        horizon=3,
        n_regimes=1,
        make_generator=lambda seed: WideLatents(seed),
    )
    perm, provenance = module._structure_permutation(
        _FakeModel(), suite, 0, "cpu", eval_dim=4
    )
    assert perm.shape == (4,)
    assert provenance == "identity(dim-mismatch)"


def test_latent_redundancy_is_a_pearson_correlation_not_a_variance_sum() -> None:
    module = _load("run_crl_shared")
    base = torch.randn(4000, 1)
    z = torch.cat([base, -2.0 * base], dim=-1)
    assert module._latent_redundancy(z) == pytest.approx(1.0, abs=1e-5)
    # Invariant to the per-column scale: a scalar variance sum would change.
    scaled = z * torch.tensor([0.01, 100.0])
    assert module._latent_redundancy(scaled) == pytest.approx(1.0, abs=1e-5)
    independent = torch.randn(4000, 2)
    assert module._latent_redundancy(independent) < 0.1


def test_structurally_undefined_metrics_are_marked_unavailable() -> None:
    # C10: varma/OU expose no ground-truth graph, and temporal suites without
    # interventions expose no one-to-one isolation target.
    assert CRL_BENCHMARKS["varma"].shd_available is False
    assert CRL_BENCHMARKS["ou"].shd_available is False
    assert CRL_BENCHMARKS["npvar"].triplet_available is False
    assert CRL_BENCHMARKS["causal3d"].triplet_available is True
    assert CRL_BENCHMARKS["voronoi"].triplet_available is True


def test_run_grid_records_divergence_instead_of_writing_a_nan_row(tmp_path, monkeypatch) -> None:
    """C22: a diverged grid cell is a failed run, not a silently written row."""
    import argparse

    module = _load("run_crl_shared")
    monkeypatch.setattr(module, "GRID_CONFIGS", module.GRID_CONFIGS[:1])
    monkeypatch.setattr(
        module,
        "build_methods",
        lambda suite, epochs_scale=1.0: [("M", lambda: None, 1)],
    )

    def diverge(*args, **kwargs):  # noqa: ANN002, ANN003
        raise module.TrainingDivergedError("boom")

    monkeypatch.setattr(module, "_run", diverge)
    args = argparse.Namespace(
        grid=True, configs=None, shard_total=0, shard_index=0, stem="smoke_grid",
        seeds=1, epochs_scale=0.0, device="cpu", batch_size=4, out_dir=tmp_path,
        only=None, data_source="fresh", pool_size=0, dataset_dir=tmp_path,
        scheduler="none", early_stopping_patience=None, restore_best_weights=False,
        amp=False, ema_decay=None, kl_warmup_epochs=0,
        resume=False, force=False,
    )
    assert module.run_grid(args) == 1
    rows = module.load_rows(tmp_path, "smoke_grid")
    assert rows == []
