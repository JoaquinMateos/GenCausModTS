from __future__ import annotations

from pathlib import Path

import numpy as np

from gcmts.data.citris import (
    BallInBoxesGenerator,
    CausalPinballGenerator,
    InterventionalPongGenerator,
    TemporalCausal3DIdentGenerator,
)


def _save(path: Path, *, n: int, c: int, size: int, latent_dim: int, target_dim: int, image_key: str = "images") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arrays = {
        image_key: rng.integers(0, 256, size=(n, size, size, c), dtype=np.uint8),
        "latents": rng.normal(size=(n, latent_dim)).astype(np.float32),
        "targets": (rng.random((n - 1, target_dim)) > 0.8).astype(np.float32),
    }
    if image_key == "imgs":
        arrays["shape_latents"] = rng.integers(0, 7, size=(n, 2), dtype=np.uint8)
        arrays["raw_latents"] = arrays["latents"].copy()
        arrays["interventions"] = arrays.pop("targets")
    np.savez(path, **arrays)


def _check(generator, latent_dim: int, target_dim: int) -> None:
    batch = generator.sample(3, horizon=4)
    assert batch.x.shape == (3, 4, generator.observed_dim)
    assert batch.z is not None and batch.z.shape == (3, 4, latent_dim)
    assert batch.context is not None and batch.context["I"].shape == (3, 4, target_dim)
    assert float(batch.x.min()) >= -1.0 and float(batch.x.max()) <= 1.0


def test_official_citris_loaders(tmp_path: Path) -> None:
    _save(tmp_path / "pong" / "complexpong_time_dep_015_32width_singleint" / "train.npz", n=20, c=4, size=32, latent_dim=8, target_dim=8)
    _save(tmp_path / "pinball" / "pinball" / "train.npz", n=20, c=4, size=64, latent_dim=12, target_dim=5)
    _save(tmp_path / "ball" / "box_x" / "train.npz", n=20, c=3, size=32, latent_dim=3, target_dim=3)
    _save(tmp_path / "temporal_causal3d" / "causal3d_time_dep_all7_conthue_01_coarse" / "train.npz", n=20, c=4, size=64, latent_dim=10, target_dim=12, image_key="imgs")

    _check(InterventionalPongGenerator(data_root=tmp_path, horizon=4, seed=0), 8, 8)
    _check(CausalPinballGenerator(data_root=tmp_path, horizon=4, seed=0), 12, 5)
    _check(BallInBoxesGenerator(data_root=tmp_path, horizon=4, seed=0), 3, 3)
    _check(TemporalCausal3DIdentGenerator(data_root=tmp_path, horizon=4, seed=0), 12, 7)
