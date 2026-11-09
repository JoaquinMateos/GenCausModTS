"""Loaders for the CITRIS/iCITRIS video datasets.

The arrays are the official Zenodo releases used by Lippe et al. (2022, 2023).
The repository only stores/download scripts here: the generated ``.npz`` files
are not committed.  Each loader returns the benchmark's ordinary ``Batch``:
images are flattened into ``(B, T, C*H*W)`` so the existing image feature
encoder can be reused, while the unflattened layout is declared through
``image_shape`` on the suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = [
    "InterventionalPongGenerator",
    "CausalPinballGenerator",
    "BallInBoxesGenerator",
    "TemporalCausal3DIdentGenerator",
]


class _CitrisNPZGenerator(BaseDataGenerator):
    """Common contiguous-window sampler for the official CITRIS ``.npz`` files."""

    def __init__(
        self,
        *,
        train_path: str | Path,
        image_shape: tuple[int, int, int],
        latent_dim: int,
        n_targets: int,
        horizon: int,
        seed: int | None = None,
        image_key: str = "images",
        latent_key: str = "latents",
        target_key: str = "targets",
        extra_latent_key: str | None = None,
        extra_target_key: str | None = None,
        target_groups: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        path = Path(train_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"CITRIS dataset not found at {path}. Run "
                "scripts/download_citris_datasets.py first."
            )
        self.train_path = path
        self.image_shape = tuple(image_shape)
        self.n_targets = int(n_targets)
        self.target_groups = target_groups
        self._seed = int(seed or 0)
        data = np.load(path, mmap_mode="r")
        self.images = data[image_key]
        self.latents = data[latent_key]
        self.targets = data[target_key] if target_key in data.files else None
        self._extra_latents = data[extra_latent_key] if extra_latent_key in data.files else None
        self._extra_targets = data[extra_target_key] if extra_target_key in data.files else None
        if self.latents.shape[-1] != latent_dim:
            raise ValueError(f"{path} has {self.latents.shape[-1]} latents, expected {latent_dim}")
        super().__init__(observed_dim=int(np.prod(image_shape)), horizon=int(horizon))
        self._rng = np.random.default_rng(self._seed)

    def _window_indices(self, n_samples: int, horizon: int) -> np.ndarray:
        max_start = int(self.images.shape[0]) - int(horizon)
        if max_start <= 0:
            raise ValueError("The dataset is shorter than the requested horizon.")
        starts = self._rng.integers(0, max_start, size=int(n_samples))
        return starts[:, None] + np.arange(int(horizon))[None, :]

    @staticmethod
    def _to_tensor(images: np.ndarray) -> Tensor:
        # Match the official dataset preparation: RGB frames in [-1, 1].
        array = torch.from_numpy(np.asarray(images)).float() / 255.0
        if array.shape[-1] == 4:
            array = array[..., :3]
        array = array * 2.0 - 1.0
        return array.permute(0, 1, 4, 2, 3).contiguous()

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        del kwargs
        horizon = int(horizon or self.horizon)
        idx = self._window_indices(n_samples, horizon)
        images = self._to_tensor(self.images[idx])
        latents = torch.from_numpy(np.asarray(self.latents[idx])).float()
        if self._extra_latents is not None:
            extra = torch.from_numpy(np.asarray(self._extra_latents[idx])).float()
            latents = torch.cat([latents, extra], dim=-1)
        if self.targets is not None:
            target_idx = idx[:, 1:] - 1  # target[t] describes t -> t+1
            targets = torch.from_numpy(np.asarray(self.targets[target_idx])).float()
            interventions = torch.zeros(n_samples, horizon, self.n_targets)
            interventions[:, 1:] = targets
        else:
            interventions = torch.zeros(n_samples, horizon, self.n_targets)
        if self._extra_targets is not None:
            extra = torch.from_numpy(np.asarray(self._extra_targets[idx[:, 1:] - 1])).float()
            if self.target_groups is not None:
                extra = torch.stack(
                    [extra[..., list(group)].amax(dim=-1) for group in self.target_groups],
                    dim=-1,
                )
            if extra.shape[-1] == self.n_targets:
                interventions[:, 1:] = extra
        return Batch(
            x=images.reshape(n_samples, horizon, -1),
            z=latents,
            context={"I": interventions},
            adjacency=None,
            instantaneous_adjacency=None,
        )


class InterventionalPongGenerator(_CitrisNPZGenerator):
    def __init__(self, *, data_root: str | Path = "data/citris", horizon: int = 8, seed: int | None = None) -> None:
        root = Path(data_root) / "pong" / "complexpong_time_dep_015_32width_singleint"
        super().__init__(
            train_path=root / "train.npz",
            image_shape=(3, 32, 32),
            latent_dim=8,
            n_targets=8,
            horizon=horizon,
            seed=seed,
        )


class CausalPinballGenerator(_CitrisNPZGenerator):
    def __init__(self, *, data_root: str | Path = "data/citris", horizon: int = 8, seed: int | None = None) -> None:
        root = Path(data_root) / "pinball" / "pinball"
        super().__init__(
            train_path=root / "train.npz",
            image_shape=(3, 64, 64),
            latent_dim=12,
            n_targets=5,
            horizon=horizon,
            seed=seed,
        )


class BallInBoxesGenerator(_CitrisNPZGenerator):
    def __init__(self, *, data_root: str | Path = "data/citris", horizon: int = 8, seed: int | None = None) -> None:
        root = Path(data_root) / "ball" / "box_x"
        super().__init__(
            train_path=root / "train.npz",
            image_shape=(3, 32, 32),
            latent_dim=3,
            n_targets=3,
            horizon=horizon,
            seed=seed,
        )


class TemporalCausal3DIdentGenerator(_CitrisNPZGenerator):
    """The official 7-shape Temporal Causal3DIdent release.

    ``latents`` contains the ten continuous factors and ``shape_latents`` the
    two categorical shape factors; ``interventions`` has twelve target columns.
    """

    def __init__(self, *, data_root: str | Path = "data/citris", horizon: int = 8, seed: int | None = None) -> None:
        root = Path(data_root) / "temporal_causal3d" / "causal3d_time_dep_all7_conthue_01_coarse"
        super().__init__(
            train_path=root / "train.npz",
            image_shape=(3, 64, 64),
            latent_dim=10,
            n_targets=7,
            horizon=horizon,
            seed=seed,
            image_key="imgs",
            latent_key="raw_latents",
            extra_latent_key="shape_latents",
            extra_target_key="interventions",
            # The official coarse Causal3D protocol groups pos-x/y/z and
            # rot-alpha/beta/gamma/spot; the remaining five factors stay
            # separate.  This keeps the target classifier's 2^K state space
            # tractable while retaining the 12-dimensional latent state.
            target_groups=((0, 1, 2), (3, 4, 5, 6), (7,), (8,), (9,), (10,), (11,)),
        )
