"""Persistent synthetic datasets: a fixed (train, eval) pool plus an audit manifest.

The review's protocol (Section 9 and ``docs/experiment_plan.md``) fixes the
*synthetic process* per seed — the latent graph, mechanisms and mixing map are
identical across methods, with a fixed seed schedule. The runners normally draw
fresh trajectories on the fly, which is CPU-bound. This module materialises that
fixed process once as a ``(train, eval)`` pool per ``(suite, seed)`` so it can be
generated on a CPU box and consumed on a GPU box without regenerating per step.

Rigor:
* Pools are drawn from the *same* seeded generator a runner would build
  (``torch.manual_seed(seed)`` before ``suite.make_generator(seed)``), so the
  latent graph, mechanisms and mixing map are exactly the protocol's and match
  across methods.
* Training resamples minibatches from the train pool; evaluation uses a
  **disjoint** held-out pool drawn from the same generator afterwards, so there
  is no train/eval leakage.
* A pool is finite: over one run each trajectory is seen about
  ``steps * batch_size / train_size`` times. Size the pool accordingly and
  report the protocol; the file stores the sizes so this is auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from gcmts.typing import Batch

__all__ = [
    "MANIFEST_NAME",
    "DatasetRecord",
    "batch_from_state",
    "batch_to_state",
    "dataset_exists",
    "dataset_path",
    "load_dataset",
    "read_manifest",
    "save_dataset",
]

MANIFEST_NAME = "manifest.json"


def batch_to_state(batch: Batch) -> dict[str, Any]:
    """Serialise a :class:`Batch` to plain containers (safe across versions)."""
    return {
        "x": batch.x,
        "z": batch.z,
        "context": dict(batch.context) if batch.context is not None else None,
        "mask": batch.mask,
        "adjacency": batch.adjacency,
        "instantaneous_adjacency": batch.instantaneous_adjacency,
        "counterfactual": batch.counterfactual,
        "outcome": batch.outcome,
        "domain": batch.domain,
    }


def batch_from_state(state: dict[str, Any]) -> Batch:
    """Rebuild a :class:`Batch` from :func:`batch_to_state` output."""
    context = state.get("context")
    return Batch(
        x=state["x"],
        z=state.get("z"),
        context=dict(context) if context is not None else None,
        mask=state.get("mask"),
        adjacency=state.get("adjacency"),
        instantaneous_adjacency=state.get("instantaneous_adjacency"),
        counterfactual=state.get("counterfactual"),
        outcome=state.get("outcome"),
        domain=state.get("domain"),
    )


@dataclass
class DatasetRecord:
    """Audit record for one persisted ``(suite, seed)`` dataset."""

    suite: str
    seed: int
    train_size: int
    eval_size: int
    horizon: int
    observed_dim: int
    latent_dim: int | None
    max_lag: int
    generator: str
    created_at: str
    sha256: str
    path: str


def dataset_path(root: str | Path, suite: str, seed: int) -> Path:
    return Path(root) / suite / f"seed_{seed:02d}.pt"


def dataset_exists(root: str | Path, suite: str, seed: int) -> bool:
    return dataset_path(root, suite, seed).exists()


def save_dataset(
    path: str | Path,
    train: Batch,
    eval_batch: Batch,
    *,
    suite: str,
    seed: int,
    max_lag: int,
    generator: str,
    latent_dim: int | None = None,
) -> DatasetRecord:
    """Write ``train``/``eval`` pools to ``path`` and update the manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": batch_to_state(train),
        "eval": batch_to_state(eval_batch),
        "meta": {
            "suite": suite,
            "seed": seed,
            "train_size": int(train.x.shape[0]),
            "eval_size": int(eval_batch.x.shape[0]),
            "horizon": int(train.x.shape[1]),
            "observed_dim": int(train.x.shape[-1]),
            "latent_dim": None if latent_dim is None else int(latent_dim),
            "max_lag": int(max_lag),
            "generator": generator,
            "created_at": datetime.now(UTC).isoformat(),
            "torch_version": torch.__version__,
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = DatasetRecord(
        suite=suite,
        seed=seed,
        train_size=int(train.x.shape[0]),
        eval_size=int(eval_batch.x.shape[0]),
        horizon=int(train.x.shape[1]),
        observed_dim=int(train.x.shape[-1]),
        latent_dim=None if latent_dim is None else int(latent_dim),
        max_lag=int(max_lag),
        generator=generator,
        created_at=payload["meta"]["created_at"],
        sha256=digest,
        path=str(path),
    )
    _update_manifest(path.parents[1], record)
    return record


def load_dataset(path: str | Path) -> tuple[Batch, Batch, dict[str, Any]]:
    """Load a persisted dataset, returning ``(train, eval, meta)``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return (
        batch_from_state(payload["train"]),
        batch_from_state(payload["eval"]),
        dict(payload["meta"]),
    )


def _update_manifest(root: Path, record: DatasetRecord) -> None:
    manifest = read_manifest(root)
    manifest["datasets"][f"{record.suite}/seed_{record.seed:02d}"] = asdict(record)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    path = root / MANIFEST_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        return {"datasets": {}, "updated_at": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"datasets": {}, "updated_at": None}
    data.setdefault("datasets", {})
    return data
