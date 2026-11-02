"""Cheap, verifiable manifests for dataset directories.

ACIC replication ``i`` is the ``i``-th directory in sorted-hash order, so a
different download silently changes the index -> replication map. A manifest
records the content hash of every recorded file and is checked on load.

The digest covers ``(relative path, size, sha256)`` for a fixed file list. The
verification is cached per directory within a process, because the ACIC loader is
constructed once per replication.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

__all__ = ["MANIFEST_NAME", "build_manifest", "verify_manifest", "write_manifest"]

MANIFEST_NAME = "manifest.json"
_VERIFIED: dict[Path, str] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entries(root: Path, names: Iterable[str]) -> list[dict[str, object]]:
    entries = []
    for name in sorted(names):
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Manifest file missing: {path}")
        entries.append(
            {"path": name, "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    return entries


def _digest(entries: list[dict[str, object]]) -> str:
    payload = "\n".join(
        f"{entry['path']}:{entry['size']}:{entry['sha256']}" for entry in entries
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(root: str | Path, names: Iterable[str]) -> dict[str, object]:
    root = Path(root)
    entries = _entries(root, names)
    return {
        "root": root.name,
        "n_files": len(entries),
        "digest": _digest(entries),
        "files": entries,
    }


def write_manifest(root: str | Path, names: Iterable[str]) -> Path:
    root = Path(root)
    payload = build_manifest(root, names)
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def verify_manifest(root: str | Path, names: Iterable[str]) -> None:
    """Raise when the directory content differs from its stored manifest."""
    root = Path(root)
    path = root / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset manifest at {path}. Run `scripts/write_acic_manifest.py` "
            "on the snapshot you intend to use and ship it with the data."
        )
    expected = json.loads(path.read_text(encoding="utf-8"))
    entries = _entries(root, names)
    actual = _digest(entries)
    if actual != expected.get("digest"):
        raise ValueError(
            f"Dataset manifest mismatch for {root}: expected digest "
            f"{expected.get('digest')}, found {actual}. The download differs from "
            "the pinned snapshot; the replication index map is not comparable."
        )
    _VERIFIED[root.resolve()] = actual


def verified(root: str | Path) -> bool:
    return Path(root).resolve() in _VERIFIED
