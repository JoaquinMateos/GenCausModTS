"""Aggregation and persistence helpers for benchmark result rows."""

from __future__ import annotations

import csv
import json
import os
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "aggregate_rows",
    "code_revision",
    "guard_resume",
    "load_rows",
    "paired_wilcoxon",
    "save_rows",
    "wall_seconds",
]

_NON_METRIC = {"epochs", "n_parameters", "seconds", "seed"}


def code_revision() -> str:
    """Best-effort identifier of the code revision (git commit, else version)."""
    import subprocess

    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import gcmts

        return str(getattr(gcmts, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def load_rows(out_dir: str | Path, stem: str) -> list[dict[str, Any]]:
    """Load previously written raw rows for ``stem`` (``[]`` if none)."""
    path = Path(out_dir) / f"{stem}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _fingerprint(config: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(
        {"code": code_revision(), **config}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def guard_resume(
    out_dir: str | Path,
    stem: str,
    config: dict[str, Any],
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """Check that existing rows were produced by the same configuration.

    Writes ``<stem>_config.json`` (fingerprint + configuration + code revision)
    the first time, and on later runs compares against it. Returns
    ``(ok, reason)``; ``ok=False`` means the caller must refuse to append (the
    training configuration or code revision changed) unless ``force`` is set.
    """
    path = Path(out_dir) / f"{stem}_config.json"
    fingerprint = _fingerprint(config)
    current = {"fingerprint": fingerprint, "code_revision": code_revision(), "config": config}
    if os.environ.get("GCMTS_FORCE", "0").strip().lower() not in {"", "0", "false", "no"}:
        force = True
    rows_path = Path(out_dir) / f"{stem}.json"
    if not path.exists():
        if rows_path.exists() and not force:
            return False, (
                "existing rows were written before the resume guard, so their "
                "configuration and code revision are unknown"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")
        return True, ""
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True, ""
    if stored.get("fingerprint") != fingerprint and not force:
        return False, (
            "existing rows were produced by a different configuration or code "
            f"revision (stored code={stored.get('code_revision')}, "
            f"current code={current['code_revision']})"
        )
    return True, ""


def wall_seconds(start: float) -> float:
    """Wall-clock seconds since ``start`` (a ``time.perf_counter()`` reading).

    Returns ``NaN`` when ``GCMTS_TIMING=0``. Use that to suppress the per-run
    ``seconds`` field when many workers run concurrently on one machine: their
    wall-clock times are then inflated by CPU contention and are no longer
    comparable across methods, or to a sequential run on an idle machine.
    """
    if os.environ.get("GCMTS_TIMING", "1") == "0":
        return float("nan")
    return _time.perf_counter() - start


def paired_wilcoxon(
    rows: list[dict[str, Any]],
    case_a: str,
    case_b: str,
    metric: str,
) -> float:
    """Paired Wilcoxon signed-rank p-value for ``case_a`` vs ``case_b`` on ``metric``.

    Rows are matched by their per-seed order of appearance, so both cases must
    have been evaluated under the same seed schedule.
    """
    from scipy.stats import wilcoxon

    values_a = [float(r[metric]) for r in rows if r["case"] == case_a and r.get(metric) is not None]
    values_b = [float(r[metric]) for r in rows if r["case"] == case_b and r.get(metric) is not None]
    n = min(len(values_a), len(values_b))
    if n < 2:
        return float("nan")
    if all(a == b for a, b in zip(values_a[:n], values_b[:n], strict=True)):
        return 1.0
    result = wilcoxon(values_a[:n], values_b[:n])
    return float(result.pvalue)


def aggregate_rows(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate per-seed result rows into ``case -> metric -> {mean, std}``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for case, group in grouped.items():
        metrics: dict[str, dict[str, float]] = {}
        keys = [
            key
            for key, value in group[0].items()
            if isinstance(value, (int, float)) and key not in _NON_METRIC
        ]
        for key in keys:
            values = [float(row[key]) for row in group if row.get(key) is not None]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": float(len(values)),
                }
        aggregated[case] = metrics
    return aggregated


def save_rows(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    stem: str,
) -> dict[str, Path]:
    """Write raw rows and their mean/std aggregation as JSON + CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{stem}.json"
    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    aggregated = aggregate_rows(rows)
    agg_path = out / f"{stem}_agg.json"
    agg_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    flat: list[dict[str, Any]] = []
    for case, metrics in aggregated.items():
        row: dict[str, Any] = {"case": case}
        for name, stats in metrics.items():
            row[f"{name}_mean"] = stats["mean"]
            row[f"{name}_std"] = stats["std"]
        flat.append(row)
    csv_path = out / f"{stem}_agg.csv"
    if flat:
        fieldnames = list(flat[0].keys())
        for row in flat:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat)
    return {"raw": raw_path, "agg_json": agg_path, "agg_csv": csv_path}
