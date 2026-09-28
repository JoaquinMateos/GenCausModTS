"""Aggregation and persistence helpers for benchmark result rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["aggregate_rows", "save_rows"]

_NON_METRIC = {"epochs", "n_parameters", "seconds"}


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
