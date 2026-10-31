"""Reference floors for the static CATE panels.

For every replication, two predictors that use no heterogeneity are evaluated:

* *zero effect*:  tau_hat = 0        -> sqrt-PEHE = RMS(tau_true)
* *constant ATE*: tau_hat = mean(tau) -> sqrt-PEHE = SD(tau_true)

A learned method that does not beat the constant-ATE floor has learned nothing
about effect heterogeneity, whatever its absolute PEHE. The script reports the
floor means and, per method, the share of replications in which the method beats
the constant-ATE floor.

    uv run python scripts/reference_floors.py --outputs outputs --reps 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ctee_static import _load_pair  # noqa: E402

PANELS = {
    "ihdp": (Path("data/semi_synthetic"), 100),
    "acic": (Path("data/acic"), 100),
    "twins": (Path("data/static"), 100),
    "news": (Path("data/static"), 50),
}


def floors(panel: str, directory: Path, reps: int) -> np.ndarray:
    values = []
    for index in range(reps):
        try:
            _, test = _load_pair(panel, index, directory)
        except Exception:  # noqa: BLE001 - a missing replication is skipped
            values.append((math.nan, math.nan))
            continue
        tau = np.asarray(test.mu1 - test.mu0, dtype=np.float64)
        if not np.isfinite(tau).all():
            values.append((math.nan, math.nan))
            continue
        values.append((float(np.sqrt(np.mean(tau**2))), float(np.std(tau))))
    return np.asarray(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--reps", type=int, default=None)
    args = parser.parse_args()

    lines = [
        "| Panel | Replications | Zero-effect floor | Constant-ATE floor | Method "
        "| Mean PEHE | Median PEHE | Beats constant ATE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for panel, (directory, default_reps) in PANELS.items():
        reps = args.reps or default_reps
        table = floors(panel, directory, reps)
        valid = table[np.isfinite(table).all(axis=1)]
        if not len(valid):
            lines.append(f"| {panel} | 0 | - | - | - | - | - | - |")
            continue
        path = args.outputs / f"ctee_static_{panel}.json"
        if not path.exists():
            lines.append(f"| {panel} | {len(valid)} | {valid[:, 0].mean():.3f} | "
                         f"{valid[:, 1].mean():.3f} | (no results) | - | - | - |")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        by_method: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_method[str(row.get("method"))].append(row)
        for method, method_rows in sorted(by_method.items()):
            pehe = np.asarray([row.get("pehe") for row in method_rows], dtype=float)
            pehe = pehe[np.isfinite(pehe)]
            if not len(pehe):
                continue
            n = min(len(pehe), len(valid))
            beats = float(np.mean(pehe[:n] < valid[:n, 1]))
            lines.append(
                f"| {panel} | {n} | {valid[:n, 0].mean():.3f} | "
                f"{valid[:n, 1].mean():.3f} | {method} | {pehe.mean():.3f} | "
                f"{np.median(pehe):.3f} | {100 * beats:.0f}% |"
            )
    text = "\n".join(lines)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
