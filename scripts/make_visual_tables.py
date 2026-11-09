#!/usr/bin/env python3
"""Fill the visual-block LaTeX tables from the final benchmark rows.

The table captions and column specifications stay in the checked-in `.tex`
files; only the tabular body is regenerated.  Missing cells are emitted as red
placeholders, so a partially completed run cannot be mistaken for a result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DATASETS = [
    ("causal3d_img", "A.4b: real Temporal Causal3DIdent (seven shapes, 64$\\times$64)", "Causal3DIdent"),
    ("pong", "A.4c: Interventional Pong (32$\\times$32)", "Pong"),
    ("pinball", "A.4d: Causal Pinball (64$\\times$64)", "Pinball"),
    ("ball_in_boxes", "A.4e: Ball-in-Boxes (32$\\times$32)", "Ball-in-Boxes"),
]
METHODS = ["CITRIS", "iCITRIS", "iVAE", "LEAP", "TDRL", "NCTRL"]
METRICS = [
    ("mcc", 3),
    ("r2_diag", 3),
    ("r2_sep", 3),
    ("dci", 3),
    ("shd", 2),
    ("triplet", 3),
    ("forecast1", 3),
    ("forecast4", 3),
]
RED = r"\textcolor{red}{--}"
NA = r"\textcolor{red}{n/a}"


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def _stat(rows: list[dict[str, Any]], metric: str) -> tuple[float, float] | None:
    values = [
        float(row[metric])
        for row in rows
        if row.get(metric) is not None
        and isinstance(row.get(metric), (int, float))
        and math.isfinite(float(row[metric]))
    ]
    if not values:
        return None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _cell(stat: tuple[float, float] | None, digits: int, *, unavailable: bool = False) -> str:
    if stat is None:
        return NA if unavailable else RED
    mean, std = stat
    scale = 10**digits
    return f"{mean:.{digits}f}\\tiny{{$\\pm${std:.{digits}f}}}"


def _replace_body(path: Path, body: list[str]) -> None:
    text = path.read_text()
    start = text.index("\\midrule") + len("\\midrule\n")
    end = text.index("\\bottomrule")
    path.write_text(text[:start] + "\n".join(body) + "\n    " + text[end:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_final_v3"))
    parser.add_argument("--latex-dir", type=Path, default=Path("../latex/pic/experiments"))
    args = parser.parse_args()
    by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for suite, _, _ in DATASETS:
        rows = _load(args.out_dir / f"crl_shared_{suite}.json")
        by_dataset[suite] = {method: [r for r in rows if r.get("case") == method or r.get("method") == method] for method in METHODS}

    # Full per-dataset table.
    body: list[str] = []
    for suite, title, label in DATASETS:
        body.append(f"    \\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\")
        for method in METHODS:
            cells = [_cell(_stat(by_dataset[suite][method], metric), digits, unavailable=(metric == "shd")) for metric, digits in METRICS]
            body.append(f"    {label} & {method} & " + " & ".join(cells) + " \\\\")
        if suite != DATASETS[-1][0]:
            body.append("    \\cmidrule(l){1-10}")
    _replace_body(args.latex_dir / "visual_full.tex", body)

    # Main summary: mean of the four per-dataset means; SHD is unavailable.
    body = []
    for method in METHODS:
        cells = []
        for metric, digits in METRICS:
            stats = [s for suite, _, _ in DATASETS if (s := _stat(by_dataset[suite][method], metric)) is not None]
            if not stats:
                cells.append(NA if metric == "shd" else RED)
            else:
                cells.append(_cell((float(np.mean([s[0] for s in stats])), float(np.mean([s[1] for s in stats]))), digits))
        body.append(f"    {method} & " + " & ".join(cells) + " \\\\")
    _replace_body(args.latex_dir / "visual_results.tex", body)
    print("wrote visual_results.tex and visual_full.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
