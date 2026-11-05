"""Statistical comparison of methods with StaTDS.

Builds case-by-method matrices from the benchmark JSON files and runs the
Friedman test (with Iman--Davenport), the Nemenyi critical difference, and Holm
post-hoc comparisons, using ``statds`` rather than a private reimplementation.

    uv run python scripts/statds_report.py --outputs outputs \
        --shared outputs_shared --out docs/statistical_analysis.md

Each block is a separate family of hypotheses: CRL representation quality (MCC),
CRL forecasting, static CATE (PEHE), and real-panel forecasting. A method is a
column; a case (benchmark, replication or panel) is a row. Rows with a missing
value are dropped for that block, and the drop count is reported.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402
from statds.no_parametrics import friedman, holm, nemenyi  # noqa: E402

CRL_SUITES = [
    "npvar", "npvar_linear", "npvar_hard", "cartpole", "causal3d", "causal3d_img",
    "lorenz", "lorenz96", "pcl", "kuramoto", "lotka_volterra", "duffing",
    "fitzhugh_nagumo", "chua", "van_der_pol", "pendulum", "mountain_car", "varma",
    "ou",
]
CRL_METHODS = ["iVAE", "LEAP", "TDRL", "NCTRL", "MOSAIC"]
STATIC_PANELS = ["ihdp", "acic", "twins", "news"]
STATIC_METHODS = ["catsg", "cepae", "crn", "ganite", "ridge_t", "nearest_neighbour"]


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def _matrix(rows: list[dict], cases: list[str], methods: list[str], metric: str,
            case_key: str = "case", method_key: str = "method",
            unit_key: str | None = None, agg: str = "mean") -> pd.DataFrame:
    """Case x method matrix, aggregated over seeds/units when given."""
    frame: dict[str, dict[str, float]] = {}
    for row in rows:
        case = str(row.get(case_key))
        method = str(row.get(method_key))
        value = row.get(metric)
        if case not in cases or method not in methods:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        key = case if unit_key is None else f"{case}:{row.get(unit_key)}"
        frame.setdefault(key, {}).setdefault(method, []).append(float(value))  # type: ignore[arg-type]
    out = {}
    for key, per_method in frame.items():
        out[key] = {
            method: (float(np.mean(vals)) if agg == "mean" else float(np.median(vals)))
            for method, vals in per_method.items()
        }
    data = pd.DataFrame.from_dict(out, orient="index")
    data = data.reindex(columns=methods)
    data.index.name = "case"
    return data


def _block(name: str, data: pd.DataFrame, *, minimize: bool, alpha: float = 0.05) -> dict:
    complete = data.dropna(axis=0, how="any")
    dropped = len(data) - len(complete)
    if len(complete) < 3 or complete.shape[1] < 3:
        return {"name": name, "skipped": True, "dropped": dropped,
                "n_cases": len(complete), "n_methods": complete.shape[1]}
    frame = complete.reset_index()
    ranks, stat, p_value, crit, hypothesis = friedman(frame, alpha, minimize=minimize)
    # StaTDS evaluates the chi-square tail with a truncated series, which
    # saturates near 1e-11 (e.g. chi2=72, df=4 gives 3.5e-11 instead of
    # 8.6e-15). Ranks and the Nemenyi CD are exact; recompute the p-value.
    p_exact = float(chi2.sf(float(stat), complete.shape[1] - 1))
    _, cd, _ = nemenyi(ranks, len(complete), alpha)
    ordered = sorted(ranks, key=lambda m: ranks[m])
    post: dict = {}
    try:
        table = holm(ranks, len(complete), alpha, control=ordered[0], type_rank="Friedman")[0]
        post = table.to_dict(orient="records") if hasattr(table, "to_dict") else {}
    except Exception as error:  # noqa: BLE001 - reporting must not fail
        post = {"error": str(error)}
    return {
        "name": name, "skipped": False, "dropped": dropped,
        "n_cases": len(complete), "n_methods": complete.shape[1],
        "ranks": {m: float(ranks[m]) for m in ordered},
        "statistic": float(stat), "p_value": float(p_value), "p_value_exact": p_exact,
        "critical_value": float(crit), "hypothesis": hypothesis,
        "critical_difference": float(cd), "posthoc": post,
        "means": {m: float(complete[m].mean()) for m in complete.columns},
        "medians": {m: float(complete[m].median()) for m in complete.columns},
        "std": {m: float(complete[m].std(ddof=1)) for m in complete.columns},
    }


def _markdown(blocks: list[dict], header: str) -> str:
    lines = [header, ""]
    for block in blocks:
        lines.append(f"### {block['name']}")
        lines.append("")
        if block.get("skipped"):
            lines.append(
                f"Not testable: {block['n_cases']} complete cases, "
                f"{block['n_methods']} methods ({block['dropped']} rows dropped)."
            )
            lines.append("")
            continue
        lines.append(
            f"Friedman over {block['n_cases']} cases: chi2 = {block['statistic']:.2f}, "
            f"p = {block['p_value_exact']:.3g} (StaTDS statistic; p recomputed with the exact "
            f"chi-square tail). Nemenyi critical distance = "
            f"{block['critical_difference']:.2f} ranks. {block['hypothesis']} "
            f"({block['dropped']} rows dropped for missing values.)"
        )
        lines.append("")
        lines.append("| Method | Mean rank | Mean | Median | SD |")
        lines.append("| --- | --- | --- | --- | --- |")
        for method, rank in block["ranks"].items():
            lines.append(
                f"| {method} | {rank:.2f} | {block['means'][method]:.3f} | "
                f"{block['medians'][method]:.3f} | {block['std'][method]:.3f} |"
            )
        post = block.get("posthoc") or {}
        if isinstance(post, list) and post:
            lines.append("")
            lines.append("Holm post-hoc against the best-ranked method:")
            lines.append("")
            lines.append("| Comparison | Z | p | adjusted p | Result |")
            lines.append("| --- | --- | --- | --- | --- |")
            for record in post:
                def _fmt_p(value: object) -> str:
                    number = float(value) if isinstance(value, (int, float)) else float("nan")
                    return "<1e-12" if number < 1e-12 else f"{number:.3g}"
                lines.append(
                    f"| {record.get('Comparison')} | {record.get('Statistic (Z)', 0):.2f} | "
                    f"{_fmt_p(record.get('p-value'))} | "
                    f"{_fmt_p(record.get('Adjusted p-value'))} | "
                    f"{record.get('Results')} |"
                )
        elif isinstance(post, dict) and "error" in post:
            lines.append("")
            lines.append(f"Holm post-hoc unavailable: {post['error']}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--shared", type=Path, default=Path("outputs_shared"))
    parser.add_argument("--out", type=Path, default=Path("docs/statistical_analysis.md"))
    parser.add_argument("--json-out", type=Path, default=Path("docs/statistical_analysis.json"))
    args = parser.parse_args()

    blocks: list[dict] = []

    crl_rows: list[dict] = []
    for suite in CRL_SUITES:
        crl_rows += _rows(args.shared / f"crl_shared_{suite}.json")
    mcc = _matrix(crl_rows, CRL_SUITES, CRL_METHODS, "mcc", case_key="dataset")
    blocks.append(_block("CRL representation quality (MCC, higher is better)", mcc,
                         minimize=False))
    f1 = _matrix(crl_rows, CRL_SUITES, CRL_METHODS, "forecast1", case_key="dataset")
    blocks.append(_block("CRL one-step forecasting (MSE, lower is better)", f1,
                         minimize=True))

    grid_rows = _rows(args.shared / "crl_shared_grid_a.json") + _rows(
        args.shared / "crl_shared_grid_b.json"
    )
    grid_cases = sorted({str(r.get("dataset")) for r in grid_rows})
    grid_mcc = _matrix(grid_rows, grid_cases, CRL_METHODS, "mcc", case_key="dataset")
    blocks.append(_block("18-problem grid, MCC", grid_mcc, minimize=False))
    grid_f1 = _matrix(grid_rows, grid_cases, CRL_METHODS, "forecast1", case_key="dataset")
    blocks.append(_block("18-problem grid, one-step forecasting (MSE, lower is better)",
                         grid_f1, minimize=True))

    cte_rows = _rows(args.outputs / "ctee_benchmarks.json")
    cte_methods = sorted({str(r.get("method")) for r in cte_rows})
    for dataset in ("harmonic", "tumour", "linear_sem"):
        data = _matrix(cte_rows, [dataset], cte_methods, "pehe",
                       case_key="dataset", unit_key="seed")
        blocks.append(_block(f"Temporal CTEE, {dataset} (sqrt-PEHE, lower is better)",
                             data, minimize=True))

    for panel in STATIC_PANELS:
        rows = _rows(args.outputs / f"ctee_static_{panel}.json")
        data = _matrix(rows, [panel], STATIC_METHODS, "pehe",
                       case_key="dataset", unit_key="seed")
        blocks.append(_block(f"Static CATE, {panel.upper()} (sqrt-PEHE, lower is better)",
                             data, minimize=True))

    real_rows = _rows(args.outputs / "real_benchmarks.json")
    panels = sorted({str(r.get("dataset")) for r in real_rows})
    real = _matrix(real_rows, panels, CRL_METHODS, "forecast1_mse",
                   case_key="dataset", unit_key="seed")
    blocks.append(_block("Real panels, one-step forecasting (MSE, lower is better)",
                         real, minimize=True))

    from importlib.metadata import version as package_version

    text = _markdown(
        blocks,
        "# Statistical analysis (StaTDS)\n\n"
        "Friedman + Nemenyi + Holm, computed with `statds` "
        + package_version("statds")
        + ".",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.json_out.write_text(json.dumps(blocks, indent=2), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
