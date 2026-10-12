# ruff: noqa: E501 -- LaTeX templates are single long lines by design.
"""Generate the LaTeX experiment tables from benchmark JSON rows.

Reads the raw per-seed rows written by ``evaluation.report.save_rows`` and
writes ``latex/pic/experiments/crl.tex``, ``ctee.tex`` and ``stats.tex`` with
mean/std aggregation and the Friedman/Nemenyi summary. Missing suites leave the
corresponding rows as ``--`` (pending) so the manuscript always compiles.

Usage::

    uv run python scripts/make_paper_tables.py \
        --shared-dir outputs_shared --outputs-dir outputs \
        --latex-dir ../latex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcmts.data.real_registry import REAL_DATASETS
from gcmts.evaluation.benchmarks import (
    CRL_BENCHMARKS,
    CTEE_REAL_BENCHMARKS,
    CTEE_STATIC_BENCHMARKS,
    METHODS_FOR_KIND,
    crl_label,
)
from gcmts.evaluation.report import aggregate_rows
from gcmts.evaluation.stats import friedman_test, metric_matrix, nemenyi, plot_cd_diagram

__all__ = ["main"]


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def _fmt(
    aggregated: dict[str, Any], case: str, metric: str, *, bold: bool = False, digits: int = 2
) -> str:
    if case not in aggregated or metric not in aggregated[case]:
        return "--"
    entry = aggregated[case][metric]
    mean, std = entry["mean"], entry["std"]
    if np.isnan(mean):
        return "--"
    text = f"{mean:.{digits}f}"
    body = f"{text}\\tiny{{$\\pm${std:.{digits}f}}}"
    return f"\\textbf{{{body}}}" if bold else body


def _time(rows: list[dict[str, Any]], case: str, *, digits: int = 0) -> str:
    """Mean±std wall-clock training seconds per seed for ``case``."""
    values = [
        float(row["seconds"])
        for row in rows
        if row.get("case") == case and row.get("seconds") is not None
    ]
    if not values:
        return "--"
    mean, std = float(np.mean(values)), float(np.std(values))
    return f"{mean:.{digits}f}\\tiny{{$\\pm${std:.{digits}f}}}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


_CRL_STATS_METRICS = ["mcc", "r2_diag", "r2_sep", "shd", "triplet", "forecast1", "forecast4"]


def _sig_note(shared_dir: Path, suite: str) -> str:
    """Per-suite significance line from the Holm-corrected paired Wilcoxon JSON."""
    stats = _load_json(shared_dir / f"crl_shared_{suite}_stats.json")
    if not stats:
        return ""
    significant = [
        metric
        for metric in _CRL_STATS_METRICS
        if any(
            comparison.get("significant")
            for comparison in (stats.get(metric) or {}).get("comparisons", [])
        )
    ]
    if significant:
        text = "Holm-corrected paired Wilcoxon finds significant pairwise differences in: " + ", ".join(
            metric.replace("_", "\\_") for metric in significant
        )
    else:
        text = "No pairwise difference survives Holm correction in any column"
    return "    \\multicolumn{10}{@{}p{\\textwidth}@{}}{\\footnotesize " + text + ".}\\\\\n"


def _best_cases(aggregated: dict[str, Any], cases: list[str], metric: str, *, higher: bool) -> set[str]:
    """Cases statistically tied with the best: within ±std of the best mean.

    The best case(s) and every case whose mean lies within the best case's
    standard deviation are returned, so the bold entries show the best value
    together with the values indistinguishable from it at one std.
    """
    values: dict[str, float] = {}
    for case in cases:
        if case in aggregated and metric in aggregated[case]:
            value = aggregated[case][metric]["mean"]
            if not np.isnan(value):
                values[case] = value
    if not values:
        return set()
    best_case = max(values, key=values.get) if higher else min(values, key=values.get)
    best_value = values[best_case]
    best_std = float(aggregated[best_case][metric].get("std", 0.0) or 0.0)
    tolerance = best_std + 1e-9
    return {case for case, value in values.items() if abs(value - best_value) <= tolerance}


def _suite_block(
    rows: list[dict[str, Any]],
    suite: str,
    methods: list[str],
    *,
    shd_available: bool,
    seeds: int = 10,
    sde: bool = False,
) -> list[str]:
    subset = [row for row in rows if row.get("suite") == suite]
    if not subset:
        return []
    # Complete blocks only: every method must have all seeds before the block
    # is published, so partial runs never masquerade as ten-seed means.
    for method in methods:
        if sum(1 for row in subset if row.get("case") == method and "seed" in row) < seeds:
            return []
    aggregated = aggregate_rows(subset)
    if sde:
        # CEGEN operates in observation space: no mixing model, so MCC and the
        # latent-graph columns are undefined and its conditional W2 is the
        # primary structural score (reported in the MCC column).
        best_w2 = _best_cases(aggregated, methods, "w2", higher=False)
        best_f1 = _best_cases(aggregated, methods, "forecast1", higher=False)
        best_f4 = _best_cases(aggregated, methods, "forecast4", higher=False)
        lines = []
        for method in methods:
            if method not in aggregated:
                continue
            w2 = _fmt(aggregated, method, "w2", bold=method in best_w2, digits=4)
            f1 = _fmt(aggregated, method, "forecast1", bold=method in best_f1)
            f4 = _fmt(aggregated, method, "forecast4", bold=method in best_f4)
            lines.append(
                f"    {method} & {suite} & {w2} & n/a & n/a & n/a & n/a & {f1} & {f4} & {_time(subset, method)} \\\\"
            )
        return lines
    best_mcc = _best_cases(aggregated, methods, "mcc", higher=True)
    best_r2d = _best_cases(aggregated, methods, "r2_diag", higher=True)
    best_r2s = _best_cases(aggregated, methods, "r2_sep", higher=False)
    best_shd = _best_cases(aggregated, methods, "shd", higher=False) if shd_available else set()
    best_triplet = _best_cases(aggregated, methods, "triplet", higher=False)
    best_f1 = _best_cases(aggregated, methods, "forecast1", higher=False)
    best_f4 = _best_cases(aggregated, methods, "forecast4", higher=False)
    lines = []
    for method in methods:
        if method not in aggregated:
            continue
        mcc = _fmt(aggregated, method, "mcc", bold=method in best_mcc)
        r2d = _fmt(aggregated, method, "r2_diag", bold=method in best_r2d)
        r2s = _fmt(aggregated, method, "r2_sep", bold=method in best_r2s)
        shd = _fmt(aggregated, method, "shd", bold=method in best_shd) if shd_available else "n/a"
        triplet = _fmt(aggregated, method, "triplet", bold=method in best_triplet)
        f1 = _fmt(aggregated, method, "forecast1", bold=method in best_f1)
        f4 = _fmt(aggregated, method, "forecast4", bold=method in best_f4)
        lines.append(
            f"    {method} & {suite} & {mcc} & {r2d} & {r2s} & {shd} & {triplet} & {f1} & {f4} & {_time(subset, method)} \\\\"
        )
    return lines


def _baselines(suite_name: str) -> list[str]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_crl_shared import SUITES, _persistence_mse
    from sklearn.linear_model import Ridge

    from gcmts.core import move_batch

    if suite_name not in SUITES:
        return []
    suite = SUITES[suite_name]
    import torch

    eval_batch = move_batch(suite.make_generator(0).sample(4096), torch.device("cpu"))
    train_batch = suite.make_generator(12345).sample(8192)
    flat_x = train_batch.x[:, :-1].reshape(-1, train_batch.x.shape[-1]).numpy()
    flat_y = train_batch.x[:, 1:].reshape(-1, train_batch.x.shape[-1]).numpy()
    ar = Ridge(alpha=1e-3).fit(flat_x, flat_y)
    test_x = eval_batch.x[:, :-1].reshape(-1, eval_batch.x.shape[-1]).numpy()
    test_y = eval_batch.x[:, 1:].reshape(-1, eval_batch.x.shape[-1]).numpy()
    ar_mse = float(((ar.predict(test_x) - test_y) ** 2).mean())
    p1 = _persistence_mse(eval_batch, 1)
    p4 = _persistence_mse(eval_batch, 4)
    label = suite_name
    return [
        f"    Persistence & {label} & n/a & n/a & n/a & n/a & n/a & {p1:.2f} & {p4:.2f} & n/a \\\\",
        f"    AR(1) & {label} & n/a & n/a & n/a & n/a & n/a & {ar_mse:.2f} & -- & n/a \\\\",
    ]


def write_crl_table(shared_dir: Path, latex_dir: Path, *, seeds: int) -> None:
    rows: list[dict[str, Any]] = []
    for key in CRL_BENCHMARKS:
        rows.extend(_load(shared_dir / f"crl_shared_{key}.json"))

    header = (
        r"""\begin{table}[t]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.12}
  \caption{Head-to-head comparison of representative Identifiable CRL methods across the shared benchmark suites. All methods in a block share the same data, latent dimension and seed schedule, so methods from different branches (I.A--I.E) are directly comparable within a block. Methods are listed in a fixed order (iVAE, LEAP, TDRL, NCTRL, MOSAIC), not by performance. Mean$\pm$std over """
        + str(seeds)
        + r""" seeds. Benchmarks A.1--A.17 compare the temporal family (iVAE, LEAP, TDRL, NCTRL, MOSAIC) across seventeen linear, nonlinear and physical processes (NP/VAR and a harder variant, CartPole, Causal3DIdent, Lorenz-63/96, PCL, Kuramoto, Lotka--Volterra, Duffing, FitzHugh--Nagumo, Chua, van der Pol, pendulum, mountain-car, VARMA and Ornstein--Uhlenbeck), while Benchmarks B, C and D are single-method (Voronoi/CITRIS+iCITRIS, SDE/CEGEN, hierarchical/CHiLD). Lower is better for $R^2_{\text{sep}}$, SHD, the triplet distance and both forecasting-MSE columns; higher is better for MCC and $R^2_{\text{diag}}$. ``n/a'' marks a metric the benchmark cannot expose (control systems define no latent graph; the SDE suite has no mixing model, so CEGEN's conditional $W_2$ is reported in the MCC column); ``--'' a result still to be produced. The forecast columns are a secondary \emph{downstream-utility} diagnostic---the future is generated by the same known mechanism---not a soundness metric; persistence and AR(1) reference rows anchor them. ``Time (s)'' is the mean$\pm$std wall-clock training time per run on a single NVIDIA RTX~2080~Ti; it is hardware-dependent and secondary to the soundness metrics.}
  \label{tab:exp_crl}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} p{0.09\textwidth} p{0.11\textwidth} r r r r r r r r @{}}
    \toprule
    \textbf{Method} & \textbf{Benchmark} & \textbf{MCC $\uparrow$} & \textbf{$R^2_{\text{diag}}$ $\uparrow$} & \textbf{$R^2_{\text{sep}}$ $\downarrow$} & \textbf{SHD $\downarrow$} & \textbf{Triplet $\downarrow$} & \textbf{1-step MSE $\downarrow$} & \textbf{4-step MSE $\downarrow$} & \textbf{Time (s)} \\
    \midrule
"""
    )
    blocks: list[str] = []
    missing: list[str] = []
    for key, benchmark in CRL_BENCHMARKS.items():
        methods = METHODS_FOR_KIND[benchmark.kind]
        lines_block = _suite_block(
            rows,
            key,
            methods,
            shd_available=benchmark.shd_available,
            seeds=seeds,
            sde=benchmark.kind == "sde",
        )
        if lines_block:
            block = (
                "    \\multicolumn{10}{@{}l}{\\emph{"
                + crl_label(key)
                + "}}\\\\\n"
                + "\n".join(lines_block)
                + "\n"
            )
            if benchmark.kind == "temporal":
                block += "\n".join(_baselines(key)) + "\n"
            block += "    \\cmidrule(l){1-10}\n" + _sig_note(shared_dir, key)
            blocks.append(block)
        else:
            blocks.append(
                "    \\multicolumn{10}{@{}l}{\\textcolor{red}{\\emph{"
                + crl_label(key)
                + " --- run pending}}}\\\\\n"
            )
            missing.append(benchmark.label)
    if missing:
        pending = (
            "\\textcolor{red}{Blocks are added as their ten-seed runs complete; pending: "
            + ", ".join(missing)
            + ". "
        )
        close = "}}"
    else:
        pending = ""
        close = "}"
    footer = (
        r"""    \bottomrule
    \addlinespace
    \multicolumn{10}{@{}p{\textwidth}@{}}{"""
        + pending
        + r"""The image-render variant of Causal3DIdent remains pending. Structural recovery is scored with the post-hoc protocol of the source methods (per-target sparse regression, Hungarian permutation alignment) and SHD/edge-AUPRC are only defined where the benchmark exposes a latent graph. Real-world series are deliberately excluded: they expose no causal ground truth, so they cannot populate any soundness metric (rule R1)."""
        + close
        + r""" \\
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "crl.tex"
    out.write_text(header + "".join(blocks) + footer, encoding="utf-8")
    print(f"wrote {out}")


def write_crl_summary(shared_dir: Path, latex_dir: Path, *, seeds: int) -> None:
    """Compact methods-by-benchmarks MCC matrix for the main text."""
    temporal = [(key, bench) for key, bench in CRL_BENCHMARKS.items() if bench.kind == "temporal"]
    methods = METHODS_FOR_KIND["temporal"]
    aggregated: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    complete: dict[str, bool] = {}
    for key, _ in temporal:
        rows = _load(shared_dir / f"crl_shared_{key}.json")
        aggregated[key] = aggregate_rows(rows) if rows else {}
        complete[key] = bool(rows) and all(_case_complete(rows, m, seeds) for m in methods)

    ranks: dict[str, list[int]] = {m: [] for m in methods}
    for key, _ in temporal:
        if not complete[key]:
            continue
        means = {m: aggregated[key][m]["mcc"]["mean"] for m in methods}
        order = sorted(methods, key=lambda m: means[m], reverse=True)
        for position, method in enumerate(order, start=1):
            ranks[method].append(position)

    def cell(key: str, method: str) -> str:
        if not complete[key]:
            return "--"
        best = max(methods, key=lambda m: aggregated[key][m]["mcc"]["mean"])
        return _fmt(aggregated[key], method, "mcc", digits=2, bold=method == best)

    header_cells = ["\\textbf{Method}"] + [f"\\textbf{{{b.label}}}" for _, b in temporal]
    header_cells.append("\\textbf{Mean rank $\\downarrow$}")
    body = []
    for method in methods:
        cells = [method] + [cell(key, method) for key, _ in temporal]
        cells.append(f"{np.mean(ranks[method]):.2f}" if ranks[method] else "--")
        body.append("    " + " & ".join(cells) + " \\\\")

    caption = (
        "Summary of representation recovery (MCC, higher is better) for the five temporal CRL "
        "methods across the seventeen shared benchmarks A.1--A.17. ``Mean rank'' is the average "
        "rank over the benchmarks with a complete block (one is best); ``--'' marks a benchmark "
        "whose ten-seed run is still pending. The full per-benchmark table, with all metrics and "
        "wall-clock times, is Table~\\ref{tab:exp_crl} in Appendix~\\ref{app:full_results}, and the "
        "Friedman/Nemenyi test of the ranking is Table~\\ref{tab:exp_stats}."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.12}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_crl_summary}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{@{} l """
        + "r " * (len(temporal) + 1)
        + r"""@{}}
    \toprule
    """
        + " & ".join(header_cells)
        + r""" \\
    \midrule
"""
        + "\n".join(body)
        + r"""
    \bottomrule
  \end{tabular}}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "crl_summary.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def _case_complete(rows: list[dict[str, Any]], case: str, seeds: int) -> bool:
    """True when ``case`` has at least ``seeds`` independent runs."""
    return sum(1 for row in rows if row.get("case") == case) >= seeds


def write_ctee_table(outputs_dir: Path, latex_dir: Path, *, seeds: int) -> None:
    rows = _load(outputs_dir / "ctee_benchmarks.json")
    if not rows:
        print("no CTEE rows; leaving ctee.tex unchanged")
        return
    aggregated = aggregate_rows(rows)
    order = [
        ("ganite_harmonic", "GANITE", "II.A adjust", "Harmonic"),
        ("wu_iptw_harmonic", "Wu-IPTW", "II.A adjust", "Harmonic"),
        ("crn_harmonic", "CRN", "II.B repr.", "Harmonic"),
        ("cepae_harmonic", "CEPAE", "II.B repr.", "Harmonic"),
        ("catsg_harmonic", "CaTSG", "II.C latent", "Harmonic"),
        ("ganite_tumour", "GANITE", "II.A adjust", "Tumour"),
        ("wu_iptw_tumour", "Wu-IPTW", "II.A adjust", "Tumour"),
        ("crn_tumour", "CRN", "II.B repr.", "Tumour"),
        ("cepae_tumour", "CEPAE", "II.B repr.", "Tumour"),
        ("catsg_tumour", "CaTSG", "II.C latent", "Tumour"),
        ("ganite_linear_sem", "GANITE", "II.A adjust", "Linear SEM"),
        ("wu_iptw_linear_sem", "Wu-IPTW", "II.A adjust", "Linear SEM"),
        ("crn_linear_sem", "CRN", "II.B repr.", "Linear SEM"),
        ("cepae_linear_sem", "CEPAE", "II.B repr.", "Linear SEM"),
        ("catsg_linear_sem", "CaTSG", "II.C latent", "Linear SEM"),
    ]

    def _abs_best(cases: list[str]) -> set[str]:
        values = {
            case: abs(aggregated[case]["cf_mbe"]["mean"])
            for case in cases
            if case in aggregated and "cf_mbe" in aggregated[case]
        }
        if not values:
            return set()
        best_case = min(values, key=values.get)
        tolerance = float(aggregated[best_case]["cf_mbe"].get("std", 0.0) or 0.0) + 1e-9
        return {case for case, value in values.items() if abs(value - values[best_case]) <= tolerance}

    def _best_for(cases: list[str]) -> dict[str, set[str]]:
        return {
            "pehe": _best_cases(aggregated, cases, "pehe", higher=False),
            "ate_error": _best_cases(aggregated, cases, "ate_error", higher=False),
            "cf_mae": _best_cases(aggregated, cases, "cf_mae", higher=False),
            "cf_mbe": _abs_best(cases),
            "mmd2": _best_cases(aggregated, cases, "mmd2", higher=False),
            "recon_axiom": _best_cases(aggregated, cases, "recon_axiom", higher=False),
            "effectiveness_ok": _best_cases(aggregated, cases, "effectiveness_ok", higher=True),
            "reversibility": _best_cases(aggregated, cases, "reversibility", higher=False),
        }

    best = {
        "Harmonic": _best_for([case for case, _, _, d in order if d == "Harmonic"]),
        "Tumour": _best_for([case for case, _, _, d in order if d == "Tumour"]),
        "Linear SEM": _best_for([case for case, _, _, d in order if d == "Linear SEM"]),
    }

    def row(case: str, method: str, branch: str, dataset: str) -> str:
        b = best[dataset]
        cells = [
            _fmt(aggregated, case, "pehe", digits=2, bold=case in b["pehe"]),
            _fmt(aggregated, case, "ate_error", digits=2, bold=case in b["ate_error"]),
            _fmt(aggregated, case, "cf_mae", digits=2, bold=case in b["cf_mae"]),
            _fmt(aggregated, case, "cf_mbe", digits=2, bold=case in b["cf_mbe"]),
            _fmt(aggregated, case, "mmd2", digits=4, bold=case in b["mmd2"]),
            _fmt(aggregated, case, "recon_axiom", digits=2, bold=case in b["recon_axiom"]),
            _fmt(aggregated, case, "effectiveness_ok", digits=1, bold=case in b["effectiveness_ok"]),
            _fmt(aggregated, case, "reversibility", digits=2, bold=case in b["reversibility"]),
            _time(rows, case),
        ]
        return f"    {method} & {branch} & {dataset} & " + " & ".join(cells) + " \\\\"

    available = [item for item in order if _case_complete(rows, item[0], seeds)]
    pending_datasets = sorted({item[3] for item in order if item not in available})
    lines = [row(*item) for item in available]
    caption = (
        "Counterfactual and treatment-effect estimators (Branches II.A observed-data adjustment, II.B representation-based identification, II.C latent-confounder \\& environment models) on two benchmarks with exact counterfactuals: the damped harmonic oscillator and the tumour-growth simulator (time-varying chemo/radio treatment). Methods are grouped by branch, not by performance. Mean $\\pm$ standard deviation over "
        + str(seeds)
        + " seeds; lower is better except ``Eff.'' (higher is better) and ``MBE'' (closer to zero is better). ``Time (s)'' is the mean$\\pm$std wall-clock training time per run on a single NVIDIA RTX~2080~Ti (hardware-dependent). ``Rec.'' is the reconstruction axiom $\\text{MAE}(\\hat X(\\mathrm{do}(a_{\\mathrm{f}})),X^{\\mathrm{f}})$, ``Eff.'' the fraction of seeds whose average estimated effect has the correct sign, and ``Rev.'' the reversibility MAE after applying the reverse treatment. See Section~\\ref{sec:exp_ctee}."
    )
    footer = ""
    if pending_datasets:
        footer = (
            "\n    \\addlinespace\n    \\multicolumn{12}{@{}p{\\textwidth}@{}"
            "}{\\textcolor{red}{Incomplete blocks suppressed until all ten seeds of every method are available: "
            + ", ".join(pending_datasets)
            + ".}}\\\\"
        )
    ctee_stats = _load_json(outputs_dir / "ctee_stats.json") or {}
    ctee_sig = [
        metric
        for metric in ("pehe", "ate_error", "cf_mae")
        if any(comparison.get("significant") for comparison in ctee_stats.get(metric, []))
    ]
    if ctee_sig:
        footer += (
            "\n    \\addlinespace\n    \\multicolumn{12}{@{}p{\\textwidth}@{}}"
            "{\\footnotesize Holm-corrected paired Wilcoxon finds significant pairwise differences in: "
            + ", ".join(metric.replace("_", "\\_") for metric in ctee_sig)
            + ".}\\\\"
        )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_ctee}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{l l l r r r r r r r r r}
    \toprule
    \textbf{Method} & \textbf{Branch} & \textbf{Benchmark} & \textbf{PEHE $\downarrow$} & \textbf{$\epsilon_{\text{ATE}}$ $\downarrow$} & \textbf{CF MAE $\downarrow$} & \textbf{MBE $\to 0$} & \textbf{MMD$^2$ $\downarrow$} & \textbf{Rec. $\downarrow$} & \textbf{Eff. $\uparrow$} & \textbf{Rev. $\downarrow$} & \textbf{Time (s)} \\
    \midrule
"""
        + "\n".join(lines)
        + footer
        + r"""
    \bottomrule
  \end{tabular}}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "ctee.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def write_sdd_table(outputs_dir: Path, latex_dir: Path, *, seeds: int) -> None:
    rows = _load(outputs_dir / "sdd_benchmarks.json")
    if not rows:
        print("no SDD rows; leaving sdd.tex unchanged")
        return
    aggregated = aggregate_rows(rows)
    order = [
        ("dann_multidomain", "DANN"),
        ("vrex_multidomain", "VREx"),
        ("sync_multidomain", "SYNC"),
        ("erm_multidomain", "ERM (baseline)"),
    ]
    lines = []
    missing_cases = []
    sdd_cases = [case for case, _ in order]
    sdd_best = {
        "ood_mse": _best_cases(aggregated, sdd_cases, "ood_mse", higher=False),
        "ood_mae": _best_cases(aggregated, sdd_cases, "ood_mae", higher=False),
        "domain_accuracy": _best_cases(aggregated, sdd_cases, "domain_accuracy", higher=True),
        "mig": _best_cases(aggregated, sdd_cases, "mig", higher=True),
        "cds": _best_cases(aggregated, sdd_cases, "cds", higher=True),
    }
    for case, label in order:
        if case not in aggregated:
            missing_cases.append(label)
            continue
        if not _case_complete(rows, case, seeds):
            missing_cases.append(label)
            continue
        lines.append(
            f"    {label} & Multi-domain synthetic & "
            + " & ".join(
                _fmt(aggregated, case, metric, bold=case in sdd_best[metric])
                for metric in ("ood_mse", "ood_mae", "domain_accuracy", "mig", "cds")
            )
            + " & " + _time(rows, case) + " \\\\"
        )
    oracle = aggregated.get("oracle_causal")
    if oracle is not None:
        lines.append(
            "    \\cmidrule(l){1-8}\n    Oracle (true causal factors) & Multi-domain synthetic & "
            + _fmt(aggregated, "oracle_causal", "ood_mse")
            + " & "
            + _fmt(aggregated, "oracle_causal", "ood_mae")
            + " & -- & -- & -- & -- \\\\"
        )
    caption = (
        "Static--dynamic (invariance) sub-branch of non-stationarity \\& environment shift (Branch I.B.2) "
        "on the multi-domain SCM of Section~\\ref{sec:exp_sdd}. Models train on domains $0$--$3$ "
        "and are tested on held-out domain $4$, whose spurious correlation has the opposite sign. "
        "Methods are listed by sub-branch (invariant prediction, then four-way split), not by "
        "performance. Mean $\\pm$ standard deviation over " + str(seeds) + " seeds; lower is better "
        "for OOD error and higher for domain-identification accuracy, MIG and CDS. The last row is a "
        "ridge regression on the \\emph{true} causal factors and is the achievable floor. ``Time (s)'' is the mean$\\pm$std wall-clock training time per run on a single NVIDIA RTX~2080~Ti."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_sdd}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} p{0.15\textwidth} p{0.20\textwidth} r r r r r r @{}}
    \toprule
    \textbf{Method} & \textbf{Benchmark} & \textbf{OOD MSE $\downarrow$} & \textbf{OOD MAE $\downarrow$} & \textbf{Dom.-ID Acc. $\uparrow$} & \textbf{MIG $\uparrow$} & \textbf{CDS $\uparrow$} & \textbf{Time (s)} \\
    \midrule
"""
        + "\n".join(lines)
        + (
            "\n    \\addlinespace\n    \\multicolumn{8}{@{}p{\\textwidth}@{}"
            "}{\\textcolor{red}{Incomplete blocks suppressed until all ten seeds of every method are available: "
            + ", ".join(missing_cases)
            + ".}}\\\\"
            if missing_cases
            else ""
        )
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "sdd.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def write_stats_section(shared_dir: Path, latex_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for stem in (
        "crl_shared_grid_a",
        "crl_shared_grid_b",
        "crl_shared_grid",
    ):
        rows.extend(_load(shared_dir / f"{stem}.json"))
    if not rows:
        print("no grid rows; leaving stats.tex unchanged")
        return

    lines: list[str] = []
    figure_names: list[str] = []
    for metric, higher in (
        ("mcc", True),
        ("r2_diag", True),
        ("r2_sep", False),
        ("forecast1", False),
    ):
        matrix, cases, problems = metric_matrix(rows, metric)
        complete = matrix[~np.isnan(matrix).any(axis=1)]
        if complete.shape[0] < 2 or complete.shape[1] < 3:
            continue
        friedman = friedman_test(matrix)
        result = nemenyi(matrix, higher_is_better=higher)
        ranks = ", ".join(
            f"{case}: {rank:.2f}"
            for case, rank in zip(cases, result["average_ranks"].tolist(), strict=True)
        )
        lines.append(
            f"\\textbf{{{metric}}} & {friedman['n_problems']:.0f} & "
            f"{friedman['statistic']:.2f} & {friedman['p_value']:.3g} & "
            f"{result['critical_difference']:.2f} & {ranks} \\\\"
        )
        if metric in {"mcc", "r2_diag", "forecast1"}:
            name = f"cd_{metric}.png"
            plot_cd_diagram(
                {
                    case: float(rank)
                    for case, rank in zip(cases, result["average_ranks"].tolist(), strict=True)
                },
                float(result["critical_difference"]),
                path=latex_dir / "pic" / "experiments" / name,
                title=f"{metric} ({'higher' if higher else 'lower'} is better)",
            )
            figure_names.append(name)

    n_problems = len({str(row.get("dataset", "")) for row in rows})
    section = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{Statistical comparison of the five temporal CRL methods over the """
        + f"{n_problems}"
        + r"""-problem configuration grid (Section~\ref{sec:exp_protocol}). The Friedman test is the omnibus test; the Nemenyi critical difference (CD) is the smallest average-rank gap significant at $\alpha=0.05$ for this number of problems and methods. Average ranks use one as best (lower is better).}
  \label{tab:exp_stats}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l r r r r p{0.42\textwidth} @{}}
    \toprule
    \textbf{Metric} & \textbf{N} & \textbf{Friedman $\chi^2$} & \textbf{$p$} & \textbf{CD} & \textbf{Average ranks} \\
    \midrule
"""
        + "\n".join(lines)
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    if figure_names:
        includes = "\n".join(
            f"  \\includegraphics[width=.32\\textwidth]{{pic/experiments/{name}}}"
            for name in figure_names
        )
        section += (
            "\n\\begin{figure*}[t]\n  \\centering\n"
            + includes
            + "\n  \\caption{Critical-difference diagrams for the eighteen-problem CRL configuration grid: methods whose average ranks are joined by a bar are not significantly different at $\\alpha=0.05$ (Nemenyi). Lower ranks are better.}\n"
            "  \\label{fig:cd_diagrams}\n\\end{figure*}\n"
        )
    out = latex_dir / "pic" / "experiments" / "stats.tex"
    out.write_text(section, encoding="utf-8")
    print(f"wrote {out}")


def write_ctee_static_table(
    outputs_dir: Path, latex_dir: Path, *, replications: int = 100
) -> None:
    """Static counterfactual benchmark table (one row-block per dataset)."""
    order_template = [
        ("ganite_{d}", "GANITE", "II.A adjust"),
        ("cepae_{d}", "CEPAE", "II.B repr."),
        ("crn_{d}", "CRN", "II.B repr."),
        ("catsg_{d}", "CaTSG", "II.C latent"),
        ("ridge_t_{d}", "Ridge T-learner", "baseline"),
        ("nearest_neighbour_{d}", "Nearest neighbour", "baseline"),
    ]
    blocks: list[str] = []
    for dataset, label in CTEE_STATIC_BENCHMARKS.items():
        rows = _load(outputs_dir / f"ctee_static_{dataset}.json")
        order = [(case.format(d=dataset), method, branch) for case, method, branch in order_template]
        complete = bool(rows) and all(
            _case_complete(rows, case, replications) for case, _, _ in order
        )
        if complete:
            aggregated = aggregate_rows(rows)
            cases = [case for case, _, _ in order]
            best = {
                metric: _best_cases(aggregated, cases, metric, higher=False)
                for metric in ("pehe", "ate_error", "policy_risk")
            }
            for case, method, branch in order:
                blocks.append(
                    f"    {method} & {branch} & {label} & "
                    + " & ".join(
                        _fmt(aggregated, case, metric, digits=2, bold=case in best[metric])
                        for metric in ("pehe", "ate_error", "policy_risk")
                    )
                    + " & " + _time(rows, case) + " \\\\"
                )
        else:
            blocks.append(
                "    \\multicolumn{7}{@{}l}{\\textcolor{red}{"
                + label + ": " + str(replications) + "-replication run pending.}} \\\\"
            )
        blocks.append("    \\cmidrule(l){1-7}")
    caption = (
        "Static counterfactual benchmarks (IHDP, Twins, Jobs/LaLonde, News and CPS). Each of the "
        + str(replications)
        + " replications is a separate benchmark instance; estimators train on the standard train "
        "split and are scored on the held-out split against the noiseless potential means "
        "$\\mu_0,\\mu_1$, with PEHE and $\\epsilon_{\\text{ATE}}$ for individual and marginal "
        "effects and the policy risk of the induced threshold rule (lower is better throughout). "
        "Ridge T-learner and nearest-neighbour matching anchor the scale. The datasets are "
        "semi-synthetic---covariates come from real studies while the potential outcomes are "
        "simulated (for Twins, both twin outcomes are observed)---which is what exposes the "
        "counterfactual ground truth (rule R1). ``Time (s)\'\' is the mean$\\pm$std wall-clock "
        "training time per replication on a single NVIDIA RTX~2080~Ti."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_ctee_static}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l l r r r r @{}}
    \toprule
    \textbf{Method} & \textbf{Branch} & \textbf{Benchmark} & \textbf{PEHE $\downarrow$} & \textbf{$\epsilon_{\text{ATE}}$ $\downarrow$} & \textbf{Policy risk $\downarrow$} & \textbf{Time (s)} \\
    \midrule
"""
        + "\n".join(blocks)
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "ctee_static.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def write_adjacent_table(outputs_dir: Path, latex_dir: Path, *, seeds: int = 10) -> None:
    """Adjacent-branch (III) and II.D results table (headline metric per method)."""
    rows = _load(outputs_dir / "adjacent_benchmarks.json")
    if not rows:
        print("no adjacent rows; leaving adjacent_results.tex unchanged")
        return
    aggregated = aggregate_rows(rows)
    order = [
        ("GBTPL", "III.A", "Propagation path", "Path-F1 $\\uparrow$", "path_f1"),
        ("LacaDM", "III.B", "Multi-objective MDP", "Hypervolume $\\uparrow$", "hypervolume"),
        ("CausalConceptTS", "III.C", "Concept-annotated series", "Faithfulness $\\uparrow$", "faithfulness"),
        ("CausalGAN", "II.D", "Binary-label SCM", "CF MAE $\\downarrow$", "cf_mae"),
    ]
    lines = []
    for case, branch, bench, metric_label, metric in order:
        if not _case_complete(rows, case, seeds):
            continue
        lines.append(
            f"    {case} & {branch} & {bench} & {metric_label} & "
            + _fmt(aggregated, case, metric)
            + " & "
            + _time(rows, case)
            + " \\\\"
        )
    if not lines:
        print("no complete adjacent blocks; leaving adjacent_results.tex unchanged")
        return
    caption = (
        "Adjacent-branch (III) and given-mechanism (II.D) results over "
        + str(seeds)
        + " seeds, each on the benchmark that exposes its object's ground truth and scored on its "
        "headline soundness metric (Section~\\ref{sec:experiments}). The three adjacent objects---a "
        "propagation path, a policy and an attribution---are heterogeneous, so the comparison is "
        "per-object rather than head-to-head. ``Time (s)'' is the mean$\\pm$std wall-clock training "
        "time per run on a single NVIDIA RTX~2080~Ti."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_adjacent_results}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l l l r r @{}}
    \toprule
    \textbf{Method} & \textbf{Branch} & \textbf{Benchmark} & \textbf{Headline metric} & \textbf{Value} & \textbf{Time (s)} \\
    \midrule
"""
        + "\n".join(lines)
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "adjacent_results.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def write_real_inventory_table(latex_dir: Path) -> None:
    """Inventory of every real / real-derived dataset named in the review."""
    role_names = {"soundness": "soundness", "external_validity": "external validity"}
    body = []
    for dataset in REAL_DATASETS.values():
        body.append(
            "    "
            + " & ".join(
                [
                    dataset.label,
                    dataset.level,
                    dataset.ground_truth,
                    dataset.modality,
                    role_names[dataset.role],
                    dataset.access,
                ]
            )
            + " \\\\"
        )
    caption = (
        "Real and real-derived datasets named in this review, with the highest rung of Pearl's "
        "hierarchy they expose, the ground truth they provide, their modality, their role under "
        "rule~R1 (soundness when the causal object is observed; external validity otherwise) and "
        "their access mode. Only ``soundness'' datasets---recorded interventions and randomised "
        "trials---may populate a soundness metric; the rest support downstream-utility and "
        "out-of-distribution comparisons only."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_real_inventory}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l p{0.20\textwidth} l l l @{}}
    \toprule
    \textbf{Dataset} & \textbf{Level} & \textbf{Ground truth} & \textbf{Modality} & \textbf{Role} & \textbf{Access} \\
    \midrule
"""
        + "\n".join(body)
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "real_inventory.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def write_external_validity_table(outputs_dir: Path, latex_dir: Path, *, seeds: int) -> None:
    """External-validity table on real panels (forecasting/reconstruction MSE)."""
    rows = _load(outputs_dir / "real_benchmarks.json")
    if not rows:
        print("no real-benchmark rows; writing pending placeholder")
    aggregated = aggregate_rows(rows) if rows else {}
    datasets = sorted({str(row.get("dataset", "")) for row in rows})
    blocks: list[str] = []
    for dataset in datasets:
        cases = sorted({str(row["case"]) for row in rows if row.get("dataset") == dataset})
        if not all(_case_complete(rows, case, seeds) for case in cases):
            continue
        best = {
            metric: _best_cases(aggregated, cases, metric, higher=False)
            for metric in ("forecast1_mse", "forecast4_mse")
        }
        for case in cases:
            method = case.split("/", 1)[1]
            blocks.append(
                "    \\multicolumn{5}{@{}l}{\\emph{" + dataset + "}}\\\\\n"
                if case == cases[0]
                else ""
            )
            blocks.append(
                f"    {method} & "
                + _fmt(aggregated, case, "forecast1_mse", digits=3, bold=case in best["forecast1_mse"])
                + " & "
                + _fmt(aggregated, case, "forecast4_mse", digits=3, bold=case in best["forecast4_mse"])
                + " & "
                + _fmt(aggregated, case, "recon_mse", digits=3)
                + " & "
                + _time(rows, case)
                + " \\\\"
            )
        blocks.append("    \\cmidrule(l){1-5}")
    if not blocks:
        blocks.append(
            "    \\multicolumn{5}{@{}l}{\\textcolor{red}{Real-panel external-validity run "
            "(ETT, Jena, COVID-19 policy panel) pending.}} \\\\"
        )
    caption = (
        "External-validity evaluation on real time series (ETT, Jena, COVID-19 policy panel), over "
        + str(seeds)
        + " seeds. Real series expose no latent ground truth, so only downstream-utility proxies "
        "are reported (rule~R1): one- and four-step forecasting MSE on a chronologically later "
        "held-out split and reconstruction MSE, with persistence and AR(1) baselines. A lower MSE "
        "means the learned representation generalises better; no soundness claim is made. "
        "``Time (s)'' is the mean$\\pm$std wall-clock training time per run on a single NVIDIA "
        "RTX~2080~Ti."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_external_validity}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l r r r r @{}}
    \toprule
    \textbf{Method} & \textbf{1-step MSE $\downarrow$} & \textbf{4-step MSE $\downarrow$} & \textbf{Recon.\ MSE $\downarrow$} & \textbf{Time (s)} \\
    \midrule
"""
        + "\n".join(blocks)
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "external_validity.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def write_ctee_real_table(outputs_dir: Path, latex_dir: Path, *, replications: int = 1) -> None:
    """Real effect cohorts (LaLonde RCT, NHEFS): marginal-effect (ATE) errors only."""
    order = [
        ("ganite", "GANITE", "II.A adjust"),
        ("cepae", "CEPAE", "II.B repr."),
        ("crn", "CRN", "II.B repr."),
        ("catsg", "CaTSG", "II.C latent"),
        ("ridge_t", "Ridge T-learner", "baseline"),
        ("nearest_neighbour", "Nearest neighbour", "baseline"),
    ]
    blocks: list[str] = []
    for key, label in CTEE_REAL_BENCHMARKS.items():
        rows = _load(outputs_dir / f"ctee_static_{key}.json")
        cases = [f"{method}_{key}" for method, _, _ in order]
        if not rows or not all(_case_complete(rows, case, replications) for case in cases):
            blocks.append(
                "    \\multicolumn{4}{@{}l}{\\textcolor{red}{"
                + label + ": run pending.}} \\\\"
            )
            continue
        aggregated = aggregate_rows(rows)
        best = _best_cases(aggregated, cases, "ate_error", higher=False)
        for (_method, name, branch), case in zip(order, cases, strict=True):
            blocks.append(
                f"    {name} & {branch} & {label} & "
                + _fmt(aggregated, case, "ate_error", digits=3, bold=case in best)
                + " & "
                + _time(rows, case)
                + " \\\\"
            )
        blocks.append("    \\cmidrule(l){1-5}")
    caption = (
        "Real effect-estimation cohorts: the randomised NSW/LaLonde trial (Dehejia--Wahba) and the "
        "observational NHEFS smoking-cessation cohort (Hern\\'an--Robins). These expose only the "
        "\\emph{marginal} effect---by randomisation for LaLonde and as a published adjusted estimate "
        "for NHEFS---so individual metrics (PEHE, policy risk) are undefined and only "
        "$\\epsilon_{\\text{ATE}}$ is reported; for NHEFS this is a reference comparison, not a "
        "soundness claim. ``Time (s)'' is the mean$\\pm$std wall-clock training time per run on a "
        "single NVIDIA RTX~2080~Ti."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_ctee_real}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l l r r @{}}
    \toprule
    \textbf{Method} & \textbf{Branch} & \textbf{Cohort} & \textbf{$\epsilon_{\text{ATE}}$ $\downarrow$} & \textbf{Time (s)} \\
    \midrule
"""
        + "\n".join(blocks)
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "ctee_real.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX experiment tables.")
    parser.add_argument("--shared-dir", type=Path, default=Path("outputs_shared"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--latex-dir", type=Path, default=Path("../latex"))
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    write_crl_summary(args.shared_dir, args.latex_dir, seeds=args.seeds)
    write_crl_table(args.shared_dir, args.latex_dir, seeds=args.seeds)
    write_ctee_table(args.outputs_dir, args.latex_dir, seeds=args.seeds)
    write_ctee_static_table(args.outputs_dir, args.latex_dir)
    write_sdd_table(args.outputs_dir, args.latex_dir, seeds=args.seeds)
    write_adjacent_table(args.outputs_dir, args.latex_dir, seeds=args.seeds)
    write_stats_section(args.shared_dir, args.latex_dir)
    write_real_inventory_table(args.latex_dir)
    write_external_validity_table(args.outputs_dir, args.latex_dir, seeds=args.seeds)
    write_ctee_real_table(args.outputs_dir, args.latex_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
