"""Audit benchmark outputs before they are reported.

Checks, in order:

* ``--pools``: every persisted synthetic pool under ``--data-dir`` is finite;
* ``errors.log``: steps that exited non-zero in the orchestrator run;
* progress files: unfinished steps, recorded errors, and ``done > total``;
* result files: seeds/replications per case against the stored configuration,
  and metrics that are non-finite for every row of a case.

A non-finite metric is ``EXPECTED`` when the method cannot produce it by design
(baselines have no training loss, persistence has no reconstruction, real cohorts
have no counterfactual truth, CEGEN has no latent MCC) and ``REVIEW`` otherwise.
A learned run whose metrics are all non-finite is ``FAIL``.

Exit status is 1 when any ``FAIL`` is found, so the script can gate a rerun.

    uv run python scripts/verify_outputs.py --pools --report docs/audit.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_METHODS = {"ridge_t", "nearest_neighbour", "Persistence", "AR(1)", "oracle_causal"}
RCT_COHORTS = {"lalonde_rct", "jobs_rct", "oregon_rct", "nhefs"}
# Fields that are bookkeeping, not evaluation metrics. ``final_loss`` is a training
# diagnostic: a diverged run can still report a finite loss on the last step.
NON_METRIC = {"seed", "seconds", "n_parameters", "epochs", "final_loss"}
SKIP_SUFFIXES = ("_agg.json", "_config.json", "_progress.json", "_stats.json")


@dataclass
class Finding:
    severity: str  # FAIL | REVIEW | WARN | OK
    where: str
    message: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_pools(data_dir: Path) -> list[Finding]:
    import torch

    from gcmts.data.synthetic_store import batch_from_state, non_finite_fields

    findings: list[Finding] = []
    paths = sorted(data_dir.glob("*/seed_*.pt"))
    if not paths:
        return [Finding("FAIL", str(data_dir), "no synthetic pools found")]
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        bad = {
            split: non_finite_fields(batch_from_state(payload[split]))
            for split in ("train", "eval")
        }
        bad = {split: names for split, names in bad.items() if names}
        if bad:
            findings.append(Finding("FAIL", str(path), f"non-finite tensors: {bad}"))
    if not any(f.severity == "FAIL" for f in findings):
        findings.append(Finding("OK", str(data_dir), f"{len(paths)} pools finite"))
    return findings


def check_error_log(outputs: Path) -> list[Finding]:
    log = outputs / "logs" / "errors.log"
    if not log.exists():
        return []
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if "exit=" in line]
    return [Finding("FAIL", "errors.log", line.strip()) for line in lines]


def check_progress(directory: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(directory.glob("*_progress.json")):
        data = _load_json(path)
        where = path.name
        done, total = int(data.get("done", 0)), int(data.get("total", 0))
        if data.get("errors"):
            findings.append(Finding("FAIL", where, f"{data['errors']} error(s) recorded"))
        if not data.get("finished"):
            findings.append(Finding("FAIL", where, f"not finished ({done}/{total})"))
        if done > total:
            findings.append(
                Finding("WARN", where, f"done {done} > total {total} (progress accounting)")
            )
    return findings


def _usable(row: dict[str, Any]) -> bool:
    """True when the row carries at least one finite evaluation metric."""
    return any(
        _finite(value) for key, value in row.items() if key not in NON_METRIC and key != "case"
    )


def _expected_units(config: dict[str, Any]) -> int | None:
    cfg = config.get("config", {})
    if "seeds" in cfg:
        return int(cfg["seeds"])
    if "replications" in cfg:
        return int(cfg["replications"])
    return None


def _nonfinite_class(case: str, method: str, dataset: str, key: str) -> str:
    if key == "final_loss" and method in BASELINE_METHODS:
        return "EXPECTED"
    if key == "recon_mse" and method in {"Persistence", "AR(1)"}:
        return "EXPECTED"
    if key == "forecast4_mse" and method in {"Persistence", "AR(1)"}:
        return "EXPECTED"
    if key in {"pehe", "policy_risk"} and dataset in RCT_COHORTS:
        return "EXPECTED"
    if key in {"mcc", "r2_diag", "r2_sep", "r2_matched", "dci"} and method == "CEGEN":
        return "EXPECTED"
    if key in {"mig", "cds"} and method == "Oracle":
        return "EXPECTED"
    return "REVIEW"


def check_results(directory: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(SKIP_SUFFIXES):
            continue
        rows = _load_json(path)
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        if "case" not in rows[0]:
            continue  # not a result file (e.g. the orchestrator's step summary)
        config_path = path.with_name(path.stem + "_config.json")
        expected = _expected_units(_load_json(config_path)) if config_path.exists() else None
        where = path.name
        if any("seed" not in row for row in rows):
            findings.append(
                Finding("REVIEW", where, "rows do not record a seed; seed count not verified")
            )
        cases: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            cases.setdefault(str(row.get("case")), []).append(row)
        for case, case_rows in cases.items():
            method = str(case_rows[0].get("method"))
            dataset = str(case_rows[0].get("dataset", ""))
            seeds = {row.get("seed") for row in case_rows} - {None, -1}
            deterministic = method in BASELINE_METHODS or case.startswith("oracle")
            count = len(case_rows)
            if expected is not None and not deterministic:
                if "seed" in case_rows[0] and len(seeds) < expected:
                    findings.append(
                        Finding("FAIL", f"{where}:{case}", f"{len(seeds)}/{expected} seeds")
                    )
                elif "seed" not in case_rows[0] and count < expected:
                    findings.append(
                        Finding("FAIL", f"{where}:{case}", f"{count}/{expected} runs")
                    )
            unusable = [row for row in case_rows if not _usable(row)]
            if unusable and not deterministic:
                findings.append(
                    Finding(
                        "FAIL",
                        f"{where}:{case}",
                        f"{len(unusable)} run(s) with no finite metric (diverged run stored)",
                    )
                )
                continue
            keys = {
                key
                for row in case_rows
                for key, value in row.items()
                if key not in NON_METRIC and key != "case" and isinstance(value, float)
            }
            for key in sorted(keys):
                if not any(_finite(row.get(key)) for row in case_rows):
                    klass = _nonfinite_class(case, method, dataset, key)
                    findings.append(
                        Finding(klass if klass == "EXPECTED" else "REVIEW",
                                f"{where}:{case}", f"all-NaN metric '{key}'")
                    )
    return findings


def _render(findings: list[Finding]) -> str:
    order = {"FAIL": 0, "REVIEW": 1, "WARN": 2, "EXPECTED": 3, "OK": 4}
    lines = ["| Status | Where | Finding |", "| --- | --- | --- |"]
    for item in sorted(findings, key=lambda f: (order.get(f.severity, 9), f.where)):
        lines.append(f"| {item.severity} | `{item.where}` | {item.message} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--shared", type=Path, default=Path("outputs_shared"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--pools", action="store_true", help="also load and check every pool")
    parser.add_argument("--report", type=Path, default=None, help="write a markdown report")
    parser.add_argument(
        "--pools-only", action="store_true",
        help="check only the synthetic pools (used as a gate before a benchmark)",
    )
    args = parser.parse_args()

    findings: list[Finding] = []
    if args.pools or args.pools_only:
        findings += check_pools(args.data_dir)
    if not args.pools_only:
        findings += check_error_log(args.outputs)
        for directory in (args.outputs, args.shared):
            findings += check_progress(directory)
            findings += check_results(directory)

    text = _render(findings)
    levels = ("FAIL", "REVIEW", "WARN", "EXPECTED", "OK")
    counts = {level: sum(f.severity == level for f in findings) for level in levels}
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(summary)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"# Output audit\n\n{summary}\n\n{text}\n", encoding="utf-8")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
