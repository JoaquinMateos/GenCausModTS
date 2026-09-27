"""Check CRL sub-taxonomy coverage against the taxonomy table.

Parses the bundled taxonomy table (``docs/taxonomy_table.tex``) and reports, for
every sub-taxonomy, which implemented ``gcmts`` method covers it (matched by
BibTeX key).

Usage::

    uv run python scripts/taxonomy_coverage.py
    uv run python scripts/taxonomy_coverage.py --strict   # nonzero if any row uncovered
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Prefer the bundled copy so the script works standalone; fall back to the
# companion-repository layout when this library is checked out inside it.
_CANDIDATES = [
    ROOT / "docs" / "taxonomy_table.tex",
    ROOT.parent / "latex" / "pic" / "taxonomy" / "table.tex",
]
TABLE = next((path for path in _CANDIDATES if path.exists()), _CANDIDATES[0])

# Implemented methods mapped to their BibTeX keys.
IMPLEMENTED: dict[str, str] = {
    "iVAE": "khemakhemVariationalAutoencodersNonlinear2020",
    "LEAP": "weiranyaoLearningTemporallyCausal2021",
    "TDRL": "yaoTemporallyDisentangledRepresentation2022",
    "NCTRL": "songTemporallyDisentangledRepresentation2023",
    "Slow Flows": "edouardpineauTimeSeriesSource2020",
    "CITRIS": "lippeCITRISCausalIdentifiability2022",
    "MOSAIC": "shichengfanMOSAICModuleDiscovery2026",
    "CEGEN": "carlremlingerConditionalLossDeep2021",
}
_KEY_TO_METHOD = {key: name for name, key in IMPLEMENTED.items()}


@dataclass
class Row:
    category: str
    sub_taxonomy: str
    keys: list[str]
    methods: list[str]

    @property
    def covered(self) -> bool:
        return bool(self.methods)


def parse_table(path: Path) -> list[Row]:
    category = ""
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line or "\\cite" not in line:
            continue
        protected = line.replace(r"\&", "@AMP@")  # keep escaped ampersands intact
        cat_match = re.search(r"\\textbf\{([^}]+)\}", protected)
        if cat_match:
            category = cat_match.group(1).replace("@AMP@", "&").strip()
        parts = protected.split("&")
        if len(parts) < 3:
            continue
        sub = parts[1].replace("@AMP@", "&").strip().rstrip("}")
        keys_match = re.search(r"\\cite\{([^}]+)\}", protected)
        if not keys_match:
            continue
        keys = [k.strip() for k in keys_match.group(1).split(",") if k.strip()]
        methods = [_KEY_TO_METHOD[k] for k in keys if k in _KEY_TO_METHOD]
        rows.append(Row(category=category, sub_taxonomy=sub, keys=keys, methods=methods))
    return rows


def render(rows: list[Row]) -> str:
    lines = [
        "# Taxonomy Coverage",
        "",
        "| Category | Sub-taxonomy | Covered | Implemented method(s) |",
        "| --- | --- | :---: | --- |",
    ]
    for row in rows:
        methods = ", ".join(row.methods) if row.methods else "—"
        lines.append(
            f"| {row.category} | {row.sub_taxonomy} | {'✅' if row.covered else '❌'} | {methods} |"
        )
    covered = sum(1 for r in rows if r.covered)
    lines += [
        "",
        f"Covered sub-taxonomies: **{covered}/{len(rows)}**.",
        "",
        "Implemented methods: " + ", ".join(IMPLEMENTED),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail if any row uncovered")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs", help="where to write the report"
    )
    args = parser.parse_args()

    rows = parse_table(TABLE)
    report = render(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "taxonomy_coverage.md").write_text(report, encoding="utf-8")
    (args.out_dir / "taxonomy_coverage.json").write_text(
        json.dumps(
            [
                {
                    "category": r.category,
                    "sub_taxonomy": r.sub_taxonomy,
                    "covered": r.covered,
                    "methods": r.methods,
                }
                for r in rows
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)

    crl_rows = [r for r in rows if r.category.startswith("1.")]
    crl_ok = all(r.covered for r in crl_rows)
    print(f"CRL sub-taxonomies covered: {sum(r.covered for r in crl_rows)}/{len(crl_rows)}")
    if args.strict and not all(r.covered for r in rows):
        return 1
    return 0 if crl_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
