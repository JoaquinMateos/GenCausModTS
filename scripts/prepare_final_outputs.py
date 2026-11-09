#!/usr/bin/env python3
"""Seed a final output directory from an earlier run without stale NCTRL rows.

The earlier run is kept for its non-NCTRL rows; the corrected NCTRL adapter is
re-run by the final benchmark.  Visual suites are not present in the earlier
run and therefore start from zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs_final_v2"))
    parser.add_argument("--dest", type=Path, default=Path("outputs_final_v3"))
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(args.source.glob("crl_shared_*.json")):
        if path.stem.endswith(("_agg", "_config", "_progress", "_stats")):
            continue
        try:
            rows = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        kept = [row for row in rows if row.get("case") != "NCTRL" and row.get("method") != "NCTRL"]
        out = args.dest / path.name
        out.write_text(json.dumps(kept, indent=2), encoding="utf-8")
        copied += len(kept)
        print(f"{path.name}: kept {len(kept)}/{len(rows)} non-NCTRL rows")
    print(f"seeded {args.dest} with {copied} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
