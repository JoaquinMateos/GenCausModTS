"""Copy wall-clock times from a sequential timing run into the main results.

The benchmark tables suppress ``seconds`` when blocks run in parallel
(``GCMTS_TIMING=0``). Run the same blocks a second time, sequentially, into a
separate tree, then merge only the timing field:

    uv run python scripts/merge_timings.py --main outputs --timing outputs_timing

Rows are matched by ``(case, method, seed)`` when a seed exists and by
``(case, method)`` otherwise. A main value is only replaced when it is missing or
non-finite, and the timing value is finite; nothing else is touched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SKIP_SUFFIXES = ("_agg.json", "_config.json", "_progress.json", "_stats.json")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _key(row: dict[str, Any]) -> tuple:
    seed = row.get("seed")
    base = (str(row.get("case")), str(row.get("method")))
    return (*base, seed) if seed is not None else base


def merge_file(main_path: Path, timing_path: Path) -> tuple[int, int]:
    main_rows = json.loads(main_path.read_text(encoding="utf-8"))
    timing_rows = json.loads(timing_path.read_text(encoding="utf-8"))
    timing = {_key(row): row.get("seconds") for row in timing_rows}
    filled = missing = 0
    for row in main_rows:
        key = _key(row)
        value = timing.get(key)
        if not _finite(value):
            missing += 1
            continue
        if not _finite(row.get("seconds")):
            row["seconds"] = float(value)
            filled += 1
    main_path.write_text(json.dumps(main_rows, indent=2), encoding="utf-8")
    return filled, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main", nargs="+", type=Path,
                        default=[Path("outputs"), Path("outputs_shared")])
    parser.add_argument("--timing", nargs="+", type=Path,
                        default=[Path("outputs_timing"), Path("outputs_shared_timing")])
    args = parser.parse_args()

    total_filled = total_missing = 0
    for main_dir, timing_dir in zip(args.main, args.timing, strict=True):
        for path in sorted(main_dir.glob("*.json")):
            if path.name.endswith(SKIP_SUFFIXES):
                continue
            timing_path = timing_dir / path.name
            if not timing_path.exists():
                print(f"no timing file for {path.name}")
                continue
            filled, missing = merge_file(path, timing_path)
            total_filled += filled
            total_missing += missing
            print(f"{path.name:45s} filled={filled:4d} missing={missing:4d}")
    print(f"total: filled={total_filled}, still missing={total_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
