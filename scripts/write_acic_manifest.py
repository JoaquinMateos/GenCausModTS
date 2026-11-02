"""Write the ACIC snapshot manifest used by the loader.

Replication ``i`` is the ``i``-th directory in sorted-hash order, so a different
download changes the index -> replication map. Run this once on the snapshot you
intend to use and ship ``data/acic/manifest.json`` with the data; the loader then
refuses a different snapshot.

    uv run python scripts/write_acic_manifest.py --data-dir data/acic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gcmts.data.manifest import write_manifest


def _names(directory: Path) -> list[str]:
    return [
        f"{path.name}/{name}"
        for path in sorted(directory.iterdir())
        if path.is_dir() and (path / "x.csv").exists()
        for name in ("x.csv", "z.csv", "y.csv", "mu0.csv", "mu1.csv")
        if (path / name).exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/acic"))
    args = parser.parse_args()
    names = _names(args.data_dir)
    if not names:
        raise SystemExit(f"No ACIC replication directories under {args.data_dir}.")
    path = write_manifest(args.data_dir, names)
    print(f"wrote {path} ({len(names)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
