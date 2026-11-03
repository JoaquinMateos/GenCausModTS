# Release checklist (T11)

Status: **blocked on the pending runs** (grid, IHDP 100, rendered-image suite,
sequential timing). Everything below is ready except the tag, the archive and
the DOI.

## What must be frozen

- `outputs/` and `outputs_shared/` — per-run rows, configs, aggregates and
  progress files. These are the primary record; the paper tables are derived.
- `outputs_timing/` and `outputs_shared_timing/` — only if the Time columns are
  kept; they hold the sequential wall-clock rows merged into the main trees.
- `data/synthetic/manifest.json` — pool digests and shapes.
- `data/acic/manifest.json` — ACIC snapshot digest (`6c3d2df1f150` as of the
  current tree).
- `docs/audit_pending.md` (or the final `docs/audit_*.md`) — must report
  `FAIL=0`.
- `docs/statistical_analysis.md` / `.json` — regenerated from the frozen rows.
- `latex/pic/experiments/*.tex` — the tables generated from those rows.
- `CITATION.cff` — version, release date and DOI.

## Steps

1. Confirm the runs are complete: `outputs/ctee_static_ihdp_progress.json` has
   `done=600`; `outputs_shared/crl_shared_grid_{a,b}.json` have 450 rows each
   and 0 errors; `outputs_shared/crl_shared_causal3d_img.json` has 50 rows.
2. Regenerate the audit, statistics and tables from the stored rows
   (`scripts/verify_outputs.py`, `scripts/statds_report.py`,
   `scripts/make_paper_tables.py`). Do not edit tables by hand.
3. Rebuild the paper (`latexmk -pdf main.tex`) and check 0 errors, 0
   overfull/underfull, 0 undefined references.
4. Record the exact environment (host, torch, CUDA/MPS) in the audit and in the
   timing caption.
5. Bump `version` and `date-released` in `CITATION.cff`, commit, and tag
   `v1.0.0`.
6. Archive the tagged commit on Zenodo; add the DOI to `CITATION.cff` and to the
   paper's data-availability statement.
7. Verify that a clean checkout plus the archived `outputs*/` reproduces every
   table with `scripts/make_paper_tables.py`.

## Acceptance

- The DOI resolves to the tagged tree.
- The archived tree reproduces every table from the stored rows, with no
  recomputation.
- `docs/audit_*.md` is part of the archive and reports `FAIL=0`.

## Not part of the release

- CPS (T7) is absent because the data are not redistributable; the paper keeps
  it as an optional manual panel.
- The II.D and adjacent-branch additions (T9/T10) are separate releases.
