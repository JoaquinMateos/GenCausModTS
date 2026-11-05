# Seed and statistical-test policy

Status: 2026-11-06. Applies to every local and `gpu_212` run.

Current GPU run: five seeds at the Optuna-selected 200-epoch budget and early
stopping. Ten seeds remain the target for the frozen final table; the current
run is a five-seed trial/intermediate comparison and must be labelled as such.

## Seed tiers

| Purpose | Seeds |
|---|---:|
| Implementation smoke / unit stability | 1 |
| Pilot / method-selection trial | 3 |
| Trial / intermediate comparison | 5 |
| Final table | 10 |

- No pilot or trial may use 10 seeds. Ten seeds are reserved for a frozen final table.
- Static CATE panels use replications, not seeds: 100 for IHDP/ACIC/Twins, 50 for News.
- Real cohorts and deterministic baselines are single runs.
- Any table caption must state the seed count it actually used.

## Statistical tests

Use inferential tests **only when there is a genuine set of distinct benchmarks/problems that can be treated as paired blocks**:

- CRL shared benchmark: 19 suites. Friedman + Nemenyi/Holm are appropriate.
- CRL configuration grid: 18 problems. Friedman + Nemenyi are appropriate.
- No pooled test over the three heterogeneous CTEE temporal datasets; report per-dataset mean and standard deviation.
- No pooled test over static panels with different outcomes/protocols; report per-panel mean and standard deviation.
- SDD has one benchmark; report mean and standard deviation, not a significance claim.
- Adjacent branches and the completed-branch representatives are different objects; report mean and standard deviation as coverage evidence, not a ranking test.

For blocks without a valid multi-dataset test, the paper reports the real obtained values with `mean ± std`; it must not describe a difference as significant or not significant.

## Enforcement

- Pilot commands must pass `--seeds 3` (or `--seeds 5` for a trial).
- Final commands must pass `--seeds 10`, and the output directory/table must be marked final.
- `scripts/run_ctee_benchmarks.py` and `scripts/run_crl_shared.py` default to 5 seeds; the final run scripts pass 10 explicitly.
- The table generator emits inferential language only for CRL/grid tables.
