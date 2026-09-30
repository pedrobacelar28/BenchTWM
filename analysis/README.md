# Analysis

Downstream notebooks and plots should read the exported files, not private
runner objects:

- `bench research summarize` writes `runs.csv`, `aggregate.csv`,
  `planning_gain.csv`, `summary.json`, and `report.md` under
  `artifacts/reports/summary/`.
- `bench research collect` writes a flat index of every run, with its training,
  evaluation, OPE, and timing scalars, to `artifacts/reports/run_index.csv`.
- Each run directory holds `metrics_summary.json` plus the per-episode and
  per-step CSV files described in [docs/operations.md](../docs/operations.md).
