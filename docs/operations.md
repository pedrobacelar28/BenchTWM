# Command Line and Artifacts

`bench` (equivalently `python -m benchtwm`).
Global options, accepted before or after the subcommand: `--artifact-root`
(default `artifacts/`), `--config-root` (default `configs/`), `--json`, and
`--color never`.

## Commands

| Command | Purpose |
|---|---|
| `doctor` | Packages, CUDA/GPUs, registered environments and algorithms, config errors |
| `catalog` | Datasets (with build state) and methods |
| `validate [--smoke]` | Check configs; optionally step every environment once |
| `dataset list \| show <id> \| build <id>` | Discover and collect datasets |
| `inspect dataset <id>` | Array shapes and dtypes of a built dataset |
| `baselines <id>` | Evaluate the environment's reference policies in the simulator |
| `train <dataset> <method>` | One run: train, evaluate, OPE |
| `suite <dataset>` | Queue every (or `--method`-selected) method over `--seeds` |
| `status [run]` | Recent runs and their status |
| `inspect run <run>` | Optimization summary, direct evaluation, OPE diagnostics |
| `evaluate <run>` | Fresh simulator evaluation of a completed run |
| `ope <run>` | Recompute OPE (`--target-mode`, `--bootstrap`, `--write-canonical`) |
| `report <run>` | Markdown and JSON report under `reports/` |
| `research sweep \| matrix \| tune \| collect \| summarize \| reevaluate` | Experiment management |

Run options (`train`, `suite`, `research sweep|matrix|tune`): `--set
KEY=JSON`, `--seed`, `--device auto|cpu|cuda`, `--force`, `--dry-run`,
`--no-progress`. Queue options: `--seeds`, `--execute` (without it only the
plan is printed), `--jobs`, `--gpus 0,1`, `--follow`.

A `<run>` is a run directory, a `run_id`, or a unique directory name.

## Queues

`suite` and `research sweep|matrix` run each job in its own worker process with
its own log, progress file, and optional GPU. Completed runs are skipped unless
`--force` is given. Queue state and `suite_results.json` (jobs, per-method
statistics, and planning gains) are written to `artifacts/queues/<timestamp>/`.

## Run lifecycle

`run_manifest.json` records identity, provenance (git commit, Python, torch,
devices), and status:

| Status | Meaning |
|---|---|
| `planned`, `queued`, `running` | Not finished |
| `completed` | All artifacts written |
| `failed` | An exception stopped the run (`error` holds it) |
| `interrupted` | Stopped by Ctrl+C or SIGTERM |
| `stale` | Shown by `status` for a `running` run whose process no longer exists |

## Artifact layout

```text
artifacts/
  datasets/<env>/<behavior>/<tier>/{train,val,ope_test}.npz, metadata.json, episodes_summary.csv
  runs/<env>/<behavior>/<tier>/<method>/seed-<n>--<hash>/
      config.json  run_manifest.json  train_history.csv
      checkpoint_best.pt  checkpoint_last.pt
      eval_summary.json  episodes_summary.csv  trajectories.csv  environment_spec.json
      ope_summary.json  ope_bootstrap.csv  metrics_summary.json
      evaluations/<timestamp>/  ope/<timestamp>/      (later recomputations)
  queues/<timestamp>/{jobs,logs,progress}/, queue_manifest.json, suite_results.json
  reports/summary/  reports/baselines/  reports/<run>/  reports/run_index.csv
  studies/<dataset>__<method>.sqlite3
```

`metrics_summary.json` collects the scalar results of a run (training summary,
evaluation summary, OPE headline, timings) in one document for downstream
analysis. Runs written before this layout (under `.../offline/<method>/`)
are still discovered and loaded.

## Downstream analysis

Notebooks and plots should read the exported files, not private runner objects:

- `bench research summarize` writes `runs.csv`, `aggregate.csv`,
  `planning_gain.csv`, `summary.json`, and `report.md` under
  `artifacts/reports/summary/`.
- `bench research collect` writes a flat index of every run, with its training,
  evaluation, OPE, and timing scalars, to `artifacts/reports/run_index.csv`.
- Each run directory holds `metrics_summary.json` plus the per-episode and
  per-step CSV files listed in the artifact layout above.
