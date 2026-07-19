# Structured Offline RL World-Model Benchmark Suite

This repository is a reproducible research suite for structured offline RL and
world-model agents. It currently integrates two healthcare-oriented benchmark
environments:

- **ICU-Sepsis**: a tabular MDP derived from MIMIC-III records.
- **EpiCare**: a procedurally generated POMDP for longitudinal treatment
  regimes with hidden disease state and observed symptoms.

The suite keeps the original ICU workflow compatible while exposing a common
environment, dataset, evaluation, and reporting contract. It is a benchmark
and research codebase, not a clinical decision-support system.

## Start Here

```bash
cd /scratch/pedrobacelar/BenchTWM/zmain_repo
pip install -r requirements.txt

# Discover all supported experiment configs.
python scripts/bench.py list
python scripts/bench.py list --env epicare

# Generate the small EpiCare SMART dataset and train DQN offline.
python scripts/bench.py dataset --config experiments/datasets/dev/epicare_smart.json
python scripts/bench.py run --config experiments/offline/dev/epicare_dqn.json

# Preview an entire suite; --execute is required to launch it.
python scripts/bench.py suite --env epicare --mode offline --tier dev

# Validate the implementation.
PYTHONPATH=. python -m unittest discover -s tests -v
```

All previous commands remain supported, including `src/run_config.py`,
`src/offline/generate_dataset.py`, and `scripts/run_sweep.py`.

## Documentation

- [Operations and commands](docs/operations.md)
- [Benchmarks and environment contract](docs/benchmarks.md)
- [Offline data schema](docs/data-schema.md)
- [Algorithms](docs/algorithms.md)
- [Training, direct evaluation, and OPE](docs/evaluation.md)
- [Adding an environment or algorithm](docs/extending.md)
- [Conference-style technical specification (LaTex)](docs/paper/main.tex)

The source papers used for the benchmark descriptions are retained outside the
package at `../repos_usados/papers_txt/`. The runnable third-party cores are
vendored under `third_party/`.

## Layout

```text
src/            Runtime code: environments, algorithms, models, data, evaluation
experiments/    Versioned JSON configurations, grouped by mode and tier
scripts/        CLI and compatible historical launch helpers
analysis/       Reports, trajectory plots, tables, and qualitative analysis
docs/           Operational and technical documentation
tests/          Unit, regression, and smoke tests
third_party/    Vendored benchmark cores and their licenses
```

Generated datasets, checkpoints, reports, and trial databases remain local and
are intentionally excluded from version control.
