# BenchTWM

BenchTWM is an offline-first benchmark suite for structured reinforcement
learning and world models. It currently provides ICU-Sepsis and EpiCare under
one dataset, training, direct-evaluation, OPE, and artifact contract.

It is research software, not a clinical decision-support system.

## Start Here

Install the dependencies listed in `requirements.txt` in your Python
environment, then run every command from the repository root.

```bash
# Check the local environment, GPU visibility, and vendored benchmark cores.
python scripts/bench.py doctor

# See exactly which offline datasets and methods are available.
python scripts/bench.py catalog

# Generate the small EpiCare SMART dataset, including held-out OPE data.
python scripts/bench.py dataset build epicare/smart/dev

# Inspect schema, behavior-policy provenance, splits, and arrays.
python scripts/bench.py inspect dataset epicare/smart/dev

# Preview the resolved offline DQN run before spending compute.
python scripts/bench.py train epicare/smart/dev dqn --dry-run

# Train, evaluate in the simulator, and compute OPE when configured.
python scripts/bench.py train epicare/smart/dev dqn

# Find the run, inspect its separate evidence streams, and write a report.
python scripts/bench.py status
python scripts/bench.py inspect run <run-id-or-path>
python scripts/bench.py report <run-id-or-path>
```

Offline training is the default. `train` never creates data implicitly: when a
dataset is missing it prints the `dataset build` command required to prepare
the logged train/validation/OPE splits.

## Main Commands

| Command | Purpose |
|---|---|
| `doctor` | Verify dependencies, local cores, CUDA, and artifact root. |
| `catalog` | List curated dataset IDs, method compatibility, scale, and OPE support. |
| `dataset list\|show\|build` | Discover, inspect, and generate offline data. |
| `train` | Run one resolved offline or explicit online recipe. |
| `suite` | Preview or execute a local multi-method/multi-seed queue. |
| `status`, `inspect`, `report` | Recover evidence and artifacts without opening Python. |
| `evaluate`, `ope`, `baselines` | Run fresh simulator evaluation, held-out OPE, or environment baselines. |
| `research` | Run explicit sweeps, matrices, Optuna tuning, collection, and re-evaluation. |
| `validate` | Validate the catalog and optionally smoke-test each environment. |

Use `--dry-run` before commands that write or consume significant compute.
Inspection commands support `--json`, for example:

```bash
python scripts/bench.py catalog --json
python scripts/bench.py status --json
```

## Artifact Layout

New generated files are ignored by Git and live only under `artifacts/`:

```text
artifacts/
  datasets/<dataset-id>/{train,val,ope_test}.npz
  runs/<env>/<dataset>/<mode>/<method>/seed-<n>--<hash>/
  queues/<timestamp>/{jobs,logs,queue_manifest.json}
  reports/
  studies/
```

Each run stores `run_manifest.json`, its resolved `config.json`, checkpoints,
optimization histories, simulator evaluation, and optional OPE outputs. The
manifest records the recipe, config fingerprint, code revision, Python, Torch,
device, and lifecycle state.

The clean interface reads and writes only the versioned `artifacts/` layout;
legacy outputs are outside the benchmark protocol and are not included in
reports.

## Examples and Documentation

- [Examples](examples/README.md)
- [CLI and operations](docs/operations.md)
- [Recipes and research](docs/recipes.md)
- [Benchmark and environment contract](docs/benchmarks.md)
- [Offline data schema](docs/data-schema.md)
- [Training, direct evaluation, and OPE](docs/evaluation.md)
- [Adding an environment or algorithm](docs/extending.md)
- [Technical specification](docs/paper/main.tex)

## Repository Layout

```text
src/            Environments, algorithms, data, evaluation, and operations
configs/        Versioned catalog, dataset recipes, method recipes, research matrices
scripts/        The single CLI entry point: bench.py
analysis/       Notes for downstream analysis using exported artifacts
examples/       Small shell and Python workflows
docs/           Operational and technical documentation
tests/          Unit, integration, and smoke tests
third_party/    Vendored runtime benchmark cores and notices
```
