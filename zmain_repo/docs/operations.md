# Operations and Reproducibility

Run every command from `zmain_repo/`. The standard dependency set is in
`requirements.txt`; benchmark cores are vendored in `third_party/`.

## Canonical CLI

`scripts/bench.py` is a thin operational layer over the existing Python
functions. It does not create a second experiment format and does not replace
legacy scripts.

```bash
# List configs discovered from experiments/**/*.json.
python scripts/bench.py list
python scripts/bench.py list --env epicare --kind offline --tier dev

# Generate a dataset. --config avoids ambiguous catalog matches.
python scripts/bench.py dataset \
  --config experiments/datasets/dev/epicare_smart.json

# Run one resolved element of a JSON parameter sweep.
python scripts/bench.py run \
  --config experiments/offline/dev/epicare_dqn.json --index 0

# Preview every matching config. Nothing starts without --execute.
python scripts/bench.py suite --env epicare --mode offline --tier dev
python scripts/bench.py suite --env epicare --mode offline --tier dev --execute

# Evaluate declared environment baselines (SMART, SoC, random, oracle for EpiCare).
python scripts/bench.py baselines \
  --config experiments/datasets/dev/epicare_smart.json --episodes 100

# Examine serialized artifacts without opening Python.
python scripts/bench.py inspect-dataset datasets/offline/epicare/dev/smart/1000/train.npz
python scripts/bench.py inspect-run results/dqn_offline/<run-hash>
```

The catalog filters mean:

| Filter | Meaning |
|---|---|
| `--env` | `icu_sepsis` or `epicare` |
| `--kind` | `datasets`, `online`, or `offline` |
| `--tier` | `dev` or `final` |
| `--algorithm` | Algorithm family such as `dqn` or `raw_model` |
| `--name` | Config stem or relative config path without `.json` |

## Compatible Legacy Commands

The following remain stable and are useful for automation already built around
them:

```bash
python src/offline/generate_dataset.py experiments/datasets/dev/epicare_smart.json
python src/run_config.py experiments/offline/dev/epicare_dqn.json 0
python scripts/run_sweep.py -j experiments/offline/dev/epicare_dqn.json
python scripts/evaluate_baselines.py experiments/datasets/dev/epicare_smart.json --episodes 100
PYTHONPATH=. python -m unittest discover -s tests -v
```

`scripts/run_offline_matrix.py` and the Optuna helpers are retained historical
automation. They are useful for their intended ICU matrices, but are not the
recommended starting point for a new multi-environment experiment.

## Run Lifecycle

1. Generate a dataset configuration. This creates `train.npz`, `val.npz`, and
   optionally `ope_test.npz` plus `metadata.json`.
2. Run an online or offline configuration. The runner resolves sweep values,
   trains, saves checkpoints, runs final simulator evaluation, and optionally
   runs OPE.
3. Inspect `metrics_summary.json` first. It links training summaries, direct
   evaluation, and method-specific losses in one readable file.
4. Use `episodes_summary.csv` and `trajectories.csv` for analysis. They carry
   `environment_spec.json`, so readers can recover action labels and metrics.

## Output Contract

Each run is stored at `results/<method_id>/<signature>/`. The signature is a
hash of the resolved configuration, which prevents accidental overwrite and
makes equivalent experiments idempotent.

| Artifact | Meaning |
|---|---|
| `config.json` | Fully resolved experiment parameters |
| `environment_spec.json` | Action labels, features, metrics, backend, and geometry |
| `metrics.pkl` | Full machine-readable trainer payload |
| `metrics_summary.json/csv` | Human-readable flattened summaries |
| `training_history.csv` | Episode-level online history or periodic offline-evaluation history |
| `offline_training_history.csv` | Dataset-only optimization and validation losses |
| `offline_eval_history.csv` | Periodic direct simulator evaluation during offline training |
| `checkpoint_last.pt`, `checkpoint_best.pt` | Last and validation-selected model states |
| `episodes_summary.csv`, `trajectories.csv` | Final direct simulator evaluation |
| `ope_summary.json`, `ope_bootstrap.csv` | Held-out OPE results, when enabled |

The best offline checkpoint is selected by validation loss. OPE is post-training
diagnostic output and never selects a checkpoint.

## Reproducibility Controls

`seed` controls agent initialization, sampling, and episode seeds. EpiCare also
uses `environment_seed`: it fixes the generated POMDP while `reset(seed=...)`
reproduces an episode inside that POMDP. ICU-Sepsis remains deterministic with
respect to its packaged MDP parameters and Gymnasium episode seed.

Use `dev` configs for smoke testing. Do not infer clinical validity from any
result; both upstream benchmark papers explicitly position their environments
as research benchmarks rather than clinical decision tools.
