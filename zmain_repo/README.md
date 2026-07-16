# ICU-Sepsis World-Model Experiments

This repository contains the code used to train, evaluate, and analyze the
offline DQN/PPO baselines and CEM-based world-model agents from the
ICU-Sepsis paper.

The repository is intentionally source-focused. Generated datasets, checkpoints,
Optuna databases, trajectory CSVs, and plot/report outputs are not included
because they are large and can be regenerated from the provided configs.

## Contents

- `src/`: training code, models, planners, environment wrappers, and utilities.
- `experiments/`: JSON configs for dataset generation and paper offline runs,
  plus optional online smoke configs for retained methods.
- `scripts/`: launchers for sweeps, Optuna studies, and result collection.
- `analysis/`: result aggregation, plotting, reporting, and qualitative-case
  analysis.
- `tests/`: unit and smoke tests.
- `third_party/icu-sepsis/`: vendored ICU-Sepsis benchmark packages used by the
  experiments.

## Setup

Create and activate a Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

The ICU-Sepsis benchmark packages are included under `third_party/`; the code
also supports a sibling `icu-sepsis/` checkout if one exists.

## Quickstart

Run one indexed config from a JSON sweep:

```bash
python src/run_config.py experiments/offline/dev/dqn.json 0
```

Run all pending configs from one or more JSON files:

```bash
python scripts/run_sweep.py -j experiments/offline/dev/dqn.json
```

Generate an offline dataset:

```bash
python src/offline/generate_dataset.py experiments/datasets/dev/expert_100000.json
```

Train one offline method on a generated dataset:

```bash
python src/run_config.py experiments/offline/dev/raw_model_discrete_cem.json 0
```

Run the test suite:

```bash
pytest -q
```

## Method Names

The paper's model-based CEM variants use descriptive names throughout the
public code:

- `raw_model`: predicts the next observation directly in the observation space.
- `latent_model`: predicts an EMA-target latent representation.
- `reconstructive_model`: predicts in latent space and reconstructs the next
  observation.

Each world-model family has discrete and continuous CEM configs, matching the
OS-WM, LP-WM, and RL-WM variants described in the paper. The retained
model-free baselines are `dqn` and `ppo`.

For offline configs, `offline_train_steps` is the gradient-update budget and
`offline_batch_size` is the number of sampled transitions per update.

See `docs/README.md` for the full layout and output schema.
