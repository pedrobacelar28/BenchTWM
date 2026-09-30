# BenchTWM

BenchTWM is an offline benchmark for reinforcement learning and world models
on structured decision data. It asks whether **model-based methods (learned
world models with planning) beat model-free offline RL** when every policy
must be learned from logged data alone, as it would be in a hospital or a
company that cannot experiment on live users.

Every run follows the same protocol: train on a fixed logged dataset, select
the checkpoint with an offline validation loss, then measure the deployed
policy twice, by direct simulator evaluation and by held-out off-policy
evaluation (OPE). The simulator is never used for training or model selection.

It is research software, not a clinical decision-support system.

## Contents

| Environments | Observation | Actions |
|---|---|---|
| `icu_sepsis`: ICU-Sepsis MDP estimated from MIMIC-III | 47 named clinical features | 25 (5x5 fluid/vasopressor grid, admissibility masks) |
| `epicare`: longitudinal-treatment POMDP | 8 symptom intensities | 16 treatments |

| Family | Methods |
|---|---|
| model-free | `bc`, `dqn`, `iql`, `cql` |
| model-based | `raw-model`, `latent-model`, `reconstructive-model` (world model + CEM planning), `dreamer` (DreamerV3-style) |

Each environment has four behavior policies and two size tiers (`dev` for
quick iteration, `final` for reported results), giving 16 datasets such as
`icu-sepsis/expert/dev` or `epicare/smart/final`. Every method runs on every
dataset.

## Install

```bash
pip install -r requirements.txt    # pinned reference versions
pip install -e . --no-deps         # benchtwm + the vendored ICU-Sepsis and EpiCare cores
```

This installs the `bench` command (also available as `python -m benchtwm`).

## Quick start

```bash
bench doctor                                  # dependencies, GPUs, registered components
bench catalog                                 # datasets and methods
bench dataset build epicare/smart/dev         # collect train/val/OPE splits
bench train epicare/smart/dev dqn --dry-run   # show the resolved plan
bench train epicare/smart/dev dqn             # train, evaluate, OPE
bench suite epicare/smart/dev --seeds 0,1,2 --execute   # every method, three seeds
bench research summarize                      # per-method statistics and planning gain
```

`train` never builds data implicitly; a missing dataset prints the
`dataset build` command to run.

## Repository layout

```text
benchtwm/
  envs/          environment registry and one module per environment
  algorithms/    algorithm registry and one module per method
  data/          dataset schema, loading, and collection
  training.py    the single offline training loop
  evaluation.py  direct simulator evaluation
  ope.py         importance-sampling OPE
  runner.py      one run: train -> evaluate -> OPE -> artifacts
  config.py      layered configs and the derived catalog
  operations/    artifacts, local queue, reports, summaries
  cli/           the `bench` command
configs/         protocol, per-algorithm defaults, environments, datasets, overrides
tests/           contract tests over every environment and algorithm, plus unit tests
third_party/     vendored ICU-Sepsis and EpiCare runtimes (see THIRD_PARTY_NOTICES.md)
docs/            documentation
artifacts/       generated datasets, runs, queues, reports (git-ignored)
```

## Adding environments and algorithms

An environment is one module in `benchtwm/envs/` plus `configs/envs/<id>.json`
and `configs/datasets/<id>.json`. An algorithm is one module in
`benchtwm/algorithms/` plus `configs/algorithms/<name>.json`. The contract
tests pick up new registrations automatically. See
[docs/extending.md](docs/extending.md).

## Documentation

- [Environments](docs/environments.md) and [datasets](docs/datasets.md)
- [Data schema](docs/data-schema.md)
- [Algorithms](docs/algorithms.md)
- [Evaluation protocol](docs/evaluation.md)
- [Configuration](docs/configuration.md)
- [Command line and artifacts](docs/operations.md)
- [Extending the benchmark](docs/extending.md)
- [DreamerV3 fidelity audit](docs/dreamer-fidelity.md)
- [Technical specification](docs/paper/main.tex)
- [Candidate structured environments and datasets](CATALOGO_AMBIENTES.md) (research notes, Portuguese)

## Tests

```bash
pytest
```
