# Configuration

All configuration lives in `configs/` as small JSON files:

```text
configs/
  protocol.json                      offline protocol shared by every run
  algorithms/<algo>.json             default hyperparameters of one algorithm
  envs/<env_id>.json                 environment identity and run settings
  datasets/<env_id>.json             behaviors x tiers of one environment
  overrides/<env_id>/<algo>.json     optional deviations for one pair
  research/*.json                    declarative experiment matrices
```

## How a run config is resolved

`bench train <dataset> <method>` builds the run config in this order, each
layer overriding the previous one:

1. `protocol.json`
2. `algorithms/<algo>.json`
3. the `run` block of `envs/<env_id>.json` (`env_kwargs`, `environment_seed`, `gamma`)
4. `overrides/<env_id>/<algo>.json`, if present
5. identity and paths (`algo`, `method_id`, `dataset_id`, `seed`, dataset files)
6. `--set KEY=JSON` options from the command line (dotted keys reach nested objects)

In any of the files in steps 1-4, a `tiers` object holds values that apply
only to one tier:

```json
{
  "hidden_dim": 256,
  "tiers": {"dev": {"learning_rate": 0.0001}, "final": {"learning_rate": 0.001}}
}
```

Inspect the result without running anything:

```bash
bench train icu-sepsis/expert/dev raw-model --dry-run --json
bench train epicare/smart/dev dqn --set offline_train_steps=500 --set env_kwargs.patient_modifiers=false --dry-run
```

The resolved config is saved as `config.json` in the run directory, and its
hash names the directory, so changing any value creates a new run.

## Datasets

`datasets/<env_id>.json` declares the collection seed, whether OPE splits are
collected, the episode counts per tier, and the behaviors with their policy
options (for example `epsilon` for `epsilon_expert`). The catalog is every
behavior times every tier. Every registered algorithm whose `action_types`
include the environment's `action_type` is offered on every dataset.

## Research matrices and tuning

```bash
bench research matrix configs/research/dev-matrix.json            # plan
bench research matrix configs/research/dev-matrix.json --execute  # run
bench research sweep epicare/smart/dev latent-model --seeds 0,1,2 --execute
bench research tune epicare/smart/dev dqn --trials 20
```

A matrix lists `datasets`, optional `methods` (default: all compatible), and
`seeds`, or a list of such blocks under `jobs`. Tuning searches each
algorithm's `tuning_space` and minimizes the offline validation loss, never a
simulator return.

`bench validate` checks the files for consistency (unregistered environments,
missing algorithm defaults, undeclared behavior policies, overrides for
unknown pairs).
