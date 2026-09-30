# Extending BenchTWM

Environments and algorithms are registries. Adding one never requires editing
the training loop, the evaluator, the CLI, or the catalog.

## Add an environment

1. **Vendor the runtime** under `third_party/<name>/` with its license and a
   `NOTICE` listing local modifications. If it is a Python package, add its
   directory to `where` in `pyproject.toml`
   (`[tool.setuptools.packages.find]`) and to the pytest `pythonpath`.
2. **Write `benchtwm/envs/<env_id>.py`**: an `EnvAdapter` subclass decorated
   with `@register_env` that sets `env_id`, `behavior_policy_ids` (stable
   integer ids), and `reference_policies`, and implements `from_config`,
   `reset`, `step`, `spec`, `transition_metrics`, `episode_metrics`, and
   `make_policy`. Observations are `float32` vectors; `info` must contain
   `action_mask`, `admissible_actions`, and a diagnostic `state_id`. Never
   expose privileged hidden state as an observation.
3. **Import the module** at the bottom of `benchtwm/envs/__init__.py`.
4. **Add `configs/envs/<env_id>.json`** (`slug`, `label`, `description`,
   `action_type`, and a `run` block with `env_kwargs`, `environment_seed`,
   `gamma`) and **`configs/datasets/<env_id>.json`** (seed, `ope_enabled`,
   `tiers`, `behaviors`).
5. **Add a tiny dataset entry** for it in `TINY_DATASETS` in
   `tests/conftest.py`.

`tests/test_envs.py` then checks the new environment's spec, reset
determinism, behavior-policy distributions, reference evaluation, dataset
schema, propensities, split separation, and collection determinism, and
`tests/test_algorithms.py` trains every algorithm on it.

Behavior policies are `Policy` subclasses (`benchtwm/policies.py`) that return
a probability vector over actions; collection samples from it and records the
exact log-propensity, so OPE works for every dataset.

## Add an algorithm

1. **Write `benchtwm/algorithms/<name>.py`**: an `OfflineAlgorithm` subclass
   decorated with `@register_algorithm` that sets `name`, `method_id`,
   `family` (`MODEL_FREE` or `MODEL_BASED`), `description`, and
   `validation_objective`, and implements `build`, `loss`, and `make_policy`.
   Override `after_update`, `batches`, `update`, `validate`, `state_dict`, or
   `tuning_space` only when needed (see `dqn.py`, `world_models.py`, and
   `dreamer/` for each case).
2. **Import the module** at the bottom of `benchtwm/algorithms/__init__.py`.
3. **Add `configs/algorithms/<name>.json`** with its default hyperparameters,
   and `configs/overrides/<env_id>/<name>.json` for environment-specific
   values.
4. **Add tiny hyperparameters** in `TINY_HYPERPARAMETERS` in
   `tests/conftest.py`.

The policy returned by `make_policy` must respect `info["action_mask"]` and
return a probability vector from `action_probabilities`; set
`ope_probability_contract` to `diagnostic` if those probabilities are not
those of a fixed stochastic policy. The contract tests train, checkpoint,
reload, and run end to end on every environment.

## Checklist

```bash
bench validate --smoke
bench catalog
bench train <dataset> <method> --dry-run
pytest
```
