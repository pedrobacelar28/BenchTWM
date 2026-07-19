# Extending the Suite

This checklist is the compatibility boundary for a third environment or a new algorithm. Keep existing adapters, NPZ files, configs, and output artifacts valid.

## Add an Environment

1. Vendor only the runtime core under `third_party/<environment>/`, preserving its license and a `NOTICE` describing local changes.
2. Add `src/envs/<environment>_adapter.py` implementing `EnvAdapter`.
3. Create an `EnvSpec` with `env_id`, dimensions, supported observation backends, action labels, feature names, metric names, primary metric, and action geometry.
4. Implement `reset`, `step`, `action_label`, `action_distance`, `transition_metrics`, and `episode_metrics`.
5. Register a stable alias and construction path in `src/envs/factory.py`.
6. Add a behavior-policy adapter in `src/offline/generate_dataset.py`. It must sample the action from a known distribution and write its exact `behavior_logprob`.
7. Add reference-policy support in `src/envs/reference.py` when the source benchmark defines baselines.
8. Add dataset, online, and offline dev configs. Use only observation backends genuinely exposed to the agent.
9. Add tests for reset reproducibility, schema validity, split separation, finite behavior propensities, all algorithm smoke paths, and generic reports.

The adapter must never place privileged latent state into the observation passed to an agent. A diagnostic ID may be exported only when clearly documented as privileged analysis data.

## Add an Algorithm

1. Create `src/algorithms/<name>.py` with `train(config, output_dir)` and `load_policy(config, checkpoint)`.
2. Return `metrics`, `checkpoint_last`, and `checkpoint_best` from training.
3. Ensure the loaded policy implements `act(obs, info, deterministic=True)` and `action_probabilities(obs, info, deterministic=True)`.
4. Respect `obs_backend` and `action_mask`; do not import an environment adapter directly from an algorithm.
5. Register online training/loading in `src/algorithms/registry.py`.
6. Add a fixed-data trainer to `src/offline/trainers.py` and its registry only if offline learning is supported.
7. Emit meaningful optimization/validation metrics and keep checkpoints serializable with `torch.save`.
8. Add dev configs for every supported environment/backend combination.
9. Add unit tests for masking and probabilities, smoke tests for online/offline modes, and OPE integration when supported.

## Config Requirements

All experiment configs should state the following explicitly for new work:

```json
{
  "env_id": "epicare",
  "environment_seed": 1,
  "env_kwargs": {},
  "obs_backend": "state_vector",
  "algo": "dqn",
  "seed": [0]
}
```

Omit `env_id` only when intentionally using legacy ICU-Sepsis defaults. Offline configs additionally need `dataset_path`, `val_dataset_path`, gradient-step budget, and validation/evaluation intervals. OPE needs a held-out `ope_dataset_path` and `ope_enabled: true`.

## Required Regression Checks

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python scripts/bench.py list
python scripts/bench.py suite --env <environment> --mode offline --tier dev
python scripts/bench.py dataset --config <dataset-config>
python scripts/bench.py run --config <offline-config>
```

For a new environment, inspect `environment_spec.json`, `episodes_summary.csv`, `trajectories.csv`, and `ope_summary.json` rather than relying only on successful process exit.
