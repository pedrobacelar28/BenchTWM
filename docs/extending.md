# Extending BenchTWM

The extension boundary is explicit: adapters define environments, algorithm
registrations define learning, and `configs/catalog.json` defines what users
can run. Do not add a command-specific code path for a new benchmark.

## Add an Environment

1. Vendor only the runtime core under `third_party/<environment>/`, preserving
   its upstream license and a local `NOTICE` describing modifications.
2. Implement `src/envs/<environment>_adapter.py` as `EnvAdapter` and declare a
   complete `EnvSpec`: observation backends, action labels/geometry, features,
   named metrics, and primary metric.
3. Register a stable `env_id` and constructor in `src/envs/factory.py`.
4. Add behavior-policy collection support in `src/offline/generate_dataset.py`.
   It must sample actions from a known distribution and record exact
   `behavior_logprobs`.
5. Add reference-policy support in `src/envs/reference.py` only when the
   environment genuinely declares baseline policies.
6. Add dataset recipes under `configs/datasets/<environment>/`, method recipes
   under `configs/recipes/<environment>/`, and public IDs in `configs/catalog.json`.
7. Run `python scripts/bench.py validate --smoke`, then add reset,
   dataset-schema, propensity, offline/online, OPE, and report tests.

An adapter must never expose privileged latent state as an agent observation.
It may export diagnostic IDs only as explicitly documented analysis fields.

## Add an Algorithm

1. Implement `train(config, output_dir)` and `load_policy(config, checkpoint)`
   under `src/algorithms/`.
2. Return `metrics`, `checkpoint_last`, and `checkpoint_best`. Checkpoints must
   remain serializable with `torch.save`.
3. The loaded policy must implement `act(obs, info, deterministic=True)` and
   `action_probabilities(obs, info, deterministic=True)`.
4. Respect `obs_backend` and `action_mask`; algorithms may not import a concrete
   environment adapter.
5. Register online training/loading in `src/algorithms/registry.py`. Register a
   fixed-data trainer in `src/offline/trainers.py` when offline learning is
   supported.
6. Add a method recipe for each compatible environment/backend/tier and a
   catalog method ID. Keep CEM geometry variants separate public methods.
7. Add a dev smoke test and include OPE coverage whenever the method provides
   action probabilities.

## Recipe Contract

Every canonical recipe states the generic environment contract explicitly:

```json
{
  "schema_version": 1,
  "env_id": "epicare",
  "environment_seed": 1,
  "env_kwargs": {},
  "obs_backend": "state_vector",
  "algo": "dqn",
  "seed": [0]
}
```

The catalog supplies dataset paths, the chosen scalar seed, run identity, and
artifact location at runtime. Do not embed output paths in recipes. Validate a
new catalog entry before execution:

```bash
python scripts/bench.py validate
python scripts/bench.py dataset show <dataset-id>
python scripts/bench.py train <dataset-id> <method-id> --dry-run
```
