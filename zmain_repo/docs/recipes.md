# Recipes and Research

`configs/catalog.json` is the public registry. It declares environment labels,
dataset IDs, behavior policies, trajectory counts, OPE availability, and
method recipe templates. The CLI expands each template to a concrete config
for a selected dataset, mode, seed, and typed overrides.

## Public Method IDs

CLI method IDs are deliberately stable and use hyphens. Python implementation
names use underscores internally, but are not CLI aliases. Use the IDs shown by
`python scripts/bench.py catalog` exactly:

| Environment | Method IDs |
|---|---|
| EpiCare v0 offline | `bc`, `dqn`, `iql`, `cql`, `raw-model`, `latent-model`, `reconstructive-model` |
| ICU-Sepsis v0 offline | `bc`, `dqn`, `iql`, `cql`, `raw-model/continuous-cem`, `latent-model/continuous-cem`, `reconstructive-model/continuous-cem` |
| Compatibility/online | PPO and discrete-CEM recipes remain available outside the v0 leaderboard |

For example, the latent EpiCare world model is invoked with:

```bash
python scripts/bench.py train epicare/smart/final latent-model
```

Dataset recipes live under `configs/datasets/<environment>/`. Method recipes
live under `configs/recipes/<environment>/<mode>/<tier>/`. They retain model
hyperparameters but do not contain artifact paths: the catalog supplies the
canonical `artifacts/datasets/<dataset-id>/` paths at runtime.

## Research Commands

```bash
# Explicit seeds for one recipe.
python scripts/bench.py research sweep epicare/smart/dev dqn --seeds 0,1,2 --execute

# Declarative method/dataset matrix.
python scripts/bench.py research matrix \
  configs/research/v0-offline-matrix.json --jobs 2 --execute

# Aggregate one row per task/method/seed and compute planning gains.
python scripts/bench.py research summarize

# Optuna search over the family-specific, documented CLI search space.
python scripts/bench.py research tune epicare/smart/dev dqn --trials 20

# Build a flat CSV index of all clean-layout runs.
python scripts/bench.py research collect

# Fresh evaluation for selected checkpoints.
python scripts/bench.py research reevaluate <run-a> <run-b> --episodes 1000
```

Matrices contain only explicit dataset IDs, method IDs, seeds, and mode. This
prevents an accidental expansion of hidden JSON lists. Tuning trials write
regular run artifacts and a SQLite study under `artifacts/studies/`.
