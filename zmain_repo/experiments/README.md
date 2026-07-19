# Experiment Configurations

Existing paths are retained for compatibility:

```text
experiments/datasets/{dev,final}/
experiments/offline/{dev,final}/
experiments/online/{dev,final}/
```

The environment is declared by `env_id`. Missing `env_id` means
`icu_sepsis`, which keeps historical ICU configs valid. EpiCare configs use
the `epicare_` prefix and set `env_id: "epicare"` explicitly.

Use the catalog instead of memorizing names:

```bash
python scripts/bench.py list
python scripts/bench.py list --env icu_sepsis --kind offline --tier dev
python scripts/bench.py list --env epicare --kind online --tier final
```

`dev` configs are small enough for functional validation but can still be
computationally substantial. `final` configs are research budgets, not smoke
tests. JSON list-valued fields form a Cartesian sweep, except for explicitly
fixed list fields handled by `src.utils.json_handling`.

Each offline config names `dataset_path`, `val_dataset_path`, and optionally
`ope_dataset_path`. Dataset configs are the source of truth for how these files
are generated.
