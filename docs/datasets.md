# Datasets

A dataset is a fixed collection of simulator trajectories logged by one
behavior policy. Its id is `<environment>/<behavior>/<tier>`, for example
`icu-sepsis/expert/dev`. The catalog is every behavior of
`configs/datasets/<env>.json` times every tier:

```bash
bench dataset list
bench dataset show epicare/smart/final
bench dataset build epicare/smart/dev
```

Every dataset has `train.npz`, `val.npz`, and a held-out `ope_test.npz`, plus
`metadata.json` and `episodes_summary.csv`. Episode ids and reset seeds never
overlap across splits. Training reads only train and validation; OPE reads
only `ope_test.npz`.

## Tiers

`dev` and `final` differ only in size; the environment and the behavior policy
are the same. Use `dev` for development and `final` for reported results.

| Environment | Tier | Train | Validation | OPE |
|---|---|---:|---:|---:|
| EpiCare | `dev` | 1,000 | 200 | 500 |
| EpiCare | `final` | 100,000 | 10,000 | 20,000 |
| ICU-Sepsis | `dev` | 5,000 | 500 | 500 |
| ICU-Sepsis | `final` | 100,000 | 10,000 | 20,000 |

(episodes per split)

## Behaviors

| Dataset behavior | Policy | Role |
|---|---|---|
| `epicare/random` | uniform over 16 treatments | broad coverage, lower reference |
| `epicare/smart` | ClinicalTrial, nonuniform stochastic | main EpiCare dataset |
| `epicare/soc` | Standard of Care, adaptive and deterministic | narrow-support stress test |
| `epicare/oracle` | greedy on the hidden disease | privileged reference; report separately |
| `icu-sepsis/random` | uniform over admissible actions | broad coverage, lower reference |
| `icu-sepsis/expert` | packaged clinician frequencies | main ICU dataset |
| `icu-sepsis/epsilon-expert` | expert with 20% uniform exploration | expert data with coverage |
| `icu-sepsis/optimal` | value-iteration optimum | simulator reference; narrow support |

Deterministic behaviors (`soc`, `oracle`, `optimal`) have narrow action
support, which makes importance-sampling OPE unreliable for most targets; the
OPE reliability diagnostics report this.

## Reproducibility

Collection is deterministic given the dataset config: two builds produce
identical arrays. (The vendored EpiCare seeds the multivariate-normal CDF used
by its reference policies; upstream it varied between processes.)

A dataset can also be collected with a trained checkpoint as the behavior
policy by passing `behavior_policy: "checkpoint"` and
`behavior_checkpoint: <run dir>` to `benchtwm.data.collect.collect_dataset`;
such datasets are not part of the public catalog.
