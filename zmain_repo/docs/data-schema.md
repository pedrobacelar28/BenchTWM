# Offline Data Schema

## Source Data Versus Local Datasets

The meaning of each curated behavior-policy variant and of the `dev`/`final`
tiers is defined in [Curated offline datasets](datasets.md). This page
specifies the arrays and metadata stored inside each generated dataset.

The upstream ICU-Sepsis package ships a parameterized MDP, not raw patient
records. Its assets include transition/reward matrices, state cluster centers,
SOFA scores, and admissible actions. The source paper traces those parameters
back to MIMIC-III; raw MIMIC-III data is not present in this repository.

The upstream EpiCare repository can generate HDF5 files containing
`observations`, `actions`, `rewards`, `terminals`, and `next_observations` from
simulated trials. This suite deliberately does not depend on those HDF5 files.
It generates an extended, versioned NPZ schema so both environments share one
offline interface.

## Split Layout

```text
<dataset-root>/
  train.npz
  val.npz
  ope_test.npz            # only when OPE is enabled
  metadata.json
  episodes_summary.csv
```

Episode identifiers and seed ranges do not overlap across train, validation,
and OPE splits. Training reads only train/validation. OPE reads only the held-out
`ope_test.npz`.

## Required Transition Arrays

Every NPZ file contains these arrays, aligned on its first axis `N`:

| Array | Type | Shape | Meaning |
|---|---|---|---|
| `observations`, `next_observations` | `float32` | `(N, obs_dim)` | Default model observation representation |
| `state_vectors`, `next_state_vectors` | `float32` | `(N, vector_dim)` | Continuous feature backend when available |
| `actions`, `next_actions` | `int64` | `(N,)` | Logged action and next action (`-1` at episode end) |
| `rewards` | `float32` | `(N,)` | Environment reward |
| `terminals` | `bool` | `(N,)` | Terminated or truncated transition |
| `state_ids`, `next_state_ids` | `int64` | `(N,)` | Discrete state or diagnostic state identifier |
| `action_masks`, `next_action_masks` | `float32` | `(N, num_actions)` | Valid-action indicators |
| `behavior_logprobs` | `float32` | `(N,)` | Log probability of the logged action |
| `behavior_policy_ids` | `int64` | `(N,)` | Stable behavior-policy identifier |
| `episode_ids`, `timesteps` | `int64` | `(N,)` | Trajectory reconstruction keys |
| `env_metrics`, `next_env_metrics` | `float32` | `(N, num_metrics)` | Metrics ordered by `metadata.environment.metric_names` |

ICU datasets also preserve `sofa_scores` and `next_sofa_scores` for backward
compatibility. These are optional aliases, not generalized schema requirements.

For v0 recipes, `state_vectors`/`next_state_vectors` are the default continuous
agent inputs (`float32`). The `discrete_id` one-hot view remains an explicit
compatibility/ablation backend. Algorithms select the exact view through
`obs_backend`; every view is paired with the same action masks.

## How Models Consume the Data

| Consumer | Inputs from the schema | Supervision or use |
|---|---|---|
| BC | Observation, action mask, action | Masked action cross-entropy |
| Offline DQN/FQI | Observation, action mask, action, reward, next observation, terminal | One-step masked TD target |
| IQL | DQN fields plus next-state value | Expectile value and advantage-weighted behavior cloning |
| CQL | DQN fields plus next-state Q | TD target plus admissible-action conservative regularization |
| Raw world model | Observation, action, next observation, reward, terminal | Next observation, reward, continuation prediction |
| Latent/reconstructive world model | Sequence windows over the same fields | Multi-step latent/decoded dynamics and rollout validation |
| OPE | Ordered trajectory, logged action, reward, behavior log probability | Importance ratios against target policy probabilities |

`state_ids` are necessary for ICU tabular diagnostics and sequence bookkeeping.
For EpiCare, they contain the privileged disease identifier but are not passed to
the MLP agents or treated as an input feature. For DTR environments, they are
deterministic fingerprints of the visible vector only; they are not a tabular
encoding and never expose the hidden simulator state.

## Behavior Policies and Propensities

The generator samples an action and records its exact sampled log propensity.
This is required for logged PPO and OPE.

| Environment | Behavior policies |
|---|---|
| ICU-Sepsis | `random_admissible`, `expert`, `optimal`, `epsilon_expert`, `policy_checkpoint` |
| EpiCare | `random`, `smart`, `soc`, `oracle` |
| DTR AhnChemo | `random` (`log(1/5)` on every public action) |
| DTR GhaffariCancer | `random` (`log(1/25)` on every public action) |

EpiCare SMART is the default provided dataset policy. It exposes a nonuniform
treatment distribution and is sampled directly, so `behavior_logprobs` reflects
the collection probability. SoC and Oracle can be deterministic, which narrows
support and should be considered before using OPE.

## Metadata and Inspection

`metadata.json` uses `schema_version: 2` and records the source config,
environment spec, policy-ID table, and split statistics. The environment spec
gives the semantic order of `env_metrics`; consumers must not assume that a
column position means SOFA or symptom burden.

```bash
python scripts/bench.py inspect dataset epicare/smart/dev
```

The loader rejects missing required arrays, empty datasets, unsupported
backends, and missing state vectors when requested. This validates the research
data schema; it is not a clinical data quality guarantee.
