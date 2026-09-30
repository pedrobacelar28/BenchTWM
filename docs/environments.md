# Environments

Only the runtime cores of the source benchmarks are vendored under
`third_party/`; their local modifications are listed in each `NOTICE`. The
original repositories and paper texts used for auditing are kept, untracked,
under `repos_usados/`.

| Benchmark | Source | Vendored | Not included |
|---|---|---|---|
| ICU-Sepsis | Choudhary, Gupta, and Thomas (2024) | Gymnasium environment and packaged MDP assets | Raw MIMIC-III data, MDP-construction helpers |
| EpiCare | Hargrave, Spaeth, and Grosenick (NeurIPS 2024) | Environment and reference policies | Source offline-RL algorithms, notebooks, HDF5 datasets |

## ICU-Sepsis (`icu_sepsis`)

A finite MDP estimated from about 17,000 MIMIC-III sepsis trajectories. It has
716 states and 25 actions; an episode lasts at most 500 steps and yields a
reward of +1 when the patient survives.

- **Observation**: the 47-feature centroid of the current state's cluster,
  with the AI Clinician column names (`gender`, `mechvent`, `age`, `HR`,
  `SysBP`, ..., `output_4hourly`; see `FEATURE_NAMES` in
  `benchtwm/envs/icu_sepsis.py`). Values are the upstream z-scored or
  log-z-scored features; because the MDP is tabular there are only 716
  distinct vectors, and binary columns such as `gender` become cluster
  averages.
- **Actions**: a 5 x 5 grid of IV-fluid bin x vasopressor bin
  (`fluid_bin = a // 5`, `vaso_bin = a % 5`). Only actions admissible in the
  current state are unmasked.
- **Metrics**: `sofa_score` per step; `success` means survival. Action distance
  is the Manhattan distance on the grid; a change of more than one bin on
  either axis is abrupt.
- **Behavior policies**: `random_admissible`, `expert` (packaged clinician
  frequencies), `epsilon_expert` (expert mixed with uniform exploration), and
  `optimal` (value iteration over the packaged MDP).

## EpiCare (`epicare`)

A procedurally generated POMDP of longitudinal treatment: a hidden disease
evolves under treatment and the agent observes symptoms.

| Component | Default |
|---|---:|
| Hidden disease states | 16 plus remission |
| Treatments (actions) | 16, all admissible |
| Observed symptoms | 8 values in `[0, 1]`, rounded to 0.1 |
| Maximum visits per episode | 8 |
| Patient modifiers | enabled |
| Environment seed | 1 |

- **Observation**: symptom intensities only. The disease id is exported as the
  diagnostic `state_id` and is never an agent input.
- **Metrics**: `symptom_burden`, `remission`, `adverse_event`,
  `time_to_remission`; `success` means remission.
- **Behavior policies**: `random`, `smart` (EpiCare's ClinicalTrial policy),
  `soc` (Standard of Care, adapts within an episode), and `oracle` (reads the
  hidden disease; a privileged reference, not a deployable policy).

## The adapter contract

Every environment is an `EnvAdapter` subclass (`benchtwm/envs/base.py`)
registered with `@register_env`:

```python
env = make_env({"env_id": "icu_sepsis", "env_kwargs": {}}, seed=0)
obs, info = env.reset(seed=0)          # float32[obs_dim], info dict
obs, reward, terminated, truncated, info = env.step(action)
info["action_mask"]                    # float32[num_actions], 1 = admissible
info["state_id"]                       # diagnostic only
env.spec                               # EnvSpec: dims, feature/action names, metrics, geometry
env.make_policy("expert", seed=0)      # a declared behavior/reference policy
env.transition_metrics(info)           # per-step metrics, ordered as spec.metric_names
env.episode_metrics(...)               # episode outcomes, including "success"
env.action_distance(a, b); env.export_columns(info, action)
```

Behavior policies implement the same `Policy` contract as learned agents
(`benchtwm/policies.py`): a probability vector over actions for each
`(obs, info)`. Dataset collection samples from it and logs the exact
propensity; `bench baselines` evaluates it directly in the simulator.

## Comparability

`return`, `success`, `length`, and the action-change statistics share names
across environments but not meanings: ICU success is survival, EpiCare success
is remission, and SOFA and symptom burden have different scales. Summaries
therefore compare methods within a dataset and never average environments.
