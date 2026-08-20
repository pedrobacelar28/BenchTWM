# Benchmarks and Environment Contract

## BenchTWM v0 Protocol

The reproducible v0 pilot is an **offline** benchmark of continuous structured
observations and categorical actions. A task is identified by:

```text
task = dataset + environment + observation_backend + action_type + train_mode
```

The public tasks are:

```text
epicare/smart/dev|obs=state_vector|action=discrete|mode=offline
icu-sepsis/expert/dev|obs=state_vector|action=discrete|mode=offline
```

The catalog also provides one `dev` and one `final` dataset for each curated
behavior variant. EpiCare provides `random`, `smart`, `soc`, and `oracle`;
ICU-Sepsis provides `random`, `expert`, `epsilon-expert`, and `optimal`. The
catalog additionally provides `setting1/random` `dev` and `final` datasets for
`dtr-ahn-chemo` and `dtr-ghaffari-cancer`; these are additive environment
families and are not mixed into the original pilot task list. The
pilot matrix above selects one representative development task per
environment; the other catalog entries are distribution and reference-policy
variants, not additional environment families.

`state_vector` is a `float32` vector whose feature names and source dimension
are recorded in `metadata.json`. `state_id` remains in every transition for
trajectory reconstruction and diagnostics, but is not a leaderboard input.
Actions remain `int64` categorical values and every policy must respect the
admissible-action mask. Continuous actions are deferred to a future schema.

The public tracks are `direct_rl` (BC, fitted/offline DQN, IQL, CQL),
`provisional_planning` (the current raw, latent, and reconstructive model + CEM
placeholders, plus the offline DreamerV3-style agent), and the reserved
`published_planning` track. Feed-forward methods use
seeds `0, 1, 2`, batch 256, 5,000 optimization steps (1,280,000 sampled
transitions), validation every 250 steps, and a final 500-episode evaluation.
Dreamer uses its source configuration of batch 16 by sequence length 64 and
the same 5,000-step/250-step validation protocol.
All public datasets have held-out OPE available. ICU is still evaluated
directly in the simulator as the primary result, while OPE remains a separate
diagnostic evidence stream. Intermediate simulator evaluation is disabled.

The question is whether planning's return/success improvement compensates for
its training, inference, and planner-call cost relative to the best direct
baseline on the same task. Environment families are never averaged together.

## Source Material and Scope

This suite vendors only the runtime cores needed to execute the two benchmarks.
Original repositories and copied paper text live outside the package at
`../../repos_usados/` and `../../repos_usados/papers_txt/`.

| Benchmark | Source | Vendored here | Excluded here |
|---|---|---|---|
| ICU-Sepsis | Choudhary, Gupta, and Thomas (2024), `repos_usados/icu-sepsis` | Gym/Gymnasium environment and packaged MDP assets | Raw MIMIC-III data and MDP construction notebooks |
| EpiCare | Hargrave, Spaeth, and Grosenick (NeurIPS 2024), `repos_usados/EpiCare` | Environment and reference policies | Source offline-RL algorithms, notebooks, and HDF5 data files |

The source papers define benchmarks. This repository defines an integration
layer and local algorithms; it does not reproduce every source-paper result.

## ICU-Sepsis

ICU-Sepsis is a finite tabular MDP constructed from approximately 17,000 sepsis
patient trajectories in MIMIC-III according to the source paper. Its packaged
environment has 716 discrete states and 25 actions. The actions form a 5 by 5
grid: fluids level by vasopressor level. Package assets include transition and
reward matrices, admissible action sets, state-cluster centers, and SOFA scores.

| Backend | Agent input | Shape in the packaged default |
|---|---|---:|
| `discrete_id` | One-hot tabular state | 716 |
| `state_vector` | Continuous state-cluster center | 47 |

Only state-action pairs sufficiently represented in the source construction are
admissible. The adapter exposes this as an action mask; policies and value
functions use the mask rather than inventing unsupported transitions.

The environment metric is `sofa_score`. The current integration maps a positive
total return to `success` and retains `survived` as an ICU-only alias. Action
distance is Manhattan distance on the 5 by 5 treatment grid.

## EpiCare

EpiCare models longitudinal treatment as a POMDP. A hidden disease state evolves
under treatment while the agent observes symptoms. The source paper emphasizes
partial observability, heterogeneous treatment effects, short episodes, safety
events, and simulated clinical-trial data collection.

| Component | Default |
|---|---:|
| Latent disease states | 16 plus remission |
| Treatments | 16 categorical actions |
| Observed symptoms | 8 continuous values in `[0, 1]` |
| Maximum visits | 8 |
| Patient modifiers | enabled |
| Environment seed | 1 in provided configs |

Only `state_vector` is allowed for EpiCare. `state_id` is retained in datasets
and exports strictly as a diagnostic hidden-state identifier; it is never an
agent observation. All 16 treatments are admissible in the current adapter.

Transition metrics are `symptom_burden`, `remission`, `adverse_event`, and
`time_to_remission`. At episode level, `success` means remission. The evaluator
also reports remission/adverse-event rates and time to remission among remitted
episodes. EpiCare action distance is categorical: zero for equal treatments and
one otherwise.

## DTR AhnChemo and GhaffariCancer

The DTR additions use the ODE implementations from the pinned selective
runtime in `third_party/dtr/`. They are integrated through the same adapter
contract and use only `state_vector` observations.

| Environment | Public features | Action contract | Hidden state excluded from the agent |
|---|---|---|---|
| `dtr-ahn-chemo` | `(tumor, immune, drug)` as `float32[3]` | Five ordinal dose bins; dose `i` maps to `i/4` in the native `[0,1]` range | Normal-cell population and all other full-state fields |
| `dtr-ghaffari-cancer` | Seven `float32` features `(T_p,N_p,L_p,C,T_s,N_s,L_s)` after natural log and lower clipping at 1 | Row-major 5×5 grid: radiotherapy `[0, 2.5, 5, 7.5, 10]`, chemotherapy `[0, 2, 4, 6, 8]` | Metastasis timing, drug concentration, radiation-exposure auxiliaries, and ODE parameters |

All actions are valid in the initial public settings. AhnChemo success means
the episode reaches horizon truncation without the simulator's death
termination. GhaffariCancer success means the simulator reaches its positive
tumor-control termination. The adapters report tumor burden, doses, terminal
outcomes, return, and duration without adding clinical normalizations.

## Shared Adapter Interface

Every environment is selected through `make_env(config, obs_backend, seed)` in
`src.envs.factory`. An adapter declares an `EnvSpec` and implements Gymnasium
`reset`/`step` plus this semantic surface:

```python
env.spec                 # dimensions, labels, features, metrics, action geometry
env.action_label(action) # readable treatment name
env.action_distance(a, b)
env.transition_metrics(info)
env.episode_metrics(total_reward=..., final_info=..., transition_metrics=...)
```

`EnvSpec` is persisted with every final evaluation. Generic tooling therefore
does not need hard-coded knowledge of SOFA, symptoms, or action dimensions.

## Comparability Rules

`return`, `success`, `length`, and action-change statistics have common field
names, but not automatically common clinical meaning. ICU success is survival;
EpiCare success is remission. SOFA and symptom burden have distinct scales, and
raw action distance has different ranges across the grid and categorical action
spaces. Reports must preserve `env_id` and compare clinical outcomes within an
environment unless a justified normalization is defined.

The v0 report writes one row per task/method/seed and then summarizes mean and
standard deviation within that task. Planning gains are
`planning_return - best_direct_return` and
`planning_success - best_direct_success`; OPE, direct evaluation, and planner
cost remain separate evidence streams.
