# Data Schema (version 3)

Every split (`train.npz`, `val.npz`, `ope_test.npz`) stores these arrays,
aligned on the first axis `N` (one row per logged transition):

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `observations`, `next_observations` | float32 | `(N, obs_dim)` | Agent observation before and after the step |
| `actions` | int64 | `(N,)` | Logged action |
| `next_actions` | int64 | `(N,)` | Next logged action in the episode, `-1` at the end |
| `rewards` | float32 | `(N,)` | Environment reward |
| `terminals` | bool | `(N,)` | Episode ended (terminated or truncated) |
| `action_masks`, `next_action_masks` | float32 | `(N, num_actions)` | Admissible actions (1) before and after the step |
| `behavior_logprobs` | float32 | `(N,)` | Exact log-probability of the logged action under the behavior policy |
| `behavior_policy_ids` | int64 | `(N,)` | Stable behavior-policy id (declared by the environment) |
| `episode_ids`, `timesteps` | int64 | `(N,)` | Trajectory reconstruction keys |
| `state_ids`, `next_state_ids` | int64 | `(N,)` | Simulator state ids; diagnostics only, never agent inputs |
| `env_metrics`, `next_env_metrics` | float32 | `(N, num_metrics)` | Per-step metrics in the order of `metadata.environment.metric_names` |

`metadata.json` records the collection config, the environment spec (feature
and action names, metric names), `observation.feature_names`, the behavior
policy and its id, per-split statistics, and SHA-256 hashes of the split
files. Consumers must look metric columns up by name, never by position.

Version 3 dropped the version-2 duplicates `state_vectors`,
`next_state_vectors`, `sofa_scores`, and `next_sofa_scores` (they equalled
`observations` and the `sofa_score` metric column). Version-2 datasets still
load: the loader reads only the arrays above.

## How methods consume the data

| Consumer | Uses |
|---|---|
| BC | observation, action mask, action |
| DQN, CQL | observation, action, reward, next observation, next mask, terminal |
| IQL | the DQN fields; its value network bootstraps from the next observation |
| CEM world models | the DQN fields, plus length-k windows for rollout diagnostics |
| Dreamer | whole episodes, re-aligned to reward-on-arrival records in memory |
| OPE | ordered trajectories, logged actions, rewards, behavior log-probabilities |

Dreamer's loader (`benchtwm/data/sequences.py`) adds `is_first`, `is_last`,
`is_terminal`, and a validity mask at training time and carries the recurrent
state between consecutive chunks of an episode; the stored schema is unchanged.
