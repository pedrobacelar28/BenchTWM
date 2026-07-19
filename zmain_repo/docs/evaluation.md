# Training, Direct Evaluation, and OPE

## Three Different Measurement Regimes

The suite deliberately separates optimization, direct simulator evaluation, and off-policy evaluation. They answer different questions.

| Regime | Data or environment used | Primary artifacts | Interpretation |
|---|---|---|---|
| Offline optimization | `train.npz` and `val.npz` only | `offline_training_history.csv` | Does the objective fit logged transitions? |
| Direct evaluation | Fresh episodes from the benchmark environment | `eval_summary.json`, episodes, trajectories | What does the policy do in this simulator? |
| OPE | Held-out `ope_test.npz` only | `ope_summary.json`, `ope_bootstrap.csv` | What return is predicted from logged behavior data? |

The phrase `offline_eval_history.csv` is historical naming: that file is a periodic **direct simulator evaluation** performed while an offline model is training. It is not an OPE estimate.

## Canonical Direct-Evaluation Metrics

Every final evaluation writes the following common fields:

- `mean_return` and episode `return`;
- `mean_success` and episode `success`;
- `mean_length` and episode `length`;
- `mean_action_stability_l1` and `mean_abrupt_jump_rate`;
- `env_id`, action indices/labels, rewards, termination, named features, and named transition metrics in trajectory exports.

The evaluator also aggregates every metric declared by `EnvSpec` as `mean_<metric>`. ICU preserves `mean_survival` and SOFA aliases. EpiCare emits symptom burden, remission, adverse-event, and time-to-remission summaries; its episode-level rates are `mean_remission_rate` and `mean_adverse_event_rate`.

Online training histories now store `success_rate` as the canonical episode result and retain `survival_rate` as an identical compatibility alias. They also store `env_initial_*`, `env_mean_*`, `env_final_*`, and `env_outcome_*` columns where the environment declares metrics. Action stability uses the environment's own distance function.

## Offline Training Metrics

The loss files are algorithm-dependent and should not be numerically compared across algorithm families.

| Family | Core train/validation metrics |
|---|---|
| DQN | TD loss |
| Logged PPO | Total, policy, value, behavior-cloning, entropy losses |
| Raw model | Dynamics, prediction, reward, continuation losses |
| Latent model | Latent dynamics, prediction, reward, continuation losses |
| Reconstructive model | Latent/dynamics, reconstruction, prediction, reward, continuation losses |

World-model runs additionally report rollout validation errors over multiple horizons. Direct return and success curves are separate simulator outputs, not validation-loss substitutes.

## Off-Policy Evaluation

OPE is enabled only when both `ope_enabled: true` and `ope_dataset_path` are present. The generator must have recorded finite `behavior_logprobs` for every transition. The target policy supplies `action_probabilities`; for logged action `a_t`, the per-step ratio is

```text
rho_t = pi_target(a_t | o_t) / pi_behavior(a_t | o_t)
```

The implementation estimates discounted **return**, not remission, SOFA, or adverse-event rates counterfactually.

| Estimator | Definition in this suite | Tradeoff |
|---|---|---|
| IS | Full trajectory likelihood ratio times trajectory return | Direct but high variance |
| WIS | Trajectory-weight-normalized IS | Lower variance, finite-sample bias |
| PDIS | Cumulative ratio applied at every reward step | Often less variable than trajectory IS |
| WPDIS | Per-step normalized PDIS | Stabilized step-wise weighting |

Bootstrap resampling is trajectory-wise and seeded. `ope_summary.json` includes 95 percent bootstrap intervals, number of episodes/transitions, target-action support rate, zero-weight trajectory rate, trajectory ESS, and maximum log weight.

## Interpreting OPE

For a deterministic target policy, a logged transition has target probability zero unless the logged action matches the deployed action. A long trajectory becomes zero weighted after its first mismatch. Thus support rate and ESS are validity diagnostics, not cosmetic statistics.

The current EpiCare dev DQN example demonstrates this issue: its SMART held-out data has many target-action mismatches, so most full trajectories have zero weight and the four estimators disagree. Such a result should be reported as inadequate support, not as a reliable policy ranking.

For CEM policies, planning samples candidate sequences internally. `deterministic=True` makes the post-plan action greedy but does not by itself freeze the planner's sampling stream. Treat CEM OPE as a diagnostic unless the planning RNG and policy-distribution semantics are controlled for the intended deployment mode.

## Standardization Guidance

Use common schemas for operational metrics, but preserve benchmark semantics. A future environment should declare a primary outcome, safety event, burden/severity metric, direction, units, and supported aggregation. Current `EnvSpec` carries names and a primary metric; it does not yet encode units, ranges, or cross-environment normalization. Do not rank ICU and EpiCare raw returns or clinical metrics against each other.
