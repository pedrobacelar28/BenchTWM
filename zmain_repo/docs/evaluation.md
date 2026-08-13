# Training, Direct Evaluation, and OPE

## v0 Run Contract

Every v0 method uses `offline_batch_size=256`, `offline_train_steps=5000`,
validation/checkpoint checks every 250 steps, and validation-loss early
stopping. Final simulator evaluation uses 500 fixed episodes and common seeds
for every method on a task. The runner records optimization/evaluation time,
mean action latency, and (for planning) planner calls and estimated model
evaluations. Resolved `task_id` and `track` are persisted in the config,
manifest, metrics, and report.

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
| BC | Masked action cross-entropy and entropy |
| DQN/FQI | Masked TD loss |
| IQL | Q, expectile-value, advantage-weighted actor, and entropy losses |
| CQL | TD and admissible-action `logsumexp` conservative losses |
| Raw model | Dynamics, prediction, reward, continuation losses |
| Latent model | Latent dynamics, prediction, reward, continuation losses |
| Reconstructive model | Latent/dynamics, reconstruction, prediction, reward, continuation losses |

World-model runs additionally report rollout validation errors over multiple horizons. Direct return and success curves are separate simulator outputs, not validation-loss substitutes.

## Early Stopping

Every offline recipe uses `validation_loss` as its checkpoint-selection and
early-stopping monitor; lower is better. The meaning of that scalar remains
algorithm-specific and it is never the simulator return. PPO remains
implemented for compatibility/online experiments but is not a public v0
offline baseline.

The training plan prints the monitor, direction, `min_delta`, patience in
validation checks, and validation cadence. At every validation check, the CLI
prints the current and best validation loss plus the number of checks without
improvement. When patience is exhausted, it announces the early-stop step and
the best-checkpoint step before proceeding to the final simulator evaluation
and optional OPE phases.

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

The current implementation also reports the observed minimum/maximum behavior
probability, `target_zero_probability_rate`, ESS as a fraction of held-out
trajectories, an OPE reliability status, and the policy-probability contract.
The latter is `exact` for BC, DQN, IQL, and CQL action distributions. CEM world-model
probabilities are explicitly marked `diagnostic` because the planner samples
candidate sequences internally.

### Target Policy Mode

By default, OPE estimates the policy actually deployed by direct evaluation:

```bash
python scripts/bench.py ope <run> --target-mode deployed --bootstrap 1000
```

For the default DQN/PPO deployment this is deterministic, so only logged
actions equal to the greedy action have nonzero target probability. This is
correct for that deployed policy, but can make ordinary IS estimators extremely
high variance. `--target-mode stochastic` instead uses the policy's stochastic
distribution (for DQN, the Q-value softmax). It may reduce literal zero target
probabilities, but it evaluates a different, counterfactual policy and must
not be compared directly with deterministic simulator results.

Use `--write-canonical` only after a deployed-mode recomputation when the
result should replace the run-level `ope_summary.json`; all recomputations are
also retained under `run/ope/<timestamp>/`.

## Interpreting OPE

For a deterministic target policy, a logged transition has target probability zero unless the logged action matches the deployed action. A long trajectory becomes zero weighted after its first mismatch. Thus support rate and ESS are validity diagnostics, not cosmetic statistics. Full behavior-policy action support is necessary but not sufficient: the behavior must also put enough probability mass on action *sequences* similar to the target policy.

EpiCare SMART itself is correctly logged: its behavior probabilities are
strictly positive for every treatment, and OPE of SMART against its own held-out
data recovers the logged mean return. Learned deterministic policies can still
have low ESS because their greedy action sequences differ from SMART. Such a
result should be reported as insufficient effective support, not as a reliable
policy ranking. A behavior policy closer to the candidate policies, a much
larger logged set, or a direct-method estimator such as FQE is required for a
stronger offline ranking.

For CEM policies, planning samples candidate sequences internally. `deterministic=True` makes the post-plan action greedy but does not by itself freeze the planner's sampling stream. Treat CEM OPE as a diagnostic unless the planning RNG and policy-distribution semantics are controlled for the intended deployment mode.

## Standardization Guidance

Use common schemas for operational metrics, but preserve benchmark semantics. A future environment should declare a primary outcome, safety event, burden/severity metric, direction, units, and supported aggregation. Current `EnvSpec` carries names and a primary metric; it does not yet encode units, ranges, or cross-environment normalization. Do not rank ICU and EpiCare raw returns or clinical metrics against each other.
