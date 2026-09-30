# Evaluation Protocol

BenchTWM is strictly offline. A run learns only from `train.npz`, selects its
checkpoint only with `val.npz`, and is then measured by two independent
evidence streams. The simulator is never consulted during training or model
selection.

| Stage | Uses | Output |
|---|---|---|
| Optimization and model selection | `train.npz`, `val.npz` | `train_history.csv`, `checkpoint_best.pt` |
| Direct simulator evaluation | fresh seeded simulator episodes | `eval_summary.json`, `episodes_summary.csv`, `trajectories.csv` |
| Off-policy evaluation | held-out `ope_test.npz` | `ope_summary.json`, `ope_bootstrap.csv` |

## Protocol defaults

From `configs/protocol.json`: 5,000 gradient steps, batch 256 (Dreamer: 16
sequences of 64), validation every 250 steps, early stopping after 20 checks
without improvement, gradient clipping at norm 10, and a final evaluation of
500 episodes with evaluation seeds `100000 + i`, identical for every method.
Research matrices use seeds 0, 1, 2.

## Model selection

Every method is monitored by its own validation loss (lower is better); the
checkpoint with the lowest value is deployed. The meaning of that loss is
algorithm-specific (`validation_objective` of each algorithm) and is never a
return. Loss values are not comparable across methods.

## Direct simulator evaluation

`eval_summary.json` contains:

- `mean_return`, `mean_success`, `mean_length`;
- `mean_action_stability_l1` and `mean_abrupt_jump_rate`, computed with the
  environment's own action distance;
- `mean_action_latency_ms`, `planner_calls`, and `planner_model_evaluations`
  (the cost of planning);
- `mean_<metric>` for every per-step metric of the environment, and
  `outcome_<name>` for every extra episode outcome (for EpiCare,
  `outcome_remission`, `outcome_adverse_event`, `outcome_time_to_remission`).

Deployment is deterministic by default (`eval_deterministic`).

## Off-policy evaluation

For each logged action the per-step ratio is
`rho_t = pi_target(a_t | o_t) / pi_behavior(a_t | o_t)`, where the behavior
probability is the exact logged propensity.

| Estimator | Definition | Trade-off |
|---|---|---|
| IS | trajectory likelihood ratio x trajectory return | unbiased, high variance |
| WIS | self-normalized IS | lower variance, finite-sample bias |
| PDIS | cumulative ratio applied at each reward | usually less variable than IS |
| WPDIS | per-step self-normalized PDIS; finished trajectories keep their final weight with zero reward | stabilized |

`ope_summary.json` holds the estimates with seeded trajectory-bootstrap 95%
intervals and diagnostics: target support on logged actions, zero-weight rate,
trajectory ESS (absolute and as a fraction), and maximum log weight. Two
labels guard interpretation:

- `reliability`: `usable`, `limited`, `weak`, `critical`, or `unusable`, from
  the ESS. Do not rank policies on `weak` or worse.
- `policy_probability_contract`: `exact` for BC, DQN, IQL, CQL, and Dreamer;
  `diagnostic` for CEM planners.

By default OPE targets the deployed policy (deterministic for most methods;
Dreamer's recipe targets its stochastic actor). A deterministic target gives
zero weight to every trajectory with one mismatching action, so low ESS is
common and is a finding about support, not a bug.
`bench ope <run> --target-mode stochastic` evaluates the policy's stochastic
distribution instead, which is a different, counterfactual policy.

## Summaries and the planning gain

`bench research summarize` (and every queue's `suite_results.json`) groups
completed runs by dataset and method and reports mean and standard deviation
across seeds of every numeric evaluation field, timings, and OPE estimates.
For each dataset it finds the best model-free method by mean return and
reports, for each model-based method, the **planning gain**
(`mean_return - best_model_free_return`, and the same for success) next to
its training time, evaluation time, action latency, and planner model
evaluations. Datasets are never averaged together.
