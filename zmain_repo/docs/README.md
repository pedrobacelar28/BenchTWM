# Project Layout

- `src/`: training code, algorithm registry, environment wrappers, models, and
  evaluation utilities.
- `analysis/`: aggregation, plotting, robustness evaluation, and clinical
  export/trajectory visualization.
- `experiments/datasets/`: offline dataset generation configs.
- `experiments/online/`: optional online smoke configs for retained methods.
- `experiments/offline/`: fixed-dataset training configs.
- `scripts/`: local sweep helpers, Optuna launchers, and result collectors.
- `results/`: per-run outputs generated locally and intentionally not included
  in this public copy.
- `tests/`: unit and smoke tests.
- `third_party/icu-sepsis/`: vendored benchmark packages required by the
  environment wrapper and reference-policy helpers.

## Script Entry Points

- `scripts/run_sweep.py`: run pending indices from JSON sweep files.
- `scripts/run_offline_matrix.py`: launch the offline experiment matrix across
  dataset sizes.
- `scripts/tune_offline_baselines.py`: Optuna tuning for offline baseline
  agents.
- `scripts/tune_offline_world_models.py`: Optuna tuning for CEM world-model
  variants.
- `scripts/collect_best_trials.py`: collect best Optuna trial output folders.
- `scripts/evaluate_best_trials.py`: run fresh simulator evaluations for
  collected best trials.
- `scripts/rerun_best_world_model_trials.py`: rerun selected best world-model
  trials with trajectory export enabled.

## Per-run outputs

Each run produces:

- `config.json`
- `metrics.pkl`
- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `episodes_summary.csv`
- `trajectories.csv`
- `offline_training_history.csv` for offline runs
- `offline_eval_history.csv` for offline runs

All outputs are stored under `results/<algo>/<hash>/`.

Offline runs are stored under explicit method names such as
`results/dqn_offline/<hash>/` and
`results/raw_model_discrete_cem_offline/<hash>/`.

## Offline Mode

The default training mode is still online. Set `train_mode: "offline"` to train
from a fixed `.npz` dataset and evaluate checkpoints in the simulator.

Offline datasets are generated with:

```bash
python src/offline/generate_dataset.py experiments/datasets/dev/expert_100000.json
```

Offline training uses the same runner:

```bash
python src/run_config.py experiments/offline/dev/dqn.json 0
```

Offline training is step-budgeted, matching the OGBench convention:

- `offline_train_steps`: total number of gradient updates.
- `offline_batch_size`: number of transitions sampled per update.
- `offline_log_interval`: frequency, in updates, for training/validation loss.
- `offline_eval_interval`: frequency, in updates, for simulator evaluation.
- `offline_eval_episodes`: number of fresh simulator episodes per periodic
  evaluation.
- `eval_episodes`: number of fresh simulator episodes in the final evaluation.
- `offline_console_style`: `"compact"` prints one readable line per validation
  and evaluation checkpoint. `"verbose"` prints one metric per line.

For example, `offline_train_steps: 5000` and `offline_batch_size: 512` means
5000 gradient updates and 2.56M sampled transitions with replacement. The
reported `effective_epochs` is
`offline_train_steps * offline_batch_size / train_dataset_transitions`; it is a
diagnostic, not the official budget. Validation loss is computed on one random
batch from the validation dataset, while evaluation metrics are computed from
complete simulator episodes.

Available dataset behavior policies are `random_admissible`, `expert`,
`optimal`, `epsilon_expert`, `mixed`, and `policy_checkpoint`. Expert and
optimal policies are only allowed inside dataset generation and reference
analysis; offline trainers only load the saved `.npz` arrays.

Offline method names are explicit:

- `dqn_offline`: fitted DQN with a fixed dataset and target network.
- `ppo_logged_offline`: logged PPO-style update using `behavior_logprobs` plus a
  behavior-cloning regularizer. This is not on-policy PPO.
- `raw_model_discrete_cem_offline` and `raw_model_continuous_cem_offline`:
  direct next-observation models with CEM planning.
- `latent_model_discrete_cem_offline` and
  `latent_model_continuous_cem_offline`: latent transition models trained with
  EMA target encoders and CEM planning.
- `reconstructive_model_discrete_cem_offline` and
  `reconstructive_model_continuous_cem_offline`: latent transition models with
  next-observation reconstruction and CEM planning.

Offline curves use evaluation checkpoints as the return curve. For offline
runs, `training_history.csv`, `offline_training_history.csv`, and
`offline_eval_history.csv` use gradient-update steps as the x-axis.
`offline_training_history.csv` stores dataset losses, while
`offline_eval_history.csv` stores simulator evaluation metrics.
