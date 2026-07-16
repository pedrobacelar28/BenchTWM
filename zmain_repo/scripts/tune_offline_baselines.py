"""Tune offline baseline agents with Optuna.

Edit the constants below and run:

    python scripts/tune_offline_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.optuna_common import run_parallel_studies, selected_specs


N_TRIALS_PER_MODEL = 20
DATASET_SIZE = 5_000
SEED_BASE = 100_000
DATASET_TAG = f"{DATASET_SIZE // 1000}k"
DATASET_DIR = PROJECT_ROOT / "datasets/offline/dev/expert" / str(DATASET_SIZE)
STUDY_DIR = PROJECT_ROOT / "runs" / f"optuna-expert-{DATASET_TAG}-baselines"
LOG_DIR = STUDY_DIR / "logs"

# Pick one or two keys from AVAILABLE_MODELS. List order maps to GPU ids 0, 1.
SELECTED_MODELS = ["dqn", "ppo"]

AVAILABLE_MODELS = {
    "dqn": {
        "name": f"dqn_offline_expert_{DATASET_TAG}",
        "base_config": PROJECT_ROOT / "experiments/offline/dev/dqn.json",
    },
    "ppo": {
        "name": f"ppo_offline_expert_{DATASET_TAG}",
        "base_config": PROJECT_ROOT / "experiments/offline/dev/ppo.json",
    },
}

COMMON_OVERRIDES = {
    "dataset_path": str(DATASET_DIR / "train.npz"),
    "val_dataset_path": str(DATASET_DIR / "val.npz"),
    "offline_train_steps": 500_000,
    "offline_batch_size": 512,
    "offline_log_interval": 1_000,
    "offline_eval_during_training": False,
    "offline_eval_interval": 0,
    "offline_eval_episodes": 10_000,
    "eval_episodes": 10_000,
    "eval_seed_base": SEED_BASE,
    "save_checkpoints": True,
    "save_trajectories": True,
    "show_progress": True,
    "offline_console_style": "compact",
    "track": False,
    "use_tensorboard": False,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.0,
}

FORCE_RUN = False


def _suggest_dqn(trial, params: dict[str, Any]) -> None:
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    params["gamma"] = trial.suggest_categorical("gamma", [0.95, 0.97, 0.99, 1.0])
    params["tau"] = trial.suggest_float("tau", 0.001, 0.2, log=True)
    params["target_network_frequency"] = trial.suggest_categorical(
        "target_network_frequency",
        [128, 256, 512, 1024, 2048],
    )


def _suggest_ppo(trial, params: dict[str, Any]) -> None:
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    params["gamma"] = trial.suggest_categorical("gamma", [0.95, 0.97, 0.99, 1.0])
    params["clip_coef"] = trial.suggest_float("clip_coef", 0.05, 0.8)
    params["ent_coef"] = trial.suggest_float("ent_coef", 1e-5, 5e-2, log=True)
    params["vf_coef"] = trial.suggest_float("vf_coef", 0.05, 1.0)
    params["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.1, 2.0)
    params["offline_bc_weight"] = trial.suggest_float("offline_bc_weight", 1e-3, 2.0, log=True)
    params["norm_adv"] = trial.suggest_categorical("norm_adv", [True, False])


def suggest_params(trial, params: dict[str, Any]) -> None:
    suggesters = {
        "dqn": _suggest_dqn,
        "ppo": _suggest_ppo,
    }
    try:
        suggesters[params["algo"]](trial, params)
    except KeyError as exc:
        raise ValueError(f"Unsupported baseline algo for tuning: {params['algo']!r}") from exc


def main() -> None:
    if not Path(COMMON_OVERRIDES["dataset_path"]).exists() or not Path(COMMON_OVERRIDES["val_dataset_path"]).exists():
        raise FileNotFoundError(f"Missing dataset files under {DATASET_DIR}")
    specs = selected_specs(
        selected_models=SELECTED_MODELS,
        available_models=AVAILABLE_MODELS,
        min_models=1,
        max_models=2,
    )
    run_parallel_studies(
        specs=specs,
        study_dir=STUDY_DIR,
        log_dir=LOG_DIR,
        n_trials=N_TRIALS_PER_MODEL,
        seed_base=SEED_BASE,
        common_overrides=COMMON_OVERRIDES,
        force_run=FORCE_RUN,
        suggest=suggest_params,
    )


if __name__ == "__main__":
    main()
