"""Tune offline CEM world-model variants with Optuna.

Edit the constants below and run:

    python scripts/tune_offline_world_models.py
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
STUDY_DIR = PROJECT_ROOT / "runs" / f"optuna-expert-{DATASET_TAG}-world-models"
LOG_DIR = STUDY_DIR / "logs"

# Pick one or two keys from AVAILABLE_MODELS. List order maps to GPU ids 0, 1.
SELECTED_MODELS = ["raw_model_continuous", "reconstructive_model_discrete"]

AVAILABLE_MODELS = {
    "raw_model_discrete": {
        "name": f"raw_model_discrete_cem_expert_{DATASET_TAG}",
        "base_config": PROJECT_ROOT / "experiments/offline/dev/raw_model_discrete_cem.json",
    },
    "raw_model_continuous": {
        "name": f"raw_model_continuous_cem_expert_{DATASET_TAG}",
        "base_config": PROJECT_ROOT / "experiments/offline/dev/raw_model_continuous_cem.json",
    },
    "latent_model_discrete": {
        "name": f"latent_model_discrete_cem_expert_{DATASET_TAG}",
        "base_config": PROJECT_ROOT / "experiments/offline/dev/latent_model_discrete_cem.json",
    },
    "latent_model_continuous": {
        "name": f"latent_model_continuous_cem_expert_{DATASET_TAG}",
        "base_config": PROJECT_ROOT / "experiments/offline/dev/latent_model_continuous_cem.json",
    },
    "reconstructive_model_discrete": {
        "name": f"reconstructive_model_discrete_cem_expert_{DATASET_TAG}",
        "base_config": PROJECT_ROOT / "experiments/offline/dev/reconstructive_model_discrete_cem.json",
    },
    "reconstructive_model_continuous": {
        "name": f"reconstructive_model_continuous_cem_expert_{DATASET_TAG}",
        "base_config": PROJECT_ROOT / "experiments/offline/dev/reconstructive_model_continuous_cem.json",
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


def _suggest_common(trial, params: dict[str, Any]) -> None:
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 384, 512])
    params["action_embed_dim"] = trial.suggest_categorical("action_embed_dim", [16, 32, 64])
    params["gamma"] = trial.suggest_categorical("gamma", [0.95, 0.97, 0.99, 1.0])
    params["reward_loss_weight"] = trial.suggest_float("reward_loss_weight", 0.5, 10.0, log=True)
    params["continue_loss_weight"] = trial.suggest_float("continue_loss_weight", 0.25, 8.0, log=True)
    params["cem_horizon"] = trial.suggest_int("cem_horizon", 4, 12)
    params["cem_samples"] = trial.suggest_categorical("cem_samples", [128, 256, 512])
    params["cem_elites"] = trial.suggest_categorical("cem_elites", [16, 32, 64])
    params["cem_iterations"] = trial.suggest_int("cem_iterations", 3, 7)
    params["cem_alpha"] = trial.suggest_float("cem_alpha", 0.05, 0.6)
    if params["cem_elites"] >= params["cem_samples"]:
        params["cem_elites"] = max(1, params["cem_samples"] // 8)


def _suggest_raw_model(trial, params: dict[str, Any]) -> None:
    _suggest_common(trial, params)
    params["dynamics_loss_weight"] = trial.suggest_float("dynamics_loss_weight", 0.25, 5.0, log=True)


def _suggest_latent_model(trial, params: dict[str, Any]) -> None:
    _suggest_common(trial, params)
    params["latent_dim"] = trial.suggest_categorical("latent_dim", [64, 128, 256])
    params["dynamics_loss_weight"] = trial.suggest_float("dynamics_loss_weight", 0.25, 5.0, log=True)
    params["target_encoder_decay"] = trial.suggest_categorical(
        "target_encoder_decay",
        [0.95, 0.98, 0.99, 0.995, 0.999],
    )


def _suggest_reconstructive_model(trial, params: dict[str, Any]) -> None:
    _suggest_common(trial, params)
    params["latent_dim"] = trial.suggest_categorical("latent_dim", [64, 128, 256])
    params["latent_loss_weight"] = trial.suggest_float("latent_loss_weight", 0.25, 5.0, log=True)
    params["reconstruction_loss_weight"] = trial.suggest_float("reconstruction_loss_weight", 0.25, 5.0, log=True)


def suggest_params(trial, params: dict[str, Any]) -> None:
    suggesters = {
        "raw_model": _suggest_raw_model,
        "latent_model": _suggest_latent_model,
        "reconstructive_model": _suggest_reconstructive_model,
    }
    try:
        suggesters[params["algo"]](trial, params)
    except KeyError as exc:
        raise ValueError(f"Unsupported world-model algo for tuning: {params['algo']!r}") from exc


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
