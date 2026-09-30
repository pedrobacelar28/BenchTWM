"""Shared fixtures: tiny collected datasets and tiny run configurations."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchtwm.data.collect import collect_dataset

TINY_DATASETS = {
    "icu_sepsis": {"env_id": "icu_sepsis", "env_kwargs": {}, "behavior_policy": "epsilon_expert", "epsilon": 0.3},
    "epicare": {"env_id": "epicare", "env_kwargs": {"patient_modifiers": True}, "environment_seed": 1, "behavior_policy": "smart"},
}

# Small networks and budgets; every algorithm-specific key is listed so the
# tests fail loudly if an algorithm starts reading an undeclared parameter.
TINY_HYPERPARAMETERS = {
    "common": {
        "offline_train_steps": 4,
        "offline_batch_size": 8,
        "offline_log_interval": 2,
        "learning_rate": 1e-3,
        "gamma": 1.0,
        "max_grad_norm": 10.0,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.0,
        "policy_hidden_dim": 16,
        "policy_hidden_layers": 1,
        "policy_temperature": 1.0,
        "eval_episodes": 2,
        "eval_seed_base": 100_000,
        "eval_deterministic": True,
        "ope_bootstrap_replicates": 4,
        "save_trajectories": True,
        "show_progress": False,
    },
    "dqn": {"tau": 0.5, "target_network_frequency": 1},
    "bc": {},
    "iql": {"iql_expectile_tau": 0.7, "iql_temperature_beta": 3.0, "iql_advantage_clip": 100.0},
    "cql": {"cql_alpha": 1.0},
    "raw_model": {"hidden_dim": 16, "action_embed_dim": 4, "cem_horizon": 2, "cem_samples": 8, "cem_elites": 2, "cem_iterations": 1},
    "latent_model": {"hidden_dim": 16, "action_embed_dim": 4, "latent_dim": 8, "target_encoder_decay": 0.99, "cem_horizon": 2, "cem_samples": 8, "cem_elites": 2, "cem_iterations": 1},
    "reconstructive_model": {"hidden_dim": 16, "action_embed_dim": 4, "latent_dim": 8, "cem_horizon": 2, "cem_samples": 8, "cem_elites": 2, "cem_iterations": 1},
    "dreamer": {
        "offline_batch_size": 2,
        "dreamer_batch_length": 4,
        "dreamer_deter": 16,
        "dreamer_hidden": 8,
        "dreamer_stoch": 2,
        "dreamer_classes": 4,
        "dreamer_blocks": 4,
        "dreamer_units": 8,
        "dreamer_enc_layers": 1,
        "dreamer_dec_layers": 1,
        "dreamer_rssm_img_layers": 1,
        "dreamer_bins": 5,
        "dreamer_imag_length": 2,
        "dreamer_warmup": 0,
    },
}


def tiny_dataset_config(env_id: str, output_dir: Path, *, seed: int = 3) -> dict:
    return {
        **TINY_DATASETS[env_id],
        "seed": seed,
        "num_train_episodes": 6,
        "num_val_episodes": 3,
        "num_ope_episodes": 3,
        "ope_enabled": True,
        "output_dir": str(output_dir),
        "overwrite": True,
        "show_progress": False,
        "quiet": True,
    }


def tiny_run_config(algo: str, env_id: str, dataset_dir: Path, **overrides) -> dict:
    return {
        **{key: value for key, value in TINY_DATASETS[env_id].items() if key in {"env_id", "env_kwargs", "environment_seed"}},
        **TINY_HYPERPARAMETERS["common"],
        **TINY_HYPERPARAMETERS[algo],
        "algo": algo,
        "seed": 0,
        "dataset_path": str(dataset_dir / "train.npz"),
        "val_dataset_path": str(dataset_dir / "val.npz"),
        "ope_dataset_path": str(dataset_dir / "ope_test.npz"),
        "ope_enabled": True,
        **overrides,
    }


@pytest.fixture(scope="session")
def tiny_datasets(tmp_path_factory) -> dict[str, Path]:
    """One tiny collected dataset per environment, shared by the whole session."""

    root = tmp_path_factory.mktemp("datasets")
    paths = {}
    for env_id in TINY_DATASETS:
        paths[env_id] = root / env_id
        collect_dataset(tiny_dataset_config(env_id, paths[env_id]))
    return paths
