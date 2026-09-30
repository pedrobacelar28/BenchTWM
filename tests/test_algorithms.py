"""Contract tests run for every registered algorithm on every environment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from benchtwm.algorithms import MODEL_BASED, MODEL_FREE, algorithm_names, get_algorithm, load_policy
from benchtwm.data.dataset import OfflineTransitionDataset
from benchtwm.envs import environment_ids
from benchtwm.evaluation import evaluate_policy
from benchtwm.operations.artifacts import initial_manifest
from benchtwm.runner import run_experiment
from benchtwm.training import train_offline

from conftest import TINY_HYPERPARAMETERS, tiny_run_config

ALGORITHMS = algorithm_names()
CASES = [(algo, env_id) for algo in ALGORITHMS for env_id in environment_ids()]


def test_every_algorithm_has_tiny_hyperparameters():
    assert set(ALGORITHMS) <= set(TINY_HYPERPARAMETERS)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_registration_metadata(algo):
    cls = get_algorithm(algo)
    assert get_algorithm(cls.method_id) is cls
    assert cls.family in {MODEL_FREE, MODEL_BASED}
    assert cls.description and cls.validation_objective


@pytest.mark.parametrize(("algo", "env_id"), CASES)
def test_train_load_and_act_within_mask(algo, env_id, tiny_datasets):
    config = tiny_run_config(algo, env_id, tiny_datasets[env_id])
    result = train_offline(config)
    assert result.summary["total_gradient_steps"] == config["offline_train_steps"]
    assert np.isfinite(result.summary["best_validation_loss"])
    assert [row["step"] for row in result.history] == [2, 4]
    policy = load_policy(config, result.best_checkpoint)
    data = OfflineTransitionDataset(tiny_datasets[env_id] / "train.npz")
    policy.reset_episode()
    for index in range(3):
        mask = data.data["action_masks"][index]
        info = {"action_mask": mask, "admissible_actions": np.flatnonzero(mask).tolist(), "state_id": int(data.data["state_ids"][index])}
        obs = data.data["observations"][index]
        probabilities = policy.action_probabilities(obs, info, deterministic=False)
        assert probabilities.shape == mask.shape and np.isclose(probabilities.sum(), 1.0)
        assert np.all(probabilities[mask <= 0] == 0.0)
        assert mask[policy.act(obs, info, deterministic=True)] == 1
        policy.observe(int(data.data["actions"][index]), float(data.data["rewards"][index]))


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_runner_writes_every_artifact(algo, tiny_datasets, tmp_path):
    config = tiny_run_config(algo, "icu_sepsis", tiny_datasets["icu_sepsis"], dataset_id="test/icu", method_id=get_algorithm(algo).method_id)
    manifest = initial_manifest(config, dataset_id="test/icu", method_id=config["method_id"], run_dir=tmp_path)
    run_dir = run_experiment(config, output_dir=tmp_path / "run", manifest=manifest, quiet=True)
    for name in ("config.json", "run_manifest.json", "train_history.csv", "checkpoint_best.pt", "checkpoint_last.pt", "eval_summary.json",
                 "episodes_summary.csv", "trajectories.csv", "environment_spec.json", "ope_summary.json", "ope_bootstrap.csv", "metrics_summary.json"):
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / "run_manifest.json").read_text())["status"] == "completed"
    summary = json.loads((run_dir / "metrics_summary.json").read_text())
    assert summary["family"] == get_algorithm(algo).family
    assert "WPDIS_estimate" in summary["ope_summary"]
    assert "validation_loss" in pd.read_csv(run_dir / "train_history.csv").columns
    # A completed run is skipped unless forced.
    assert run_experiment(config, output_dir=run_dir, quiet=True) == run_dir


def test_failed_run_is_recorded(tiny_datasets, tmp_path):
    config = tiny_run_config("dqn", "icu_sepsis", tiny_datasets["icu_sepsis"], dataset_path=str(tmp_path / "missing.npz"))
    manifest = initial_manifest(config, dataset_id="test/icu", method_id="dqn", run_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        run_experiment(config, output_dir=tmp_path / "run", manifest=manifest, quiet=True)
    assert json.loads((tmp_path / "run" / "run_manifest.json").read_text())["status"] == "failed"


def test_early_stopping_selects_first_checkpoint_when_validation_never_improves(tiny_datasets):
    config = tiny_run_config("dqn", "icu_sepsis", tiny_datasets["icu_sepsis"], learning_rate=0.0, offline_train_steps=10, offline_log_interval=1, early_stopping_patience=1)
    summary = train_offline(config).summary
    assert summary["early_stopped"] and summary["best_validation_step"] == 1 and summary["early_stop_step"] == 2
    assert summary["num_validation_checks"] == 2


def test_world_models_record_rollout_diagnostics(tiny_datasets):
    config = tiny_run_config("reconstructive_model", "icu_sepsis", tiny_datasets["icu_sepsis"], cem_horizon=5)
    row = train_offline(config).history[-1]
    for key in ("validation_rollout1_state_error", "validation_rollout5_discounted_return_error", "validation_rollout5_cumulative_state_error"):
        assert key in row


def test_simulator_evaluation_is_deterministic(tiny_datasets):
    config = tiny_run_config("dqn", "icu_sepsis", tiny_datasets["icu_sepsis"])
    policy = load_policy(config, train_offline(config).best_checkpoint)
    first = evaluate_policy(policy, config=config, num_episodes=3, eval_seed_base=7)
    second = evaluate_policy(policy, config=config, num_episodes=3, eval_seed_base=7)
    pd.testing.assert_frame_equal(first["trajectories"], second["trajectories"])
