"""Contract tests run against every registered environment."""

from __future__ import annotations

import numpy as np
import pytest

from benchtwm.data.collect import collect_dataset
from benchtwm.data.dataset import REQUIRED_ARRAYS, OfflineTransitionDataset
from benchtwm.envs import environment_ids, get_environment, make_env
from benchtwm.evaluation import evaluate_policy

from conftest import TINY_DATASETS, tiny_dataset_config

ENVIRONMENTS = environment_ids()


def _env(env_id: str):
    return make_env(TINY_DATASETS[env_id], seed=0)


def test_every_environment_has_tiny_fixtures():
    assert set(ENVIRONMENTS) == set(TINY_DATASETS)


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_spec_matches_observations_and_info(env_id):
    env = _env(env_id)
    try:
        obs, info = env.reset(seed=0)
        spec = env.spec
        assert obs.dtype == np.float32 and obs.shape == (spec.obs_dim,)
        assert len(spec.feature_names) == spec.obs_dim
        assert len(spec.action_labels) == spec.num_actions
        assert info["action_mask"].shape == (spec.num_actions,)
        assert set(np.flatnonzero(info["action_mask"])) == set(info["admissible_actions"])
        assert isinstance(info["state_id"], int)
        assert list(env.transition_metrics(info)) == list(spec.metric_names)
        _, reward, terminated, truncated, next_info = env.step(int(info["admissible_actions"][0]))
        assert np.isfinite(reward) and isinstance(terminated, bool) and isinstance(truncated, bool)
        assert next_info["action_mask"].shape == (spec.num_actions,)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_reset_seed_reproduces_trajectory(env_id):
    env = _env(env_id)
    try:
        rollouts = []
        for _ in range(2):
            obs, info = env.reset(seed=21)
            steps = [obs]
            for _ in range(3):
                obs, reward, terminated, truncated, info = env.step(int(info["admissible_actions"][0]))
                steps.append(np.append(obs, reward))
                if terminated or truncated:
                    break
            rollouts.append(steps)
        for first, second in zip(*rollouts):
            np.testing.assert_array_equal(first, second)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_behavior_policies_are_valid_distributions(env_id):
    env = _env(env_id)
    try:
        for name in env.behavior_policy_ids:
            policy = env.make_policy(name, seed=0, options=TINY_DATASETS[env_id])
            obs, info = env.reset(seed=1)
            policy.reset_episode()
            for deterministic in (False, True):
                probabilities = policy.action_probabilities(obs, info, deterministic=deterministic)
                assert probabilities.shape == (env.num_actions,), name
                assert np.isclose(probabilities.sum(), 1.0), name
                assert np.all(probabilities[info["action_mask"] <= 0] == 0.0), name
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_reference_policies_evaluate(env_id):
    config = {**TINY_DATASETS[env_id], "method_id": "baseline"}
    env = _env(env_id)
    try:
        for name in get_environment(env_id).reference_policies:
            policy = env.make_policy(name, seed=0, options=config)
            summary = evaluate_policy(policy, config=config, num_episodes=2, eval_seed_base=0, export_csv=False)["summary"]
            assert summary["episodes"] == 2
            for metric in env.spec.metric_names:
                assert f"mean_{metric}" in summary
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_collected_dataset_schema_propensities_and_splits(env_id, tiny_datasets):
    root = tiny_datasets[env_id]
    splits = {name: np.load(root / f"{name}.npz") for name in ("train", "val", "ope_test")}
    for name, data in splits.items():
        assert REQUIRED_ARRAYS.issubset(data.files), name
        assert data["observations"].dtype == np.float32
        assert np.isfinite(data["behavior_logprobs"]).all() and (data["behavior_logprobs"] <= 0).all()
        assert all(mask[action] == 1 for action, mask in zip(data["actions"], data["action_masks"]))
    episode_sets = [set(data["episode_ids"].tolist()) for data in splits.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(episode_sets) for b in episode_sets[i + 1 :])
    dataset = OfflineTransitionDataset(root / "train.npz")
    assert dataset.obs_dim == make_env(TINY_DATASETS[env_id]).obs_dim
    assert dataset.feature_names == make_env(TINY_DATASETS[env_id]).spec.feature_names


@pytest.mark.parametrize("env_id", ENVIRONMENTS)
def test_dataset_collection_is_deterministic(env_id, tmp_path):
    for name in ("a", "b"):
        collect_dataset(tiny_dataset_config(env_id, tmp_path / name, seed=13))
    for split in ("train.npz", "val.npz", "ope_test.npz"):
        first, second = np.load(tmp_path / "a" / split), np.load(tmp_path / "b" / split)
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key], err_msg=f"{split}:{key}")
