"""Importance-sampling OPE estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchtwm.envs import make_env
from benchtwm.ope import _estimate, evaluate_ope
from benchtwm.policies import Policy, masked_uniform

from conftest import TINY_DATASETS


class UniformPolicy(Policy):
    def action_probabilities(self, obs, info, deterministic=True):
        return masked_uniform(info["action_mask"])


class RecordingUniformPolicy(UniformPolicy):
    def __init__(self):
        super().__init__()
        self.resets = 0
        self.observed = 0

    def reset_episode(self):
        self.resets += 1

    def observe(self, action, reward):
        self.observed += 1


def test_bootstrap_is_seeded(tiny_datasets):
    path = tiny_datasets["epicare"] / "ope_test.npz"
    first = evaluate_ope(UniformPolicy(), dataset_path=path, bootstrap_replicates=4, seed=4)
    second = evaluate_ope(UniformPolicy(), dataset_path=path, bootstrap_replicates=4, seed=4)
    assert first[0] == second[0]
    pd.testing.assert_frame_equal(first[1], second[1])
    assert set(first[0]["estimators"]) == {"IS", "WIS", "PDIS", "WPDIS"}


def test_on_policy_target_recovers_the_logged_return(tiny_datasets):
    path = tiny_datasets["epicare"] / "ope_test.npz"
    data = np.load(path)
    logged_return = np.mean([data["rewards"][data["episode_ids"] == episode].sum() for episode in np.unique(data["episode_ids"])])
    env = make_env(TINY_DATASETS["epicare"])
    try:
        behavior = env.make_policy("smart", seed=0)
        summary, _ = evaluate_ope(behavior, dataset_path=path, bootstrap_replicates=2, seed=0, deterministic=False)
    finally:
        env.close()
    for estimator in summary["estimators"].values():
        assert np.isclose(estimator["estimate"], logged_return, atol=1e-4)
    assert summary["reliability"]["trajectory_ess_fraction"] > 0.99
    assert summary["policy_probability_contract"]["status"] == "exact"


def test_wpdis_pads_terminated_trajectories_with_zero_reward():
    estimates, _ = _estimate(
        [
            {"rewards": np.asarray([1.0, 2.0]), "ratios": np.asarray([1.0, 1.0])},
            {"rewards": np.asarray([3.0]), "ratios": np.asarray([1.0])},
        ],
        gamma=1.0,
    )
    assert estimates == {"IS": 3.0, "WIS": 3.0, "PDIS": 3.0, "WPDIS": 3.0}


def test_recurrent_targets_see_every_episode_boundary(tiny_datasets):
    path = tiny_datasets["icu_sepsis"] / "ope_test.npz"
    data = np.load(path)
    policy = RecordingUniformPolicy()
    summary, bootstrap = evaluate_ope(policy, dataset_path=path, bootstrap_replicates=2, deterministic=False)
    assert policy.resets == len(np.unique(data["episode_ids"]))
    assert policy.observed == len(data["actions"])
    assert summary["diagnostics"]["target_logged_action_support_rate"] == 1.0
    assert len(bootstrap) == 2
