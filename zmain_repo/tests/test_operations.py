"""Regression tests for the user-facing benchmark operations layer."""

from __future__ import annotations

import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np

from scripts.bench import _dataset_profile, _validate_offline_inputs, discover_configs
from src.algorithms.common import TrainLogger


class TestOperations(unittest.TestCase):
    def test_catalog_exposes_both_environments_and_epicare_online_configs(self):
        entries = discover_configs()
        paths = {str(entry.relative_path) for entry in entries}
        environments = {entry.env_id for entry in entries}
        self.assertEqual(environments, {"icu_sepsis", "epicare"})
        self.assertIn("online/dev/epicare_dqn.json", paths)
        self.assertIn("offline/dev/epicare_dqn.json", paths)
        self.assertIn("offline/dev/dqn.json", paths)

    def test_online_logger_uses_environment_success_and_action_geometry(self):
        logger = TrainLogger(show_progress=False)
        logger.log_episode(
            1,
            total_reward=3.0,
            episode_length=2,
            total_env_steps=2,
            actions=[0, 15],
            success=False,
            action_distance=lambda previous, current: float(previous != current),
            transition_metrics=[
                {"symptom_burden": 0.8, "remission": 0.0},
                {"symptom_burden": 0.4, "remission": 0.0},
            ],
            outcome={"success": False, "remission": False, "adverse_event": False},
        )
        payload = logger.metrics_payload()
        self.assertEqual(payload["success_rate"].tolist(), [0.0])
        self.assertEqual(payload["survival_rate"].tolist(), [0.0])
        self.assertEqual(payload["action_stability_l1"].tolist(), [1.0])
        self.assertAlmostEqual(payload["environment_metrics"]["mean_symptom_burden"][0], 0.6)
        self.assertEqual(payload["environment_metrics"]["outcome_remission"].tolist(), [0.0])

    def test_dataset_profile_reports_offline_input_and_ope_preconditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.npz"
            np.savez_compressed(
                path,
                observations=np.zeros((2, 3), dtype=np.float32),
                next_observations=np.ones((2, 3), dtype=np.float32),
                state_vectors=np.zeros((2, 3), dtype=np.float32),
                next_state_vectors=np.ones((2, 3), dtype=np.float32),
                actions=np.asarray([0, 1], dtype=np.int64),
                next_actions=np.asarray([1, 0], dtype=np.int64),
                rewards=np.asarray([1.0, -1.0], dtype=np.float32),
                terminals=np.asarray([False, True]),
                state_ids=np.asarray([0, 1], dtype=np.int64),
                next_state_ids=np.asarray([1, 2], dtype=np.int64),
                action_masks=np.ones((2, 2), dtype=np.float32),
                next_action_masks=np.ones((2, 2), dtype=np.float32),
                behavior_logprobs=np.asarray([-0.7, -0.7], dtype=np.float32),
                behavior_policy_ids=np.asarray([11, 11], dtype=np.int64),
                episode_ids=np.asarray([0, 0], dtype=np.int64),
                timesteps=np.asarray([0, 1], dtype=np.int64),
                env_metrics=np.zeros((2, 1), dtype=np.float32),
                next_env_metrics=np.zeros((2, 1), dtype=np.float32),
            )
            profile = _dataset_profile(
                path,
                {
                    "environment": {"obs_backend": "state_vector", "metric_names": ["symptom_burden"]},
                    "policy_ids": {"smart": 11},
                },
            )
        self.assertEqual(profile["primary_input"], "state_vectors")
        self.assertEqual(profile["num_actions"], 2)
        self.assertTrue(profile["all_rows_have_action"])
        self.assertTrue(profile["finite_behavior_logprobs"])
        self.assertEqual(profile["behavior_counts"], {"smart": 2})

    def test_offline_preflight_checks_schema_and_distinct_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.npz"
            val_path = root / "val.npz"
            np.savez_compressed(
                train_path,
                observations=np.zeros((1, 2), dtype=np.float32),
                next_observations=np.zeros((1, 2), dtype=np.float32),
                state_vectors=np.zeros((1, 2), dtype=np.float32),
                next_state_vectors=np.zeros((1, 2), dtype=np.float32),
                actions=np.asarray([0], dtype=np.int64),
                next_actions=np.asarray([0], dtype=np.int64),
                rewards=np.asarray([0.0], dtype=np.float32),
                terminals=np.asarray([True]),
                state_ids=np.asarray([0], dtype=np.int64),
                next_state_ids=np.asarray([0], dtype=np.int64),
                action_masks=np.ones((1, 1), dtype=np.float32),
                next_action_masks=np.ones((1, 1), dtype=np.float32),
                behavior_logprobs=np.asarray([0.0], dtype=np.float32),
                behavior_policy_ids=np.asarray([11], dtype=np.int64),
                episode_ids=np.asarray([0], dtype=np.int64),
                timesteps=np.asarray([0], dtype=np.int64),
            )
            shutil.copyfile(train_path, val_path)
            checks = _validate_offline_inputs(
                {
                    "env_id": "epicare",
                    "obs_backend": "state_vector",
                    "dataset_path": str(train_path),
                    "val_dataset_path": str(val_path),
                    "ope_enabled": False,
                }
            )
        self.assertIn("ready", checks["train"])
        self.assertIn("ready", checks["validation"])


if __name__ == "__main__":
    unittest.main()
