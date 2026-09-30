from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.algorithms.registry import get_policy_loader
from src.offline.generate_dataset import generate_dataset
from src.offline.trainers import get_offline_trainer
from src.operations.catalog import Catalog
from src.operations.artifacts import initial_manifest
from src.run_experiment import run_experiment


def _tiny_icu_dataset(root: Path) -> dict:
    return {
        "dataset_name": "v0_tiny_icu",
        "output_dir": str(root),
        "behavior_policy": "expert",
        "env_id": "icu_sepsis",
        "obs_backend": "state_vector",
        "seed": 3,
        "environment_seed": 1,
        "num_train_episodes": 4,
        "num_val_episodes": 2,
        "gamma": 1.0,
        "overwrite": True,
        "optimal_value_iteration_steps": 2,
    }


class TestV0Benchmark(unittest.TestCase):
    def test_direct_baselines_train_load_and_respect_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            generate_dataset(_tiny_icu_dataset(dataset_dir))
            data = np.load(dataset_dir / "train.npz")
            for algo in ("bc", "iql", "cql"):
                with self.subTest(algo=algo):
                    config = {
                        "algo": algo,
                        "train_mode": "offline",
                        "env_id": "icu_sepsis",
                        "obs_backend": "state_vector",
                        "seed": 0,
                        "dataset_path": str(dataset_dir / "train.npz"),
                        "val_dataset_path": str(dataset_dir / "val.npz"),
                        "offline_train_steps": 2,
                        "offline_batch_size": 4,
                        "offline_log_interval": 1,
                        "offline_eval_interval": 0,
                        "offline_eval_during_training": False,
                        "learning_rate": 1e-3,
                        "policy_architecture": "mlp",
                        "policy_hidden_dim": 16,
                        "policy_hidden_layers": 1,
                        "gamma": 1.0,
                        "iql_expectile_tau": 0.7,
                        "iql_temperature_beta": 3.0,
                        "iql_advantage_clip": 100.0,
                        "cql_alpha": 1.0,
                        "max_grad_norm": 10.0,
                        "early_stopping_patience": 2,
                        "early_stopping_min_delta": 0.0,
                        "success_threshold": 0.0,
                        "show_progress": False,
                    }
                    result = get_offline_trainer(algo)(config, root / f"{algo}-out")
                    self.assertIn("model_state_dict", result["checkpoint_best"])
                    policy = get_policy_loader(algo)(config, result["checkpoint_best"])
                    mask = data["action_masks"][0]
                    obs = data["state_vectors"][0]
                    info = {"action_mask": mask, "admissible_actions": np.flatnonzero(mask).tolist(), "state_id": int(data["state_ids"][0])}
                    probabilities = policy.action_probabilities(obs, info, deterministic=False)
                    self.assertTrue(np.isfinite(probabilities).all())
                    self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
                    self.assertTrue(np.all(probabilities[mask <= 0] == 0.0))
                    self.assertIn(policy.act(obs, info), info["admissible_actions"])

    def test_v0_catalog_persists_continuous_task_and_tracks(self):
        catalog = Catalog.load()
        params, _, recipe = catalog.run_config(
            "icu-sepsis/expert/dev",
            "bc",
            artifact_root="/tmp/benchtwm-v0-test",
            seed=2,
        )
        self.assertEqual(params["obs_backend"], "state_vector")
        self.assertEqual(params["action_type"], "discrete")
        self.assertEqual(params["track"], "direct_rl")
        self.assertEqual(params["task_id"], "icu-sepsis/expert/dev|obs=state_vector|action=discrete|mode=offline")
        self.assertEqual(recipe.track, "direct_rl")
        self.assertNotIn("ppo", [item.method_id for item in catalog.compatible_methods("icu-sepsis/expert/dev")])


if __name__ == "__main__":
    unittest.main()
