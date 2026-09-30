import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.operations.reporting import build_run_report
from src.envs.factory import make_env
from src.envs.reference import make_reference_policy
from src.offline.dataset import REQUIRED_ARRAYS
from src.offline.generate_dataset import generate_dataset
from src.offline.ope import _estimate, evaluate_ope
from src.offline.trainers import get_offline_trainer


def _dataset_config(output_dir: Path) -> dict:
    return {
        "env_id": "epicare",
        "environment_seed": 1,
        "obs_backend": "state_vector",
        "output_dir": str(output_dir),
        "behavior_policy": "smart",
        "seed": 7,
        "num_train_episodes": 8,
        "num_val_episodes": 3,
        "num_ope_episodes": 4,
        "ope_enabled": True,
        "overwrite": True,
    }


def _offline_config(data_dir: Path, algo: str) -> dict:
    config = {
        "env_id": "epicare",
        "environment_seed": 1,
        "obs_backend": "state_vector",
        "algo": algo,
        "train_mode": "offline",
        "seed": 0,
        "dataset_path": str(data_dir / "train.npz"),
        "val_dataset_path": str(data_dir / "val.npz"),
        "offline_train_steps": 1,
        "offline_batch_size": 4,
        "offline_log_interval": 1,
        "offline_eval_interval": 1,
        "offline_eval_episodes": 1,
        "eval_episodes": 1,
        "eval_seed_base": 100,
        "gamma": 1.0,
        "learning_rate": 0.001,
        "success_threshold": 0.0,
        "show_progress": False,
        "policy_architecture": "mlp",
        "policy_hidden_dim": 16,
        "policy_hidden_layers": 1,
        "early_stopping_patience": 0,
    }
    if algo == "dqn":
        return {**config, "tau": 0.5, "target_network_frequency": 1}
    shared = {**config, "hidden_dim": 16, "action_embed_dim": 4, "cem_horizon": 2, "cem_samples": 4, "cem_elites": 2, "cem_iterations": 1, "cem_alpha": 0.2, "reward_loss_weight": 1.0, "continue_loss_weight": 1.0}
    if algo == "raw_model":
        return {**shared, "dynamics_loss_weight": 1.0}
    if algo == "latent_model":
        return {**shared, "latent_dim": 8, "dynamics_loss_weight": 1.0, "target_encoder_decay": 0.99}
    return {**shared, "latent_dim": 8, "latent_loss_weight": 1.0, "reconstruction_loss_weight": 1.0}


class _UniformPolicy:
    def action_probabilities(self, obs, info, deterministic=True):
        del obs, deterministic
        mask = np.asarray(info["action_mask"], dtype=np.float64)
        return mask / mask.sum()


class TestEpiCareIntegration(unittest.TestCase):
    def test_reset_seed_reproduces_episode(self):
        config = {"env_id": "epicare", "environment_seed": 3, "obs_backend": "state_vector"}
        env = make_env(config)
        obs_a, _ = env.reset(seed=21)
        step_a = env.step(2)
        obs_b, _ = env.reset(seed=21)
        step_b = env.step(2)
        np.testing.assert_array_equal(obs_a, obs_b)
        self.assertEqual(step_a[1:4], step_b[1:4])
        np.testing.assert_array_equal(step_a[0], step_b[0])
        env.close()

    def test_dataset_schema_and_ope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = generate_dataset(_dataset_config(root))
            data = np.load(root / "train.npz")
            self.assertTrue(REQUIRED_ARRAYS.issubset(data.files))
            self.assertEqual(data["observations"].dtype, np.float32)
            self.assertTrue(np.isfinite(data["behavior_logprobs"]).all())
            self.assertTrue((root / "ope_test.npz").exists())
            self.assertEqual(metadata["environment"]["env_id"], "epicare")
            summary_a, bootstrap_a = evaluate_ope(_UniformPolicy(), dataset_path=root / "ope_test.npz", obs_backend="state_vector", bootstrap_replicates=4, seed=4)
            summary_b, bootstrap_b = evaluate_ope(_UniformPolicy(), dataset_path=root / "ope_test.npz", obs_backend="state_vector", bootstrap_replicates=4, seed=4)
            self.assertEqual(summary_a, summary_b)
            pd.testing.assert_frame_equal(bootstrap_a, bootstrap_b)
            self.assertEqual(set(summary_a["estimators"]), {"IS", "WIS", "PDIS", "WPDIS"})

    def test_wpdis_matches_on_policy_return_for_variable_length_epicare_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_dataset(_dataset_config(root))
            data = np.load(root / "ope_test.npz")
            logged_returns = [
                float(data["rewards"][data["episode_ids"] == episode_id].sum())
                for episode_id in np.unique(data["episode_ids"])
            ]
            config = {"env_id": "epicare", "environment_seed": 1, "obs_backend": "state_vector"}
            env = make_env(config)
            try:
                smart = make_reference_policy(env, "smart", seed=0)
                summary, _ = evaluate_ope(
                    smart,
                    dataset_path=root / "ope_test.npz",
                    obs_backend="state_vector",
                    bootstrap_replicates=2,
                    seed=0,
                    deterministic=False,
                )
            finally:
                env.close()
            for estimator in summary["estimators"].values():
                self.assertAlmostEqual(estimator["estimate"], float(np.mean(logged_returns)), places=4)
            self.assertEqual(summary["reliability"]["status"], "limited")
            self.assertGreater(summary["reliability"]["trajectory_ess_fraction"], 0.99)
            self.assertEqual(summary["policy_probability_contract"]["status"], "exact")

    def test_wpdis_pads_terminated_trajectories_with_zero_reward(self):
        estimates, _ = _estimate(
            [
                {"rewards": np.asarray([1.0, 2.0]), "ratios": np.asarray([1.0, 1.0])},
                {"rewards": np.asarray([3.0]), "ratios": np.asarray([1.0])},
            ],
            gamma=1.0,
        )
        self.assertEqual(estimates, {"IS": 3.0, "WIS": 3.0, "PDIS": 3.0, "WPDIS": 3.0})

    def test_all_offline_algorithms_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            generate_dataset(_dataset_config(data_dir))
            for algo in ["dqn", "raw_model", "latent_model", "reconstructive_model"]:
                with self.subTest(algo=algo):
                    result = get_offline_trainer(algo)(_offline_config(data_dir, algo), Path(tmp) / algo)
                    self.assertIn("checkpoint_best", result)
                    self.assertIn("metrics", result)
                    self.assertIn("eval_mean_success", result["metrics"]["offline_eval_history"])
                    self.assertIn("eval_mean_symptom_burden", result["metrics"]["offline_eval_history"])

    def test_generic_report_for_epicare_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "artifacts" / "runs" / "epicare" / "smart" / "dev" / "offline" / "dqn" / "seed-0--test"
            run_dir.mkdir(parents=True)
            (run_dir / "run_manifest.json").write_text(json.dumps({"run_id": "epicare:smart:dqn", "status": "completed", "dataset_id": "epicare/smart/dev", "method_id": "dqn", "environment": "epicare", "seed": 0}), encoding="utf-8")
            (run_dir / "metrics_summary.json").write_text(json.dumps({"train_summary": {"best_validation_loss": 0.1}}), encoding="utf-8")
            (run_dir / "eval_summary.json").write_text(json.dumps({"mean_return": 1.0, "mean_remission_rate": 0.5}), encoding="utf-8")
            (run_dir / "ope_summary.json").write_text(json.dumps({"estimators": {"WIS": {"estimate": 0.2}}, "diagnostics": {"trajectory_ess": 4.0}}), encoding="utf-8")
            report_dir, summary = build_run_report(run_dir, root=root / "reports")
            self.assertTrue((report_dir / "summary.json").exists())
            self.assertIn("optimization", summary)
            self.assertIn("direct_evaluation", summary)
            self.assertIn("ope", summary)


if __name__ == "__main__":
    unittest.main()
