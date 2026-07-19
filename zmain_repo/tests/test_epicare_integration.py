import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.report_run import generate_report
from src.algorithms.registry import get_trainer
from src.envs.factory import make_env
from src.offline.dataset import REQUIRED_ARRAYS
from src.offline.generate_dataset import generate_dataset
from src.offline.ope import evaluate_ope
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
    if algo == "ppo":
        return {**config, "clip_coef": 0.2, "vf_coef": 0.5, "ent_coef": 0.01, "max_grad_norm": 0.5, "norm_adv": True, "offline_bc_weight": 0.1}
    shared = {**config, "hidden_dim": 16, "action_embed_dim": 4, "cem_horizon": 2, "cem_samples": 4, "cem_elites": 2, "cem_iterations": 1, "cem_alpha": 0.2, "reward_loss_weight": 1.0, "continue_loss_weight": 1.0}
    if algo == "raw_model":
        return {**shared, "dynamics_loss_weight": 1.0}
    if algo == "latent_model":
        return {**shared, "latent_dim": 8, "dynamics_loss_weight": 1.0, "target_encoder_decay": 0.99}
    return {**shared, "latent_dim": 8, "latent_loss_weight": 1.0, "reconstruction_loss_weight": 1.0}


def _online_config(algo: str) -> dict:
    config = {
        "env_id": "epicare",
        "environment_seed": 1,
        "obs_backend": "state_vector",
        "algo": algo,
        "seed": 0,
        "max_episodes": 1,
        "eval_episodes": 1,
        "eval_seed_base": 100,
        "learning_rate": 0.001,
        "gamma": 1.0,
        "success_threshold": 0.0,
        "show_progress": False,
        "use_tensorboard": False,
        "policy_architecture": "mlp",
        "policy_hidden_dim": 16,
        "policy_hidden_layers": 1,
    }
    if algo == "dqn":
        return {**config, "buffer_size": 32, "tau": 0.5, "target_network_frequency": 1, "batch_size": 4, "start_e": 1.0, "end_e": 0.1, "exploration_fraction": 0.5, "learning_starts": 100, "train_frequency": 1}
    if algo == "ppo":
        return {**config, "num_steps": 8, "num_minibatches": 1, "gae_lambda": 0.95, "update_epochs": 1, "norm_adv": True, "clip_coef": 0.2, "clip_vloss": False, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5}
    shared = {**config, "hidden_dim": 16, "action_embed_dim": 4, "replay_size": 32, "cem_horizon": 2, "cem_samples": 4, "cem_elites": 2, "cem_iterations": 1, "cem_alpha": 0.2, "warmup_steps": 100, "batch_size": 4, "updates_per_step": 1, "reward_loss_weight": 1.0, "continue_loss_weight": 1.0}
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

    def test_all_offline_algorithms_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            generate_dataset(_dataset_config(data_dir))
            for algo in ["dqn", "ppo", "raw_model", "latent_model", "reconstructive_model"]:
                with self.subTest(algo=algo):
                    result = get_offline_trainer(algo)(_offline_config(data_dir, algo), Path(tmp) / algo)
                    self.assertIn("checkpoint_best", result)
                    self.assertIn("metrics", result)
                    self.assertIn("eval_mean_success", result["metrics"]["offline_eval_history"])
                    self.assertIn("eval_mean_symptom_burden", result["metrics"]["offline_eval_history"])

    def test_all_online_algorithms_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            for algo in ["dqn", "ppo", "raw_model", "latent_model", "reconstructive_model"]:
                with self.subTest(algo=algo):
                    result = get_trainer(algo)(_online_config(algo), Path(tmp) / algo)
                    self.assertIn("checkpoint_best", result)

    def test_generic_report_for_epicare_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "config.json").write_text(json.dumps({"env_id": "epicare", "obs_backend": "state_vector", "seed": 0}), encoding="utf-8")
            with (output_dir / "metrics.pkl").open("wb") as handle:
                pickle.dump({}, handle)
            pd.DataFrame({"step": [1], "loss": [1.0]}).to_csv(output_dir / "offline_training_history.csv", index=False)
            pd.DataFrame({"step": [1], "eval_mean_return": [1.0]}).to_csv(output_dir / "offline_eval_history.csv", index=False)
            pd.DataFrame({"episode_id": [0], "return": [1.0], "success": [True], "length": [1], "action_stability_l1": [0.0], "abrupt_jump_rate": [0.0], "mean_symptom_burden": [0.4]}).to_csv(output_dir / "episodes_summary.csv", index=False)
            pd.DataFrame({"episode_id": [0], "t": [0], "action_idx": [0], "action_label": ["Treatment 0"], "reward": [1.0], "success": [True], "metric_symptom_burden": [0.4]}).to_csv(output_dir / "trajectories.csv", index=False)
            (output_dir / "environment_spec.json").write_text(json.dumps({"env_id": "epicare", "metric_names": ["symptom_burden"], "action_geometry": {"type": "categorical"}}), encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(json.dumps([{"dataset_tag": "tiny", "algorithm": "dqn", "method_id": "dqn_offline", "output_dir": str(output_dir), "resolved_config": {}}]), encoding="utf-8")
            report_dir = generate_report(run_dir, root / "report", ["png"])
            self.assertTrue((report_dir / "tables" / "generic_run_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
