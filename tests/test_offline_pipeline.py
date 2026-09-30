import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from benchtwm.offline.dataset import REQUIRED_ARRAYS, OfflineTransitionDataset
from benchtwm.offline.generate_dataset import generate_dataset
from benchtwm.offline.trainers import _training_batches, get_offline_trainer
from benchtwm.operations.artifacts import initial_manifest
from benchtwm.run_experiment import run_experiment


def _dataset_config(output_dir: Path, policy: str = "random_admissible", seed: int = 5) -> dict:
    config = {
        "dataset_name": f"tiny_{policy}",
        "output_dir": str(output_dir),
        "behavior_policy": policy,
        "seed": seed,
        "num_train_episodes": 4,
        "num_val_episodes": 2,
        "gamma": 1.0,
        "optimal_value_iteration_steps": 2,
        "optimal_value_iteration_delta": 1e-3,
        "overwrite": True,
    }
    if policy == "epsilon_expert":
        config["epsilon"] = 0.3
    return config


def _offline_common(ds_dir: Path, algo: str, obs_backend: str = "state_vector") -> dict:
    return {
        "algo": algo,
        "train_mode": "offline",
        "seed": 0,
        "obs_backend": obs_backend,
        "dataset_path": str(ds_dir / "train.npz"),
        "val_dataset_path": str(ds_dir / "val.npz"),
        "offline_train_steps": 2,
        "offline_batch_size": 4,
        "offline_eval_interval": 1,
        "offline_log_interval": 1,
        "offline_eval_episodes": 1,
        "eval_episodes": 1,
        "eval_seed_base": 100000,
        "save_checkpoints": True,
        "save_trajectories": False,
        "success_threshold": 0.85,
        "gamma": 1.0,
        "use_tensorboard": False,
        "track": False,
        "show_progress": False,
    }


def _tiny_offline_config(ds_dir: Path, algo: str, obs_backend: str = "state_vector") -> dict:
    common = _offline_common(ds_dir, algo, obs_backend=obs_backend)
    if algo == "dqn":
        return {**common, "learning_rate": 0.01, "tau": 0.5, "target_network_frequency": 1}
    if algo in {"raw_model", "latent_model", "reconstructive_model"}:
        config = {
            **common,
            "learning_rate": 0.001,
            "hidden_dim": 32,
            "action_embed_dim": 8,
            "cem_horizon": 2,
            "cem_samples": 4,
            "cem_elites": 2,
            "cem_iterations": 1,
            "cem_alpha": 0.2,
            "reward_loss_weight": 1.0,
            "continue_loss_weight": 1.0,
        }
        if algo == "raw_model":
            return {
                **config,
                "dynamics_loss_weight": 1.0,
            }
        if algo == "latent_model":
            return {
                **config,
                "latent_dim": 16,
                "dynamics_loss_weight": 1.0,
                "target_encoder_decay": 0.99,
            }
        return {
            **config,
            "latent_dim": 16,
            "latent_loss_weight": 1.0,
            "reconstruction_loss_weight": 1.0,
        }
    raise KeyError(algo)


class TestOfflinePipeline(unittest.TestCase):
    def test_dataset_generation_is_deterministic_and_schema_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "ds1"
            out2 = Path(tmp) / "ds2"
            generate_dataset(_dataset_config(out1, seed=13))
            generate_dataset(_dataset_config(out2, seed=13))
            data1 = np.load(out1 / "train.npz")
            data2 = np.load(out2 / "train.npz")
            self.assertTrue(REQUIRED_ARRAYS.issubset(set(data1.files)))
            for key in REQUIRED_ARRAYS:
                np.testing.assert_array_equal(data1[key], data2[key])
            train_episode_ids = set(np.load(out1 / "train.npz")["episode_ids"].tolist())
            val_episode_ids = set(np.load(out1 / "val.npz")["episode_ids"].tolist())
            self.assertTrue(train_episode_ids.isdisjoint(val_episode_ids))

    def test_behavior_policies_generate_finite_admissible_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for policy in ["random_admissible", "expert", "optimal", "epsilon_expert"]:
                with self.subTest(policy=policy):
                    out = Path(tmp) / policy
                    generate_dataset(_dataset_config(out, policy=policy, seed=17))
                    data = np.load(out / "train.npz")
                    self.assertTrue(np.isfinite(data["behavior_logprobs"]).all())
                    for action, mask in zip(data["actions"], data["action_masks"]):
                        self.assertEqual(int(mask[int(action)]), 1)

    def test_offline_dataset_state_vector_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ds"
            generate_dataset(_dataset_config(out))
            ds = OfflineTransitionDataset(out / "train.npz", obs_backend="state_vector", seed=0)
            batch = ds.sample(3)
            self.assertEqual(batch.obs.shape[0], 3)
            self.assertEqual(batch.obs.shape[1], 47)
            episode_batch = ds.episode_batch(0, ds.num_episodes + 1, shuffle=False)
            self.assertGreaterEqual(episode_batch.action.shape[0], ds.size)

    def test_transition_and_sequence_iterators_cover_expected_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ds"
            generate_dataset(_dataset_config(out))
            ds = OfflineTransitionDataset(out / "train.npz", obs_backend="state_vector", seed=0)
            transition_total = 0
            for batch in ds.iter_transition_batches(7):
                transition_total += int(batch.action.shape[0])
            self.assertEqual(transition_total, ds.size)

            horizon = 3
            sequence_total = 0
            for batch in ds.iter_sequence_batches(horizon, 5):
                self.assertEqual(batch.actions.shape[1], horizon)
                self.assertEqual(batch.next_obs.shape[1], horizon)
                sequence_total += int(batch.actions.shape[0])
            expected_sequences = sum(max(0, len(indices) - horizon + 1) for indices in ds._episode_indices)
            self.assertEqual(sequence_total, expected_sequences)

    def test_all_offline_trainers_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = Path(tmp) / "ds"
            generate_dataset(_dataset_config(ds_dir, policy="epsilon_expert"))
            variants = [
                ("dqn", "state_vector"),
                ("raw_model", "state_vector"),
                ("latent_model", "state_vector"),
                ("reconstructive_model", "state_vector"),
            ]
            for algo, obs_backend in variants:
                with self.subTest(algo=algo, obs_backend=obs_backend):
                    result = get_offline_trainer(algo)(_tiny_offline_config(ds_dir, algo, obs_backend), Path(tmp))
                    self.assertIn("metrics", result)
                    self.assertEqual(len(result["metrics"]["returns"]), 2)
                    self.assertIn("checkpoint_best", result)

    def test_early_stopping_triggers_from_validation_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = Path(tmp) / "ds"
            generate_dataset(_dataset_config(ds_dir))
            config = _tiny_offline_config(ds_dir, "dqn")
            config.update(
                {
                    "learning_rate": 0.0,
                    "offline_train_steps": 5,
                    "offline_log_interval": 1,
                    "offline_eval_interval": 2,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
                }
            )
            result = get_offline_trainer("dqn")(config, Path(tmp))
            summary = result["metrics"]["summary"]
            self.assertTrue(summary["early_stopped"])
            self.assertEqual(summary["best_validation_step"], 1)
            self.assertEqual(summary["early_stop_step"], 2)
            self.assertEqual(summary["num_validation_checks"], 2)
            self.assertEqual(len(result["metrics"]["returns"]), 1)

    def test_early_exit_closes_offline_progress_before_final_evaluation(self):
        args = SimpleNamespace(
            offline_train_steps=2,
            offline_batch_size=4,
            offline_log_interval=1,
            offline_eval_interval=1,
            show_progress=True,
        )
        dataset = Mock()
        dataset.sample.return_value = object()
        bar = Mock()
        bar.__iter__ = Mock(return_value=iter([1, 2]))
        with patch("benchtwm.offline.trainers.tqdm", return_value=bar):
            batches = _training_batches(args, dataset, device="cpu", desc="test")
            next(batches)
            batches.close()
        bar.close.assert_called_once()

    def test_color_wm_rollout_metrics_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = Path(tmp) / "ds"
            generate_dataset(_dataset_config(ds_dir, policy="epsilon_expert"))
            config = _tiny_offline_config(ds_dir, "reconstructive_model", obs_backend="state_vector")
            config["cem_horizon"] = 5
            result = get_offline_trainer("reconstructive_model")(config, Path(tmp))
            extra = result["metrics"]["extra_metrics"]
            self.assertIn("validation_rollout1_state_error", extra)
            self.assertIn("validation_rollout5_discounted_return_error", extra)
            self.assertIn("validation_rollout5_cumulative_state_error", extra)

    def test_runner_writes_offline_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = Path(tmp) / "ds"
            generate_dataset(_dataset_config(ds_dir))
            config = _tiny_offline_config(ds_dir, "dqn")
            config["save_trajectories"] = True
            run_dir = Path(tmp) / "run"
            manifest = initial_manifest(
                {**config, "env_id": "icu_sepsis"},
                dataset_id="test/icu",
                method_id="dqn",
                run_dir=run_dir,
            )
            run_dir = run_experiment(config, force_run=True, output_dir=run_dir, manifest=manifest, quiet=True)
            self.assertEqual(run_dir.name, "run")
            self.assertTrue((run_dir / "metrics.pkl").exists())
            self.assertTrue((run_dir / "offline_training_history.csv").exists())
            self.assertTrue((run_dir / "offline_eval_history.csv").exists())
            self.assertIn("step", (run_dir / "offline_training_history.csv").read_text())
            with open(run_dir / "metrics_summary.json", "r", encoding="utf-8") as handle:
                summary = json.load(handle)["train_summary"]
            self.assertIn("total_sampled_transitions", summary)
            self.assertIn("effective_epochs", summary)
            self.assertTrue((run_dir / "checkpoint_best.pt").exists())
            self.assertTrue((run_dir / "episodes_summary.csv").exists())
            self.assertTrue((run_dir / "trajectories.csv").exists())
            with (run_dir / "run_manifest.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
