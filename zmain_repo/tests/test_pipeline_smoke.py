import tempfile
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.algorithms.registry import get_trainer
from src.run_experiment import run_experiment


SEED_BASE = {
    "eval_episodes": 2,
    "eval_seed_base": 100000,
    "save_checkpoints": True,
    "save_trajectories": False,
    "gamma": 1.0,
    "success_threshold": 0.8,
    "use_tensorboard": False,
    "track": False,
    "show_progress": False,
}


def tiny_config(algo: str, obs_backend: str | None = None):
    common = {
        **SEED_BASE,
        "algo": algo,
        "seed": 0,
        "max_episodes": 3,
    }
    if algo == "dqn":
        return {
            **common,
            "obs_backend": "discrete_id",
            "control_mode": "epsilon_greedy",
            "learning_rate": 0.001,
            "buffer_size": 32,
            "tau": 0.5,
            "target_network_frequency": 5,
            "batch_size": 2,
            "start_e": 1.0,
            "end_e": 0.1,
            "exploration_fraction": 0.5,
            "learning_starts": 1,
            "train_frequency": 1,
        }
    if algo == "ppo":
        return {
            **common,
            "obs_backend": "discrete_id",
            "control_mode": "policy_gradient",
            "learning_rate": 0.005,
            "num_steps": 8,
            "num_minibatches": 1,
            "gae_lambda": 0.95,
            "update_epochs": 2,
            "norm_adv": True,
            "clip_coef": 0.2,
            "clip_vloss": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.5,
        }
    if algo in {"raw_model", "latent_model", "reconstructive_model"}:
        if obs_backend is None:
            raise KeyError(f"{algo} requires obs_backend")
        config = {
            **common,
            "obs_backend": obs_backend,
            "control_mode": "mpc_cem",
            "learning_rate": 0.001,
            "hidden_dim": 32,
            "action_embed_dim": 8,
            "replay_size": 64,
            "batch_size": 2,
            "warmup_steps": 1,
            "updates_per_step": 1,
            "cem_horizon": 3,
            "cem_samples": 8,
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


class TestPipelineSmoke(unittest.TestCase):
    def test_all_trainers_smoke(self):
        variants = [
            ("dqn", None),
            ("ppo", None),
            ("raw_model", "discrete_id"),
            ("raw_model", "state_vector"),
            ("latent_model", "discrete_id"),
            ("latent_model", "state_vector"),
            ("reconstructive_model", "discrete_id"),
            ("reconstructive_model", "state_vector"),
        ]
        for algo, obs_backend in variants:
            with self.subTest(algo=algo, obs_backend=obs_backend):
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = get_trainer(algo)(tiny_config(algo, obs_backend), Path(tmpdir))
                    self.assertIn("metrics", result)
                    self.assertEqual(len(result["metrics"]["returns"]), 3)
                    self.assertIn("checkpoint_best", result)

    def test_runner_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = tiny_config("dqn")
            config["save_trajectories"] = True
            config["eval_episodes"] = 2
            run_dir = run_experiment(config, force_run=True)
            self.assertTrue((run_dir / "metrics.pkl").exists())
            self.assertTrue((run_dir / "metrics_summary.json").exists())
            self.assertTrue((run_dir / "metrics_summary.csv").exists())
            self.assertTrue((run_dir / "training_history.csv").exists())
            self.assertTrue((run_dir / "extra_metrics.csv").exists())
            self.assertTrue((run_dir / "eval_summary.json").exists())
            self.assertTrue((run_dir / "checkpoint_best.pt").exists())
            self.assertTrue((run_dir / "episodes_summary.csv").exists())
            self.assertTrue((run_dir / "trajectories.csv").exists())


if __name__ == "__main__":
    unittest.main()
