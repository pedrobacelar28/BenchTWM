import tempfile
import sys
import unittest
from pathlib import Path

import torch

from benchtwm.algorithms.dqn import load_policy
from benchtwm.evaluation import evaluate_policy
from benchtwm.models.mlp import build_q_network


class TestExports(unittest.TestCase):
    def test_clinical_export_is_deterministic(self):
        config = {
            "algo": "dqn",
            "env_id": "icu_sepsis",
            "seed": 0,
            "obs_backend": "state_vector",
            "gamma": 1.0,
            "policy_hidden_dim": 16,
            "policy_hidden_layers": 1,
        }
        torch.manual_seed(0)
        network = build_q_network(config, obs_dim=47, num_actions=25)
        checkpoint = {"model_state_dict": network.state_dict()}
        policy = load_policy(config, checkpoint)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result1 = evaluate_policy(policy, config={**config, "eval_episodes": 3, "eval_seed_base": 100000}, num_episodes=3, eval_seed_base=100000, output_dir=out_dir, export_csv=True)
            result2 = evaluate_policy(policy, config={**config, "eval_episodes": 3, "eval_seed_base": 100000}, num_episodes=3, eval_seed_base=100000, output_dir=out_dir, export_csv=True)
            self.assertEqual(result1["episodes"]["return"].tolist(), result2["episodes"]["return"].tolist())
            self.assertTrue((out_dir / "episodes_summary.csv").exists())
            self.assertTrue((out_dir / "trajectories.csv").exists())


if __name__ == "__main__":
    unittest.main()
