import tempfile
import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.algorithms.dqn import load_policy
from src.evaluation import evaluate_policy
from src.models.mlp import TabularQNetwork


class TestExports(unittest.TestCase):
    def test_clinical_export_is_deterministic(self):
        config = {
            "algo": "dqn",
            "seed": 0,
            "obs_backend": "discrete_id",
            "gamma": 1.0,
        }
        network = TabularQNetwork(716, 25)
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
