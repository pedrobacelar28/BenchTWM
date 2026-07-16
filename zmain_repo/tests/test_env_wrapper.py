import os
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.envs.icu_adapter import ICUSepsisEnvWrapper


class TestEnvWrapper(unittest.TestCase):
    def test_discrete_backend_shape_and_action_decode(self):
        env = ICUSepsisEnvWrapper(obs_backend="discrete_id", seed=0)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape[0], env.num_states)
        self.assertEqual(info["action_mask"].shape[0], env.num_actions)
        fluid, vaso = env.decode_action(24)
        self.assertEqual((fluid, vaso), (4, 4))
        env.close()

    def test_state_vector_backend_shape(self):
        env = ICUSepsisEnvWrapper(obs_backend="state_vector", seed=0)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape[0], env.spec.state_vector_dim)
        self.assertEqual(info["state_id"], env.last_state_id)
        env.close()


if __name__ == "__main__":
    unittest.main()
