import os
import sys
import unittest
from pathlib import Path

from benchtwm.envs.icu_adapter import ICUSepsisEnvWrapper


class TestEnvWrapper(unittest.TestCase):
    def test_state_vector_backend_shape(self):
        env = ICUSepsisEnvWrapper(seed=0)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape[0], env.spec.state_vector_dim)
        self.assertEqual(info["state_id"], env.last_state_id)
        self.assertEqual(info["action_mask"].shape[0], env.num_actions)
        self.assertEqual(env.decode_action(24), (4, 4))
        env.close()


if __name__ == "__main__":
    unittest.main()
