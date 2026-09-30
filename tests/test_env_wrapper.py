import unittest

from benchtwm.envs import make_env
from benchtwm.envs.icu_sepsis import FEATURE_NAMES, decode_action


class TestEnvWrapper(unittest.TestCase):
    def test_icu_state_vector_features_and_action_mask(self):
        env = make_env({"env_id": "icu_sepsis"}, seed=0)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape[0], env.spec.obs_dim)
        self.assertEqual(env.spec.feature_names, FEATURE_NAMES)
        self.assertEqual(info["action_mask"].shape[0], env.num_actions)
        self.assertEqual(decode_action(24), (4, 4))
        env.close()


if __name__ == "__main__":
    unittest.main()
