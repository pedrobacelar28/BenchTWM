from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.envs.factory import make_env
from src.offline.generate_dataset import generate_dataset
from src.utils.io import load_json


class TestDTRIntegration(unittest.TestCase):
    def _config(self, env_id: str, *, max_t: int = 3) -> dict:
        return {
            "env_id": env_id,
            "obs_backend": "state_vector",
            "env_kwargs": {"setting": 1, "max_t": max_t},
            "environment_seed": 1,
        }

    def test_public_vectors_are_structured_and_hidden_state_is_sanitized(self):
        expected = {"dtr-ahn-chemo": 3, "dtr-ghaffari-cancer": 7}
        for env_id, dimension in expected.items():
            with self.subTest(env_id=env_id):
                env = make_env(self._config(env_id), seed=11)
                try:
                    observation, info = env.reset(seed=11)
                    self.assertEqual(observation.shape, (dimension,))
                    self.assertEqual(observation.dtype, np.float32)
                    self.assertEqual(info["state_vector"].shape, (dimension,))
                    self.assertEqual(info["state_vector"].dtype, np.float32)
                    self.assertNotIn("state", info)
                    self.assertEqual(info["action_mask"].shape, (env.num_actions,))
                    np.testing.assert_array_equal(info["action_mask"], np.ones(env.num_actions, dtype=np.float32))
                    self.assertNotEqual(dimension, env.num_actions)
                    self.assertGreater(np.count_nonzero(observation), 1)
                finally:
                    env.close()

    def test_action_geometries_cover_native_grids(self):
        ahn = make_env(self._config("dtr-ahn-chemo"), seed=0)
        try:
            np.testing.assert_allclose(
                [float(ahn._physical_action(index)[0]) for index in range(5)],
                np.linspace(0.0, 1.0, 5),
            )
            self.assertEqual(ahn.spec.action_geometry["action_to_dose"], [0.0, 0.25, 0.5, 0.75, 1.0])
        finally:
            ahn.close()

        ghaffari = make_env(self._config("dtr-ghaffari-cancer"), seed=0)
        try:
            self.assertEqual(ghaffari.num_actions, 25)
            np.testing.assert_allclose(ghaffari._physical_action(0), [0.0, 0.0])
            np.testing.assert_allclose(ghaffari._physical_action(4), [0.0, 8.0])
            np.testing.assert_allclose(ghaffari._physical_action(20), [10.0, 0.0])
            np.testing.assert_allclose(ghaffari._physical_action(24), [10.0, 8.0])
        finally:
            ghaffari.close()

    def test_same_seed_and_actions_are_deterministic(self):
        actions = [0, 4, 2]
        for env_id in ("dtr-ahn-chemo", "dtr-ghaffari-cancer"):
            with self.subTest(env_id=env_id):
                first = make_env(self._config(env_id), seed=17)
                second = make_env(self._config(env_id), seed=17)
                try:
                    obs_a, _ = first.reset(seed=17)
                    obs_b, _ = second.reset(seed=17)
                    np.testing.assert_array_equal(obs_a, obs_b)
                    for action in actions:
                        next_a, reward_a, term_a, trunc_a, _ = first.step(action)
                        next_b, reward_b, term_b, trunc_b, _ = second.step(action)
                        np.testing.assert_array_equal(next_a, next_b)
                        self.assertEqual(reward_a, reward_b)
                        self.assertEqual((term_a, trunc_a), (term_b, trunc_b))
                        if term_a or trunc_a:
                            break
                finally:
                    first.close()
                    second.close()

    def test_random_dataset_has_exact_finite_propensities_and_split_separation(self):
        for env_id, num_actions in (("dtr-ahn-chemo", 5), ("dtr-ghaffari-cancer", 25)):
            with self.subTest(env_id=env_id), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generate_dataset(
                    {
                        **self._config(env_id, max_t=2),
                        "output_dir": str(root),
                        "dataset_name": "tiny_dtr",
                        "behavior_policy": "random",
                        "seed": 19,
                        "num_train_episodes": 3,
                        "num_val_episodes": 2,
                        "num_ope_episodes": 2,
                        "ope_enabled": True,
                        "overwrite": True,
                        "show_progress": False,
                        "quiet": True,
                    }
                )
                train = np.load(root / "train.npz")
                validation = np.load(root / "val.npz")
                ope = np.load(root / "ope_test.npz")
                self.assertTrue(np.isfinite(train["behavior_logprobs"]).all())
                self.assertTrue(np.allclose(train["behavior_logprobs"], np.log(1.0 / num_actions)))
                self.assertEqual(train["state_vectors"].shape[1], 3 if num_actions == 5 else 7)
                self.assertTrue(set(train["episode_ids"]).isdisjoint(set(validation["episode_ids"])))
                self.assertTrue(set(train["episode_ids"]).isdisjoint(set(ope["episode_ids"])))
                self.assertEqual(train["state_ids"].dtype, np.int64)
                metadata = load_json(root / "metadata.json")
                self.assertEqual(metadata["runtime"]["source_commit"], "1f4fed49769b371c01d90959d2e012154068b3be")
                self.assertEqual(set(metadata["file_hashes"]), {"train.npz", "val.npz", "ope_test.npz"})


if __name__ == "__main__":
    unittest.main()
