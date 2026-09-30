import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchtwm.algorithms.dreamer import DreamerPolicy, load_policy
from benchtwm.envs import make_env
from benchtwm.models.dreamer import DreamerAgent, DreamerConfig
from benchtwm.data.dataset import OfflineTransitionDataset
from benchtwm.data.sequences import DreamerSequenceBatch, DreamerSequenceStream
from benchtwm.ope import evaluate_ope
from benchtwm.offline.trainers import get_offline_trainer
from benchtwm.operations.catalog import Catalog
from benchtwm.run_experiment import run_experiment


def _write_dataset(path: Path, lengths=(5, 2), obs_dim=8, num_actions=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(lengths)
    observations = np.arange(total * obs_dim, dtype=np.float32).reshape(total, obs_dim) / 10.0
    next_observations = observations + 0.1
    actions = np.arange(total, dtype=np.int64) % num_actions
    episode_ids = np.concatenate([np.full(length, index, dtype=np.int64) for index, length in enumerate(lengths)])
    timesteps = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    terminals = np.concatenate([
        np.asarray([False] * (length - 1) + [True], dtype=bool) for length in lengths
    ])
    next_actions = np.full(total, -1, dtype=np.int64)
    cursor = 0
    for length in lengths:
        next_actions[cursor : cursor + length - 1] = actions[cursor + 1 : cursor + length]
        cursor += length
    masks = np.ones((total, num_actions), dtype=np.float32)
    next_masks = masks.copy()
    rewards = np.linspace(-1.0, 1.0, total, dtype=np.float32)
    np.savez_compressed(
        path,
        observations=observations,
        next_observations=next_observations,
        state_vectors=observations,
        next_state_vectors=next_observations,
        actions=actions,
        next_actions=next_actions,
        rewards=rewards,
        terminals=terminals,
        state_ids=np.arange(total, dtype=np.int64) + 10000,
        next_state_ids=np.arange(total, dtype=np.int64) + 20000,
        action_masks=masks,
        next_action_masks=next_masks,
        behavior_logprobs=np.full(total, -np.log(num_actions), dtype=np.float32),
        behavior_policy_ids=np.zeros(total, dtype=np.int64),
        episode_ids=episode_ids,
        timesteps=timesteps,
    )


def _tiny_config(train_path: Path, val_path: Path):
    return {
        "algo": "dreamer",
        "method_id": "dreamer",
        "train_mode": "offline",
        "track": "provisional_planning",
        "obs_backend": "state_vector",
        "dataset_path": str(train_path),
        "val_dataset_path": str(val_path),
        "offline_train_steps": 1,
        "offline_batch_size": 2,
        "dreamer_batch_length": 4,
        "offline_log_interval": 1,
        "offline_eval_interval": 0,
        "offline_eval_during_training": False,
        "offline_eval_episodes": 1,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.0,
        "learning_rate": 4e-5,
        "dreamer_deter": 16,
        "dreamer_hidden": 8,
        "dreamer_stoch": 2,
        "dreamer_classes": 4,
        "dreamer_blocks": 4,
        "dreamer_units": 8,
        "dreamer_enc_layers": 1,
        "dreamer_dec_layers": 1,
        "dreamer_rssm_img_layers": 1,
        "dreamer_rssm_obs_layers": 1,
        "dreamer_rssm_dyn_layers": 1,
        "dreamer_bins": 5,
        "dreamer_imag_length": 2,
        "dreamer_warmup": 0,
        "eval_episodes": 1,
        "eval_seed_base": 1000,
        "gamma": 1.0,
        "success_threshold": 0.0,
        "save_checkpoints": True,
        "save_trajectories": False,
        "use_tensorboard": False,
        "show_progress": False,
        "seed": 0,
    }


class _RecurrentUniformPolicy:
    def __init__(self):
        self.resets = 0
        self.observations = []

    def reset_episode(self):
        self.resets += 1

    def action_probabilities(self, obs, info, deterministic=False):
        del obs, deterministic
        mask = np.asarray(info["action_mask"], dtype=np.float64)
        return mask / mask.sum()

    def observe(self, action, reward):
        self.observations.append((int(action), float(reward)))


class TestDreamerIntegration(unittest.TestCase):
    def test_sequence_stream_preserves_long_episode_carry_and_real_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.npz"
            _write_dataset(path)
            dataset = OfflineTransitionDataset(path, obs_backend="state_vector", seed=0)
            stream = DreamerSequenceStream(dataset, batch_size=1, sequence_length=4, shuffle=False, cycle=False)
            first = stream.next_batch()
            second = stream.next_batch()
            third = stream.next_batch()
            self.assertIsInstance(first, DreamerSequenceBatch)
            self.assertEqual(first.valid_records, 4)
            self.assertEqual(first.is_first[0].tolist(), [True, False, False, False])
            # The second chunk continues episode zero, then starts episode one.
            self.assertEqual(second.is_first[0].tolist(), [False, False, True, False])
            self.assertFalse(bool(second.is_first[0, 0]))
            self.assertTrue(bool(second.is_last[0, 1]))
            self.assertEqual(third.valid_records, 1)
            self.assertTrue(bool(third.is_last[0, 0]))
            self.assertIsNone(stream.next_batch())
            self.assertFalse(hasattr(first, "state_id"))

    def test_policy_masks_actions_and_resets_recurrent_state(self):
        config = DreamerConfig(
            obs_dim=3,
            num_actions=5,
            deter=16,
            hidden=8,
            stoch=2,
            classes=4,
            blocks=4,
            units=8,
            enc_layers=1,
            dec_layers=1,
            rssm_img_layers=1,
            bins=5,
            imag_length=2,
        )
        policy = DreamerPolicy(DreamerAgent(config))
        mask = np.asarray([0, 0, 1, 0, 0], dtype=np.float32)
        probabilities = policy.action_probabilities(np.zeros(3, dtype=np.float32), {"action_mask": mask})
        np.testing.assert_array_equal(probabilities, mask)
        policy.observe(2, 1.0)
        policy.reset_episode()
        self.assertTrue(policy._is_first)
        self.assertEqual(int(policy._previous_action.item()), 0)

    def test_recurrent_ope_invokes_episode_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ope_test.npz"
            _write_dataset(path)
            policy = _RecurrentUniformPolicy()
            summary, bootstrap = evaluate_ope(
                policy,
                dataset_path=path,
                obs_backend="state_vector",
                bootstrap_replicates=2,
                deterministic=False,
                show_progress=False,
            )
            self.assertEqual(policy.resets, 2)
            self.assertEqual(len(policy.observations), 7)
            self.assertEqual(summary["diagnostics"]["target_logged_action_support_rate"], 1.0)
            self.assertEqual(len(bootstrap), 2)

    def test_tiny_offline_training_checkpoint_and_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.npz"
            val_path = Path(tmp) / "val.npz"
            _write_dataset(train_path)
            _write_dataset(val_path, lengths=(3, 2))
            config = _tiny_config(train_path, val_path)
            result = get_offline_trainer("dreamer")(config, Path(tmp))
            self.assertIn("checkpoint_best", result)
            self.assertEqual(result["checkpoint_best"]["checkpoint_format"], "benchtwm-dreamer-v1")
            self.assertTrue(np.isfinite(result["metrics"]["summary"]["best_validation_loss"]))
            load_config = {
                **config,
                "env_id": "epicare",
                "env_kwargs": {"patient_modifiers": True},
            }
            policy = load_policy(load_config, result["checkpoint_best"])
            env = make_env(load_config, seed=0)
            try:
                obs, info = env.reset(seed=0)
                action = policy.act(obs, info, deterministic=True)
                self.assertIn(action, info["admissible_actions"])
            finally:
                env.close()

    def test_runner_performs_direct_evaluation_and_recurrent_ope(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.npz"
            val_path = Path(tmp) / "val.npz"
            ope_path = Path(tmp) / "ope_test.npz"
            _write_dataset(train_path)
            _write_dataset(val_path, lengths=(3, 2))
            _write_dataset(ope_path, lengths=(2, 2))
            config = {
                **_tiny_config(train_path, val_path),
                "env_id": "epicare",
                "env_kwargs": {"patient_modifiers": True},
                "environment_seed": 1,
                "ope_enabled": True,
                "ope_dataset_path": str(ope_path),
                "ope_bootstrap_replicates": 2,
                "ope_deterministic": False,
                "eval_deterministic": True,
                "save_trajectories": True,
            }
            run_dir = run_experiment(
                config,
                force_run=True,
                output_dir=Path(tmp) / "run",
                quiet=True,
            )
            for filename in (
                "checkpoint_best.pt",
                "offline_training_history.csv",
                "eval_summary.json",
                "episodes_summary.csv",
                "ope_summary.json",
                "ope_bootstrap.csv",
            ):
                self.assertTrue((run_dir / filename).exists(), filename)

    def test_catalog_and_policy_contract_cover_all_environments(self):
        catalog = Catalog.load()
        datasets = [
            "epicare/soc/dev",
            "icu-sepsis/expert/dev",
        ]
        for dataset_id in datasets:
            self.assertIn("dreamer", [item.method_id for item in catalog.compatible_methods(dataset_id)])

        env_configs = [
            {"env_id": "icu_sepsis", "env_kwargs": {}},
            {"env_id": "epicare", "env_kwargs": {"patient_modifiers": True}},
        ]
        for raw in env_configs:
            with self.subTest(env_id=raw["env_id"]):
                env = make_env(raw, seed=0)
                try:
                    cfg = DreamerConfig(
                        obs_dim=env.obs_dim,
                        num_actions=env.num_actions,
                        deter=16,
                        hidden=8,
                        stoch=2,
                        classes=4,
                        blocks=4,
                        units=8,
                        enc_layers=1,
                        dec_layers=1,
                        rssm_img_layers=1,
                        bins=5,
                        imag_length=2,
                    )
                    policy = DreamerPolicy(DreamerAgent(cfg))
                    obs, info = env.reset(seed=0)
                    self.assertEqual(np.asarray(obs).shape, (env.obs_dim,))
                    action = policy.act(obs, info, deterministic=True)
                    self.assertIn(action, info["admissible_actions"])
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
