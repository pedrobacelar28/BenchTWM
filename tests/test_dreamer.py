"""Dreamer integration: sequence streams, recurrent policy, and checkpoints."""

from __future__ import annotations

import numpy as np
import torch

from benchtwm.algorithms import DataSpec, get_algorithm, load_policy
from benchtwm.algorithms.dreamer import CHECKPOINT_FORMAT, DreamerPolicy
from benchtwm.algorithms.dreamer.agent import DreamerAgent, DreamerConfig
from benchtwm.data.dataset import OfflineTransitionDataset
from benchtwm.data.sequences import DreamerSequenceBatch, DreamerSequenceStream
from benchtwm.envs import make_env

from conftest import TINY_DATASETS, tiny_run_config


def _write_dataset(path, lengths=(5, 2), obs_dim=8, num_actions=16):
    total = sum(lengths)
    observations = np.arange(total * obs_dim, dtype=np.float32).reshape(total, obs_dim) / 10.0
    actions = np.arange(total, dtype=np.int64) % num_actions
    next_actions = np.full(total, -1, dtype=np.int64)
    cursor = 0
    for length in lengths:
        next_actions[cursor : cursor + length - 1] = actions[cursor + 1 : cursor + length]
        cursor += length
    masks = np.ones((total, num_actions), dtype=np.float32)
    np.savez_compressed(
        path,
        observations=observations,
        next_observations=observations + 0.1,
        actions=actions,
        next_actions=next_actions,
        rewards=np.linspace(-1.0, 1.0, total, dtype=np.float32),
        terminals=np.concatenate([[False] * (length - 1) + [True] for length in lengths]),
        state_ids=np.arange(total, dtype=np.int64),
        next_state_ids=np.arange(total, dtype=np.int64) + 1,
        action_masks=masks,
        next_action_masks=masks.copy(),
        behavior_logprobs=np.full(total, -np.log(num_actions), dtype=np.float32),
        behavior_policy_ids=np.zeros(total, dtype=np.int64),
        episode_ids=np.concatenate([np.full(length, index, dtype=np.int64) for index, length in enumerate(lengths)]),
        timesteps=np.concatenate([np.arange(length, dtype=np.int64) for length in lengths]),
    )


def _tiny_agent(obs_dim=3, num_actions=5):
    return DreamerAgent(
        DreamerConfig(obs_dim=obs_dim, num_actions=num_actions, deter=16, hidden=8, stoch=2, classes=4, blocks=4, units=8, enc_layers=1, dec_layers=1, rssm_img_layers=1, bins=5, imag_length=2)
    )


def test_sequence_stream_carries_long_episodes_and_resets_at_real_boundaries(tmp_path):
    _write_dataset(tmp_path / "train.npz")
    stream = DreamerSequenceStream(OfflineTransitionDataset(tmp_path / "train.npz"), batch_size=1, sequence_length=4, shuffle=False, cycle=False)
    first, second, third = stream.next_batch(), stream.next_batch(), stream.next_batch()
    assert isinstance(first, DreamerSequenceBatch) and first.valid_records == 4
    assert first.is_first[0].tolist() == [True, False, False, False]
    # The second chunk continues episode zero, then starts episode one.
    assert second.is_first[0].tolist() == [False, False, True, False]
    assert bool(second.is_last[0, 1])
    assert third.valid_records == 1 and bool(third.is_last[0, 0])
    assert stream.next_batch() is None
    assert not hasattr(first, "state_id")


def test_policy_masks_actions_and_resets_recurrent_state():
    policy = DreamerPolicy(_tiny_agent())
    mask = np.asarray([0, 0, 1, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(policy.action_probabilities(np.zeros(3, dtype=np.float32), {"action_mask": mask}), mask)
    policy.observe(2, 1.0)
    policy.reset_episode()
    assert policy._is_first and int(policy._previous_action.item()) == 0


def test_checkpoint_round_trip_through_the_generic_loader(tmp_path):
    for split, lengths in (("train", (5, 2)), ("val", (3, 2))):
        _write_dataset(tmp_path / f"{split}.npz", lengths=lengths)
    config = tiny_run_config("dreamer", "epicare", tmp_path, offline_train_steps=1, offline_log_interval=1)
    algorithm = get_algorithm("dreamer")(config, DataSpec(obs_dim=8, num_actions=16), torch.device("cpu"))
    algorithm.update(next(algorithm.batches(OfflineTransitionDataset(tmp_path / "train.npz"))), 1)
    loss, _ = algorithm.validate(OfflineTransitionDataset(tmp_path / "val.npz"))
    checkpoint = algorithm.state_dict()
    assert np.isfinite(loss) and checkpoint["checkpoint_format"] == CHECKPOINT_FORMAT
    policy = load_policy(config, checkpoint)
    env = make_env(TINY_DATASETS["epicare"], seed=0)
    try:
        obs, info = env.reset(seed=0)
        assert info["action_mask"][policy.act(obs, info)] == 1
    finally:
        env.close()
