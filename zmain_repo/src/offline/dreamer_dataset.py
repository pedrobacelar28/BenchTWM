"""Episode-preserving sequence streams for the offline Dreamer trainer.

The transition schema stores ``(obs_t, action_t, reward_t, obs_{t+1})``.
Dreamer consumes state records whose reward belongs to the preceding action.
This module reconstructs that representation without exposing ``state_id`` or
changing the shared :class:`OfflineTransitionDataset` contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.offline.dataset import OfflineTransitionDataset


@dataclass
class DreamerSequenceBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    is_first: torch.Tensor
    is_last: torch.Tensor
    is_terminal: torch.Tensor
    valid_mask: torch.Tensor
    action_masks: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.observations.shape[1])

    @property
    def valid_records(self) -> int:
        return int(self.valid_mask.sum().item())


class DreamerSequenceStream:
    """Fixed-width recurrent stream that never invents episode boundaries.

    Each batch lane advances through complete episodes.  When a sequence ends
    in the middle of a long episode, the next call continues that episode and
    the trainer supplies the detached recurrent carry from the previous call.
    Short episodes are packed into the same sequence and marked with
    ``is_first`` so the RSSM resets at the real boundary.
    """

    def __init__(
        self,
        dataset: OfflineTransitionDataset,
        *,
        batch_size: int,
        sequence_length: int,
        seed: int = 0,
        shuffle: bool = True,
        cycle: bool = True,
    ):
        if dataset.obs_backend != "state_vector":
            raise ValueError("DreamerSequenceStream requires obs_backend='state_vector'")
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.cycle = bool(cycle)
        self.num_actions = dataset.num_actions
        self.obs_dim = dataset.obs_dim
        self.num_state_records = int(sum(len(indices) + 1 for indices in dataset._episode_indices))
        self._episodes = [
            indices[np.argsort(dataset.data["timesteps"][indices], kind="stable")]
            for indices in dataset._episode_indices
        ]
        self.reset()

    def reset(self) -> None:
        self._epoch = 0
        self._position = 0
        self._order = self._make_order(self._epoch)
        self._slots: list[tuple[np.ndarray, int] | None] = [self._take_episode() for _ in range(self.batch_size)]

    def _make_order(self, epoch: int) -> np.ndarray:
        order = np.arange(len(self._episodes), dtype=np.int64)
        if self.shuffle:
            order = np.random.default_rng(self.seed + 1009 * epoch).permutation(order)
        return order

    def _take_episode(self) -> tuple[np.ndarray, int] | None:
        if self._position >= len(self._order):
            if not self.cycle:
                return None
            self._epoch += 1
            self._position = 0
            self._order = self._make_order(self._epoch)
        episode = self._episodes[int(self._order[self._position])]
        self._position += 1
        return episode, 0

    @property
    def exhausted(self) -> bool:
        return all(slot is None for slot in self._slots)

    def _record(self, indices: np.ndarray, cursor: int) -> tuple[np.ndarray, int, float, bool, bool, bool, np.ndarray]:
        data = self.dataset.data
        transitions = len(indices)
        if cursor == 0:
            first = int(indices[0])
            return (
                np.asarray(self.dataset._obs_array("obs")[first], dtype=np.float32),
                int(data["actions"][first]),
                0.0,
                True,
                False,
                False,
                np.asarray(data["action_masks"][first], dtype=np.float32),
            )

        previous = int(indices[cursor - 1])
        final = cursor == transitions
        if final:
            action = 0  # No action follows the terminal observation; reset masks it before reuse.
            action_mask = np.asarray(data["next_action_masks"][previous], dtype=np.float32)
        else:
            current = int(indices[cursor])
            action = int(data["actions"][current])
            action_mask = np.asarray(data["action_masks"][current], dtype=np.float32)
        terminal = bool(data["terminals"][previous]) if final else False
        return (
            np.asarray(self.dataset._obs_array("next_obs")[previous], dtype=np.float32),
            action,
            float(data["rewards"][previous]),
            False,
            final,
            terminal,
            action_mask,
        )

    def next_batch(self, *, device: str | torch.device = "cpu") -> DreamerSequenceBatch | None:
        if self.exhausted:
            return None
        B, T, D, A = self.batch_size, self.sequence_length, self.obs_dim, self.num_actions
        obs = np.zeros((B, T, D), dtype=np.float32)
        actions = np.zeros((B, T), dtype=np.int64)
        rewards = np.zeros((B, T), dtype=np.float32)
        is_first = np.ones((B, T), dtype=bool)
        is_last = np.zeros((B, T), dtype=bool)
        is_terminal = np.zeros((B, T), dtype=bool)
        valid = np.zeros((B, T), dtype=np.float32)
        action_masks = np.ones((B, T, A), dtype=np.float32)

        for lane in range(B):
            for time in range(T):
                slot = self._slots[lane]
                if slot is None:
                    break
                indices, cursor = slot
                record = self._record(indices, cursor)
                obs[lane, time], actions[lane, time], rewards[lane, time] = record[:3]
                is_first[lane, time], is_last[lane, time], is_terminal[lane, time] = record[3:6]
                action_masks[lane, time] = record[6]
                valid[lane, time] = 1.0
                cursor += 1
                self._slots[lane] = self._take_episode() if cursor > len(indices) else (indices, cursor)

        if not valid.any():
            return None
        return DreamerSequenceBatch(
            observations=torch.as_tensor(obs, dtype=torch.float32, device=device),
            actions=torch.as_tensor(actions, dtype=torch.int64, device=device),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
            is_first=torch.as_tensor(is_first, dtype=torch.bool, device=device),
            is_last=torch.as_tensor(is_last, dtype=torch.bool, device=device),
            is_terminal=torch.as_tensor(is_terminal, dtype=torch.bool, device=device),
            valid_mask=torch.as_tensor(valid, dtype=torch.float32, device=device),
            action_masks=torch.as_tensor(action_masks, dtype=torch.float32, device=device),
        )

