"""Offline transition dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.bootstrap import PROJECT_ROOT
from src.utils.io import load_json


REQUIRED_ARRAYS = {
    "observations",
    "next_observations",
    "actions",
    "next_actions",
    "rewards",
    "terminals",
    "state_ids",
    "next_state_ids",
    "action_masks",
    "next_action_masks",
    "behavior_logprobs",
    "behavior_policy_ids",
    "episode_ids",
    "timesteps",
}


@dataclass
class OfflineBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    action_mask: torch.Tensor
    next_action_mask: torch.Tensor
    state_id: torch.Tensor
    next_state_id: torch.Tensor
    next_action: torch.Tensor
    behavior_logprob: torch.Tensor


@dataclass
class OfflineSequenceBatch:
    start_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_obs: torch.Tensor
    next_state_id: torch.Tensor


def resolve_dataset_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return path


class OfflineTransitionDataset:
    def __init__(self, path: str | Path, *, obs_backend: str = "discrete_id", seed: int | None = None):
        self.path = resolve_dataset_path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if obs_backend not in {"discrete_id", "state_vector"}:
            raise ValueError(f"Unsupported obs_backend={obs_backend!r}")
        self.obs_backend = obs_backend
        metadata_path = self.path.parent / "metadata.json"
        self.metadata = load_json(metadata_path) if metadata_path.exists() else {}
        with np.load(self.path, allow_pickle=False) as payload:
            missing = sorted(REQUIRED_ARRAYS - set(payload.files))
            if missing:
                raise ValueError(f"Offline dataset {self.path} is missing required arrays: {missing}")
            self.data = {key: payload[key] for key in payload.files}
        if obs_backend == "state_vector" and "state_vectors" not in self.data:
            raise ValueError(f"Offline dataset {self.path} does not contain state_vectors for obs_backend='state_vector'")
        if obs_backend == "state_vector" and "next_state_vectors" not in self.data:
            raise ValueError(f"Offline dataset {self.path} does not contain next_state_vectors for obs_backend='state_vector'")
        self.size = int(len(self.data["actions"]))
        if self.size == 0:
            raise ValueError(f"Offline dataset {self.path} is empty")
        self.seed = int(seed or 0)
        self.rng = np.random.default_rng(seed)
        self.episode_ids = np.asarray(self.data["episode_ids"], dtype=np.int64)
        self.unique_episode_ids = np.unique(self.episode_ids)
        self._episode_indices = [np.flatnonzero(self.episode_ids == episode_id) for episode_id in self.unique_episode_ids]
        self._episode_permutations: dict[int, np.ndarray] = {}
        self._sequence_windows_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.size

    @property
    def num_episodes(self) -> int:
        return int(len(self.unique_episode_ids))

    @property
    def mean_episode_length(self) -> float:
        return float(self.size / max(1, self.num_episodes))

    @property
    def obs_dim(self) -> int:
        return int(self._obs_array("obs").shape[1])

    @property
    def feature_names(self) -> tuple[str, ...]:
        observation = self.metadata.get("observation", {})
        names = observation.get("feature_names")
        if names is None:
            names = self.metadata.get("environment", {}).get("feature_names", [])
        return tuple(str(name) for name in names)

    @property
    def num_actions(self) -> int:
        return int(self.data["action_masks"].shape[1])

    @property
    def num_states(self) -> int:
        return int(self.data["observations"].shape[1])

    def _obs_array(self, which: str) -> np.ndarray:
        if self.obs_backend == "state_vector":
            return self.data["state_vectors" if which == "obs" else "next_state_vectors"]
        return self.data["observations" if which == "obs" else "next_observations"]

    def _batch_from_indices(self, idxs: np.ndarray, device: str | torch.device = "cpu") -> OfflineBatch:
        obs = torch.tensor(self._obs_array("obs")[idxs], dtype=torch.float32, device=device)
        next_obs = torch.tensor(self._obs_array("next_obs")[idxs], dtype=torch.float32, device=device)
        return OfflineBatch(
            obs=obs,
            action=torch.tensor(self.data["actions"][idxs], dtype=torch.int64, device=device),
            reward=torch.tensor(self.data["rewards"][idxs], dtype=torch.float32, device=device),
            next_obs=next_obs,
            done=torch.tensor(self.data["terminals"][idxs].astype(np.float32), dtype=torch.float32, device=device),
            action_mask=torch.tensor(self.data["action_masks"][idxs], dtype=torch.float32, device=device),
            next_action_mask=torch.tensor(self.data["next_action_masks"][idxs], dtype=torch.float32, device=device),
            state_id=torch.tensor(self.data["state_ids"][idxs], dtype=torch.int64, device=device),
            next_state_id=torch.tensor(self.data["next_state_ids"][idxs], dtype=torch.int64, device=device),
            next_action=torch.tensor(self.data["next_actions"][idxs], dtype=torch.int64, device=device),
            behavior_logprob=torch.tensor(self.data["behavior_logprobs"][idxs], dtype=torch.float32, device=device),
        )

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> OfflineBatch:
        idxs = self.rng.integers(0, self.size, size=int(batch_size))
        return self._batch_from_indices(idxs, device=device)

    def iter_transition_batches(
        self,
        batch_size: int,
        *,
        device: str | torch.device = "cpu",
        shuffle: bool = False,
    ):
        idxs = np.arange(self.size, dtype=np.int64)
        if shuffle:
            idxs = self.rng.permutation(idxs)
        batch_size = int(batch_size)
        for start in range(0, self.size, batch_size):
            yield self._batch_from_indices(idxs[start : start + batch_size], device=device)

    def _permutation_for_epoch(self, epoch: int) -> np.ndarray:
        if epoch not in self._episode_permutations:
            rng = np.random.default_rng(self.seed + epoch * 1009)
            self._episode_permutations[epoch] = rng.permutation(self.num_episodes)
        return self._episode_permutations[epoch]

    def _episode_positions(self, start_episode: int, num_episodes: int, *, shuffle: bool) -> np.ndarray:
        positions = []
        for absolute_episode in range(int(start_episode), int(start_episode) + int(num_episodes)):
            epoch = absolute_episode // self.num_episodes
            offset = absolute_episode % self.num_episodes
            if shuffle:
                positions.append(int(self._permutation_for_epoch(epoch)[offset]))
            else:
                positions.append(int(offset))
        return np.asarray(positions, dtype=np.int64)

    def episode_batch(
        self,
        start_episode: int,
        num_episodes: int,
        *,
        device: str | torch.device = "cpu",
        shuffle: bool = True,
    ) -> OfflineBatch:
        positions = self._episode_positions(start_episode, num_episodes, shuffle=shuffle)
        idxs = np.concatenate([self._episode_indices[int(position)] for position in positions])
        return self._batch_from_indices(idxs, device=device)

    def iter_episode_batches(
        self,
        *,
        num_episodes: int | None = None,
        batch_episodes: int | None = None,
        device: str | torch.device = "cpu",
        shuffle: bool = False,
    ):
        total_episodes = self.num_episodes if num_episodes is None else int(num_episodes)
        batch_episodes = self.num_episodes if batch_episodes is None else int(batch_episodes)
        start_episode = 0
        while start_episode < total_episodes:
            current = min(batch_episodes, total_episodes - start_episode)
            yield self.episode_batch(start_episode, current, device=device, shuffle=shuffle)
            start_episode += current

    def _sequence_windows(self, horizon: int) -> np.ndarray:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if horizon not in self._sequence_windows_cache:
            windows = []
            for episode_indices in self._episode_indices:
                if len(episode_indices) < horizon:
                    continue
                for start in range(len(episode_indices) - horizon + 1):
                    windows.append(episode_indices[start : start + horizon])
            if windows:
                self._sequence_windows_cache[horizon] = np.asarray(windows, dtype=np.int64)
            else:
                self._sequence_windows_cache[horizon] = np.empty((0, horizon), dtype=np.int64)
        return self._sequence_windows_cache[horizon]

    def _sequence_batch_from_windows(
        self,
        windows: np.ndarray,
        *,
        device: str | torch.device = "cpu",
    ) -> OfflineSequenceBatch:
        obs = self._obs_array("obs")[windows]
        next_obs = self._obs_array("next_obs")[windows]
        return OfflineSequenceBatch(
            start_obs=torch.tensor(obs[:, 0, :], dtype=torch.float32, device=device),
            actions=torch.tensor(self.data["actions"][windows], dtype=torch.int64, device=device),
            rewards=torch.tensor(self.data["rewards"][windows], dtype=torch.float32, device=device),
            dones=torch.tensor(self.data["terminals"][windows].astype(np.float32), dtype=torch.float32, device=device),
            next_obs=torch.tensor(next_obs, dtype=torch.float32, device=device),
            next_state_id=torch.tensor(self.data["next_state_ids"][windows], dtype=torch.int64, device=device),
        )

    def iter_sequence_batches(
        self,
        horizon: int,
        batch_size: int,
        *,
        device: str | torch.device = "cpu",
    ):
        windows = self._sequence_windows(horizon)
        batch_size = int(batch_size)
        for start in range(0, len(windows), batch_size):
            yield self._sequence_batch_from_windows(windows[start : start + batch_size], device=device)
