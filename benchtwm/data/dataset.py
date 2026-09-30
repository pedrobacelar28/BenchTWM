"""Loading offline datasets (schema: ``docs/data-schema.md``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchtwm.paths import PROJECT_ROOT
from benchtwm.utils.io import load_json

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


@dataclass
class OfflineSequenceBatch:
    """``k`` consecutive logged transitions starting at ``start_obs``."""

    start_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_obs: torch.Tensor


def resolve_dataset_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


class OfflineTransitionDataset:
    """One split (``train.npz``, ``val.npz``, or ``ope_test.npz``) held in memory."""

    def __init__(self, path: str | Path, *, seed: int | None = None):
        self.path = resolve_dataset_path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        metadata_path = self.path.parent / "metadata.json"
        self.metadata = load_json(metadata_path) if metadata_path.exists() else {}
        with np.load(self.path, allow_pickle=False) as payload:
            missing = sorted(REQUIRED_ARRAYS - set(payload.files))
            if missing:
                raise ValueError(f"Offline dataset {self.path} is missing required arrays: {missing}")
            self.data = {key: payload[key] for key in payload.files}
        self.size = int(len(self.data["actions"]))
        if self.size == 0:
            raise ValueError(f"Offline dataset {self.path} is empty")
        self.rng = np.random.default_rng(seed)
        episode_ids = self.data["episode_ids"]
        #: Transition indices of every episode, in timestep order.
        self.episode_indices = [
            indices[np.argsort(self.data["timesteps"][indices], kind="stable")]
            for indices in (np.flatnonzero(episode_ids == episode) for episode in np.unique(episode_ids))
        ]
        self._windows: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.size

    @property
    def num_episodes(self) -> int:
        return len(self.episode_indices)

    @property
    def obs_dim(self) -> int:
        return int(self.data["observations"].shape[1])

    @property
    def num_actions(self) -> int:
        return int(self.data["action_masks"].shape[1])

    @property
    def observations(self) -> np.ndarray:
        return self.data["observations"]

    @property
    def next_observations(self) -> np.ndarray:
        return self.data["next_observations"]

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = self.metadata.get("observation", {}).get("feature_names") or self.metadata.get("environment", {}).get("feature_names", [])
        return tuple(str(name) for name in names)

    def _batch(self, indices: np.ndarray, device) -> OfflineBatch:
        def tensor(key: str, dtype) -> torch.Tensor:
            return torch.as_tensor(self.data[key][indices], dtype=dtype, device=device)

        return OfflineBatch(
            obs=tensor("observations", torch.float32),
            action=tensor("actions", torch.int64),
            reward=tensor("rewards", torch.float32),
            next_obs=tensor("next_observations", torch.float32),
            done=tensor("terminals", torch.float32),
            action_mask=tensor("action_masks", torch.float32),
            next_action_mask=tensor("next_action_masks", torch.float32),
        )

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> OfflineBatch:
        """Uniform transitions with replacement (the training batch)."""

        return self._batch(self.rng.integers(0, self.size, size=int(batch_size)), device)

    def iter_transition_batches(self, batch_size: int, *, device: str | torch.device = "cpu"):
        """Every transition exactly once, in storage order (validation)."""

        for start in range(0, self.size, int(batch_size)):
            yield self._batch(np.arange(start, min(start + int(batch_size), self.size)), device)

    def iter_sequence_batches(self, horizon: int, batch_size: int, *, device: str | torch.device = "cpu"):
        """Every length-``horizon`` window that stays inside one episode."""

        horizon = int(horizon)
        if horizon not in self._windows:
            windows = [indices[start : start + horizon] for indices in self.episode_indices for start in range(len(indices) - horizon + 1)]
            self._windows[horizon] = np.asarray(windows, dtype=np.int64).reshape(-1, horizon)
        windows = self._windows[horizon]
        for start in range(0, len(windows), int(batch_size)):
            chunk = windows[start : start + int(batch_size)]
            yield OfflineSequenceBatch(
                start_obs=torch.as_tensor(self.data["observations"][chunk[:, 0]], dtype=torch.float32, device=device),
                actions=torch.as_tensor(self.data["actions"][chunk], dtype=torch.int64, device=device),
                rewards=torch.as_tensor(self.data["rewards"][chunk], dtype=torch.float32, device=device),
                dones=torch.as_tensor(self.data["terminals"][chunk], dtype=torch.float32, device=device),
                next_obs=torch.as_tensor(self.data["next_observations"][chunk], dtype=torch.float32, device=device),
            )
