"""Gym wrapper with consistent observation backends and metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bootstrap import bootstrap_icu_sepsis

bootstrap_icu_sepsis()

import gymnasium as gym  # noqa: E402
import icu_sepsis  # noqa: F401,E402

from src.utils.metrics import ACTION_LEVELS, decode_action


@dataclass
class EnvSpec:
    num_states: int
    num_actions: int
    obs_dim: int
    state_vector_dim: int
    obs_backend: str


class ICUSepsisEnvWrapper:
    def __init__(self, obs_backend: str = "discrete_id", seed: int | None = None, params=None):
        if obs_backend not in {"discrete_id", "state_vector"}:
            raise ValueError(f"Unsupported obs_backend={obs_backend}")
        self.obs_backend = obs_backend
        self.env = gym.make("Sepsis/ICU-Sepsis-v2", params=params) if params is not None else gym.make("Sepsis/ICU-Sepsis-v2")
        obs, info = self.env.reset(seed=seed)
        self._num_states = int(self.env.observation_space.n)
        self._num_actions = int(self.env.action_space.n)
        self._state_vector_dim = int(len(info["state_vector"]))
        self.spec = EnvSpec(
            num_states=self._num_states,
            num_actions=self._num_actions,
            obs_dim=self._num_states if obs_backend == "discrete_id" else self._state_vector_dim,
            state_vector_dim=self._state_vector_dim,
            obs_backend=obs_backend,
        )
        self.last_state_id = int(obs)

    @property
    def num_states(self) -> int:
        return self.spec.num_states

    @property
    def num_actions(self) -> int:
        return self.spec.num_actions

    @property
    def obs_dim(self) -> int:
        return self.spec.obs_dim

    def close(self) -> None:
        self.env.close()

    def one_hot(self, state_id: int) -> np.ndarray:
        vector = np.zeros(self.num_states, dtype=np.float32)
        vector[int(state_id)] = 1.0
        return vector

    def transform_obs(self, state_id: int, info: dict) -> np.ndarray:
        if self.obs_backend == "discrete_id":
            return self.one_hot(state_id)
        return np.asarray(info["state_vector"], dtype=np.float32)

    def action_mask(self, info: dict) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.float32)
        mask[np.asarray(info["admissible_actions"], dtype=np.int64)] = 1.0
        return mask

    def enrich_info(self, state_id: int, info: dict) -> dict:
        admissible_actions = list(info["admissible_actions"])
        action_mask = self.action_mask(info)
        enriched = dict(info)
        enriched["state_id"] = int(state_id)
        enriched["action_mask"] = action_mask
        enriched["admissible_actions"] = admissible_actions
        return enriched

    def reset(self, seed: int | None = None):
        state_id, info = self.env.reset(seed=seed)
        self.last_state_id = int(state_id)
        info = self.enrich_info(state_id, info)
        return self.transform_obs(state_id, info), info

    def step(self, action_idx: int):
        next_state_id, reward, terminated, truncated, info = self.env.step(int(action_idx))
        self.last_state_id = int(next_state_id)
        info = self.enrich_info(next_state_id, info)
        return self.transform_obs(next_state_id, info), float(reward), bool(terminated), bool(truncated), info

    @staticmethod
    def decode_action(action_idx: int) -> tuple[int, int]:
        return decode_action(int(action_idx), action_levels=ACTION_LEVELS)
