"""Gym wrapper with consistent observation backends and metadata."""

from __future__ import annotations

import numpy as np

from src.bootstrap import bootstrap_icu_sepsis

bootstrap_icu_sepsis()

import gymnasium as gym  # noqa: E402
import icu_sepsis  # noqa: F401,E402

from src.utils.metrics import ACTION_LEVELS, decode_action
from src.envs.base import EnvAdapter, EnvSpec


class ICUSepsisEnvWrapper(EnvAdapter):
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
            env_id="icu_sepsis",
            num_states=self._num_states,
            num_actions=self._num_actions,
            obs_dim=self._num_states if obs_backend == "discrete_id" else self._state_vector_dim,
            state_vector_dim=self._state_vector_dim,
            obs_backend=obs_backend,
            supported_obs_backends=("discrete_id", "state_vector"),
            action_labels=tuple(f"Fluid {fluid}; Vaso {vaso}" for fluid in range(ACTION_LEVELS) for vaso in range(ACTION_LEVELS)),
            feature_names=tuple(f"state_vector_{idx}" for idx in range(self._state_vector_dim)),
            metric_names=("sofa_score",),
            primary_metric="sofa_score",
            action_geometry={"type": "grid", "shape": [ACTION_LEVELS, ACTION_LEVELS], "axes": ["fluid_bin", "vaso_bin"]},
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

    def action_distance(self, previous: int, current: int) -> float:
        previous_fluid, previous_vaso = self.decode_action(previous)
        current_fluid, current_vaso = self.decode_action(current)
        return float(abs(previous_fluid - current_fluid) + abs(previous_vaso - current_vaso))

    def action_is_abrupt(self, previous: int, current: int) -> bool:
        previous_fluid, previous_vaso = self.decode_action(previous)
        current_fluid, current_vaso = self.decode_action(current)
        return bool(abs(previous_fluid - current_fluid) > 1 or abs(previous_vaso - current_vaso) > 1)

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {"sofa_score": float(info["sofa_score"])}

    def episode_metrics(self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]) -> dict[str, float | bool]:
        del final_info, transition_metrics
        survived = bool(total_reward > 0.0)
        return {"success": survived, "survived": survived}
