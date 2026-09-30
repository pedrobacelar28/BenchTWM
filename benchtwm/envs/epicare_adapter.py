"""Gymnasium adapter for the vendored EpiCare POMDP."""

from __future__ import annotations

import numpy as np

from benchtwm.envs.base import EnvAdapter, EnvSpec

from epicare import EpiCare


class EpiCareEnvWrapper(EnvAdapter):
    def __init__(self, *, seed: int | None = None, environment_seed: int = 1, **kwargs):
        self.obs_backend = "state_vector"
        self.environment_seed = int(environment_seed)
        self.env = EpiCare(seed=self.environment_seed, **kwargs)
        observation, _ = self.env.reset(seed=seed)
        num_actions = int(self.env.action_space.n)
        symptom_dim = int(np.asarray(observation).shape[0])
        self.spec = EnvSpec(
            env_id="epicare",
            num_actions=num_actions,
            obs_dim=symptom_dim,
            state_vector_dim=symptom_dim,
            obs_backend="state_vector",
            num_states=int(self.env.n_diseases + 1),
            supported_obs_backends=("state_vector",),
            action_labels=tuple(f"Treatment {idx}" for idx in range(num_actions)),
            feature_names=tuple(f"symptom_{idx}" for idx in range(symptom_dim)),
            metric_names=("symptom_burden", "remission", "adverse_event", "time_to_remission"),
            primary_metric="symptom_burden",
            action_geometry={"type": "categorical"},
        )
        self.last_state_id = self._state_id()

    def _state_id(self) -> int:
        if self.env.current_disease == "Remission":
            return int(self.env.n_diseases)
        return int(self.env.current_disease_index)

    def _enrich_info(self, observation, info: dict) -> dict:
        enriched = dict(info)
        enriched["state_id"] = self._state_id()
        enriched["state_vector"] = np.asarray(observation, dtype=np.float32)
        enriched["action_mask"] = np.ones(self.num_actions, dtype=np.float32)
        enriched["admissible_actions"] = list(range(self.num_actions))
        enriched.setdefault("remission", False)
        enriched.setdefault("adverse_event", False)
        enriched["symptom_burden"] = float(np.mean(observation))
        enriched["time_to_remission"] = -1.0 if self.env.time_to_remission is None else float(self.env.time_to_remission)
        return enriched

    def reset(self, seed: int | None = None):
        observation, info = self.env.reset(seed=seed)
        observation = np.asarray(observation, dtype=np.float32)
        self.last_state_id = self._state_id()
        return observation, self._enrich_info(observation, info)

    def step(self, action_idx: int):
        observation, reward, terminated, truncated, info = self.env.step(int(action_idx))
        observation = np.asarray(observation, dtype=np.float32)
        self.last_state_id = self._state_id()
        return observation, float(reward), bool(terminated), bool(truncated), self._enrich_info(observation, info)

    def close(self) -> None:
        self.env.close()

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {
            "symptom_burden": float(info["symptom_burden"]),
            "remission": float(bool(info.get("remission", False))),
            "adverse_event": float(bool(info.get("adverse_event", False))),
            "time_to_remission": float(info.get("time_to_remission", -1.0)),
        }

    def episode_metrics(self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]) -> dict[str, float | bool]:
        del total_reward
        remission = bool(any(row.get("remission", 0.0) > 0.0 for row in transition_metrics))
        adverse_event = bool(any(row.get("adverse_event", 0.0) > 0.0 for row in transition_metrics))
        time_to_remission = float(final_info.get("time_to_remission", -1.0))
        return {
            "success": remission,
            "remission": remission,
            "adverse_event": adverse_event,
            "time_to_remission": time_to_remission if remission else np.nan,
        }
