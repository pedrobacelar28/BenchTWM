"""Adapters for the selective AhnChemo/GhaffariCancer DTR runtime.

The upstream simulators keep a richer state than the public benchmark
observation.  This module is the boundary that prevents that hidden state
from leaking into ``info`` or the offline dataset.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.bootstrap import bootstrap_dtr
from src.envs.base import EnvAdapter, EnvSpec

bootstrap_dtr()

from dtr_runtime.ahn_chemo_env import AhnChemoEnv  # noqa: E402
from dtr_runtime.ghaffari_cancer_env import GhaffariCancerEnv  # noqa: E402


def _state_fingerprint(vector: np.ndarray) -> int:
    """Return a stable positive int64 diagnostic ID for a public vector."""

    digest = hashlib.blake2b(np.asarray(vector, dtype=np.float32).tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _all_actions(num_actions: int) -> tuple[np.ndarray, list[int]]:
    return np.ones(num_actions, dtype=np.float32), list(range(num_actions))


class _DTRWrapper(EnvAdapter):
    """Shared reset/step, seed, masking and info-sanitization contract."""

    runtime_env_cls: type
    settings: dict[int, dict[str, float]]
    spec: EnvSpec
    runtime_metadata = {
        "name": "DTRGym-selective",
        "version": "0.1.0",
        "source_commit": "1f4fed49769b371c01d90959d2e012154068b3be",
        "source_url": "https://github.com/GilesLuo/DTRGym",
        "public_setting": 1,
        "prepared_settings": [1, 2, 3, 4, 5],
    }

    def __init__(self, *, setting: int = 1, seed: int | None = None, max_t: int | None = None, **kwargs):
        self.setting = int(setting)
        if self.setting not in self.settings:
            raise ValueError(f"Unsupported DTR setting={self.setting}; expected 1..5")
        runtime_kwargs = dict(self.settings[self.setting])
        if max_t is not None:
            runtime_kwargs["max_t"] = int(max_t)
        runtime_kwargs.update(kwargs)
        self.runtime_kwargs = dict(runtime_kwargs)
        self.env = self.runtime_env_cls(**runtime_kwargs)
        observation, _ = self.env.reset(seed=seed)
        self._last_observation = np.asarray(observation, dtype=np.float32)
        self._last_action = 0

    @property
    def num_actions(self) -> int:
        return self.spec.num_actions

    def _physical_action(self, action_idx: int) -> np.ndarray | float:
        raise NotImplementedError

    def _action_metrics(self, action_idx: int) -> dict[str, float]:
        del action_idx
        return {}

    def _observation_metrics(self, observation: np.ndarray) -> dict[str, float]:
        del observation
        return {}

    def _terminal_metrics(self, *, terminated: bool, truncated: bool) -> dict[str, bool]:
        del terminated, truncated
        return {}

    def _public_info(
        self,
        observation: np.ndarray,
        *,
        terminated: bool = False,
        truncated: bool = False,
        action_idx: int = 0,
        reward: float = 0.0,
    ) -> dict[str, Any]:
        vector = np.asarray(observation, dtype=np.float32).reshape(self.spec.obs_dim)
        mask, admissible = _all_actions(self.num_actions)
        # Deliberately construct a new dictionary.  The upstream ``state``
        # object is never copied into the agent-facing contract.
        info: dict[str, Any] = {
            "state_id": _state_fingerprint(vector),
            "state_vector": vector.copy(),
            "action_mask": mask,
            "admissible_actions": admissible,
            "action_idx": int(action_idx),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        info.update(self._observation_metrics(vector))
        info.update(self._action_metrics(action_idx))
        info.update(self._terminal_metrics(terminated=terminated, truncated=truncated))
        return info

    def reset(self, seed: int | None = None):
        observation, _raw_info = self.env.reset(seed=seed)
        del _raw_info
        observation = np.asarray(observation, dtype=np.float32)
        self._last_observation = observation.copy()
        self._last_action = 0
        return observation, self._public_info(observation)

    def step(self, action_idx: int):
        action_idx = int(action_idx)
        if not 0 <= action_idx < self.num_actions:
            raise ValueError(f"action {action_idx} is outside [0, {self.num_actions})")
        physical_action = self._physical_action(action_idx)
        observation, reward, terminated, truncated, _raw_info = self.env.step(physical_action)
        del _raw_info
        observation = np.asarray(observation, dtype=np.float32)
        self._last_observation = observation.copy()
        self._last_action = action_idx
        return (
            observation,
            float(np.asarray(reward).reshape(-1)[0]),
            bool(terminated),
            bool(truncated),
            self._public_info(
                observation,
                terminated=bool(terminated),
                truncated=bool(truncated),
                action_idx=action_idx,
                reward=float(np.asarray(reward).reshape(-1)[0]),
            ),
        )

    def close(self) -> None:
        self.env.close()


class DTRAhnChemoEnvWrapper(_DTRWrapper):
    """AhnChemo with three visible features and five dose bins."""

    runtime_env_cls = AhnChemoEnv
    settings = {
        1: {"delayed_steps": 0, "obs_noise": 0.0, "state_noise": 0.0, "pkpd_noise": 0.0, "missing_rate": 0.0},
        2: {"delayed_steps": 0, "obs_noise": 0.0, "state_noise": 0.0, "pkpd_noise": 0.1, "missing_rate": 0.0},
        3: {"delayed_steps": 0, "obs_noise": 0.2, "state_noise": 0.1, "pkpd_noise": 0.1, "missing_rate": 0.0},
        4: {"delayed_steps": 0, "obs_noise": 0.5, "state_noise": 0.2, "pkpd_noise": 0.1, "missing_rate": 0.0},
        5: {"delayed_steps": 0, "obs_noise": 0.5, "state_noise": 0.2, "pkpd_noise": 0.1, "missing_rate": 0.75},
    }
    _DOSE_LEVELS = np.linspace(0.0, 1.0, 5, dtype=np.float32)

    def __init__(self, **kwargs):
        self.spec = EnvSpec(
            env_id="dtr-ahn-chemo",
            num_actions=5,
            obs_dim=3,
            state_vector_dim=3,
            obs_backend="state_vector",
            supported_obs_backends=("state_vector",),
            action_labels=tuple(f"Dose {dose:.2f}" for dose in self._DOSE_LEVELS),
            feature_names=("tumor_population", "immune_population", "drug_concentration"),
            metric_names=("tumor_population", "immune_population", "drug_concentration", "dose_applied", "death"),
            primary_metric="tumor_population",
            action_geometry={
                "type": "ordinal",
                "shape": [5],
                "axes": ["dose_bin"],
                "physical_range": [0.0, 1.0],
                "action_to_dose": [float(value) for value in self._DOSE_LEVELS],
            },
        )
        self.observation_transform = {
            "type": "identity_with_native_clipping",
            "source_features": ["Tumor Cell Population", "Immune Cell Population", "Drug Concentration"],
            "hidden_features": ["Normal Cell Population"],
        }
        super().__init__(**kwargs)

    def _physical_action(self, action_idx: int) -> np.ndarray:
        return np.asarray([self._DOSE_LEVELS[int(action_idx)]], dtype=np.float32)

    def _action_metrics(self, action_idx: int) -> dict[str, float]:
        return {"dose_applied": float(self._DOSE_LEVELS[int(action_idx)])}

    def _observation_metrics(self, observation: np.ndarray) -> dict[str, float]:
        return {
            "tumor_population": float(observation[0]),
            "immune_population": float(observation[1]),
            "drug_concentration": float(observation[2]),
        }

    def _terminal_metrics(self, *, terminated: bool, truncated: bool) -> dict[str, bool]:
        return {"death": bool(terminated), "survived_horizon": bool(truncated and not terminated)}

    def action_distance(self, previous: int, current: int) -> float:
        return float(abs(int(previous) - int(current)) / 4.0)

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {
            "tumor_population": float(info["tumor_population"]),
            "immune_population": float(info["immune_population"]),
            "drug_concentration": float(info["drug_concentration"]),
            "dose_applied": float(info.get("dose_applied", 0.0)),
            "death": float(bool(info.get("death", False))),
        }

    def episode_metrics(self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]) -> dict[str, float | bool]:
        del total_reward, transition_metrics
        death = bool(final_info.get("death", False))
        return {
            "success": not death,
            "survived": not death,
            "death": death,
            "survived_horizon": bool(final_info.get("truncated", False) and not death),
        }


class DTRGhaffariCancerEnvWrapper(_DTRWrapper):
    """GhaffariCancer with seven log features and a 5x5 action grid."""

    runtime_env_cls = GhaffariCancerEnv
    settings = {
        1: {"delayed_steps": 0, "obs_noise": 0.0, "state_noise": 0.0, "pkpd_noise": 0.0, "missing_rate": 0.0},
        2: {"delayed_steps": 0, "obs_noise": 0.0, "state_noise": 0.0, "pkpd_noise": 0.1, "missing_rate": 0.0},
        3: {"delayed_steps": 0, "obs_noise": 0.1, "state_noise": 0.2, "pkpd_noise": 0.1, "missing_rate": 0.0},
        4: {"delayed_steps": 0, "obs_noise": 0.2, "state_noise": 0.5, "pkpd_noise": 0.1, "missing_rate": 0.0},
        5: {"delayed_steps": 0, "obs_noise": 0.2, "state_noise": 0.5, "pkpd_noise": 0.1, "missing_rate": 0.5},
    }
    _RADIATION_LEVELS = np.linspace(0.0, 10.0, 5, dtype=np.float32)
    _CHEMOTHERAPY_LEVELS = np.linspace(0.0, 8.0, 5, dtype=np.float32)

    def __init__(self, **kwargs):
        action_labels = tuple(
            f"Radiotherapy {radiation:g}; Chemotherapy {chemotherapy:g}"
            for radiation in self._RADIATION_LEVELS
            for chemotherapy in self._CHEMOTHERAPY_LEVELS
        )
        self.spec = EnvSpec(
            env_id="dtr-ghaffari-cancer",
            num_actions=25,
            obs_dim=7,
            state_vector_dim=7,
            obs_backend="state_vector",
            supported_obs_backends=("state_vector",),
            action_labels=action_labels,
            feature_names=(
                "primary_tumor",
                "primary_nk_cells",
                "primary_lymphocytes",
                "lymphocyte_concentration",
                "secondary_tumor",
                "secondary_nk_cells",
                "secondary_lymphocytes",
            ),
            metric_names=(
                "primary_tumor",
                "secondary_tumor",
                "tumor_burden",
                "radiotherapy_dose",
                "chemotherapy_dose",
                "positive_termination",
                "negative_termination",
            ),
            primary_metric="tumor_burden",
            action_geometry={
                "type": "grid",
                "shape": [5, 5],
                "axes": ["radiotherapy", "chemotherapy"],
                "radiotherapy_levels": [float(value) for value in self._RADIATION_LEVELS],
                "chemotherapy_levels": [float(value) for value in self._CHEMOTHERAPY_LEVELS],
                "indexing": "row_major",
            },
        )
        self.observation_transform = {
            "type": "natural_log",
            "base": "e",
            "lower_clip_before_log": 1.0,
            "source_features": ["T_p", "N_p", "L_p", "C", "T_s", "N_s", "L_s"],
            "hidden_features": ["M", "u", "v", "x", "ODE_parameters", "metastasis_state"],
        }
        super().__init__(**kwargs)

    def _physical_action(self, action_idx: int) -> np.ndarray:
        row, col = divmod(int(action_idx), 5)
        return np.asarray([self._RADIATION_LEVELS[row], self._CHEMOTHERAPY_LEVELS[col]], dtype=np.float32)

    def _action_metrics(self, action_idx: int) -> dict[str, float]:
        row, col = divmod(int(action_idx), 5)
        return {
            "radiotherapy_dose": float(self._RADIATION_LEVELS[row]),
            "chemotherapy_dose": float(self._CHEMOTHERAPY_LEVELS[col]),
        }

    def _observation_metrics(self, observation: np.ndarray) -> dict[str, float]:
        # The public vector is log-scaled.  These physical-scale metrics are
        # derived only from visible entries; hidden ODE variables are not read.
        primary = float(np.exp(np.clip(observation[0], -80.0, 80.0)))
        secondary = float(np.exp(np.clip(observation[4], -80.0, 80.0)))
        return {
            "primary_tumor": primary,
            "secondary_tumor": secondary,
            "tumor_burden": primary + secondary,
        }

    def _terminal_metrics(self, *, terminated: bool, truncated: bool) -> dict[str, bool]:
        # The upstream flags are available on the runtime, but no hidden state
        # is copied into info.  These are public terminal outcomes.
        positive = bool(getattr(self.env, "positive_terminated", False)) if terminated else False
        negative = bool(getattr(self.env, "negative_terminated", False)) if terminated else False
        return {
            "positive_termination": positive,
            "negative_termination": negative,
            "survived_horizon": bool(truncated and not terminated),
        }

    def action_distance(self, previous: int, current: int) -> float:
        previous_row, previous_col = divmod(int(previous), 5)
        current_row, current_col = divmod(int(current), 5)
        return float(abs(previous_row - current_row) + abs(previous_col - current_col))

    def action_is_abrupt(self, previous: int, current: int) -> bool:
        previous_row, previous_col = divmod(int(previous), 5)
        current_row, current_col = divmod(int(current), 5)
        return bool(abs(previous_row - current_row) > 1 or abs(previous_col - current_col) > 1)

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {
            "primary_tumor": float(info["primary_tumor"]),
            "secondary_tumor": float(info["secondary_tumor"]),
            "tumor_burden": float(info["tumor_burden"]),
            "radiotherapy_dose": float(info.get("radiotherapy_dose", 0.0)),
            "chemotherapy_dose": float(info.get("chemotherapy_dose", 0.0)),
            "positive_termination": float(bool(info.get("positive_termination", False))),
            "negative_termination": float(bool(info.get("negative_termination", False))),
        }

    def episode_metrics(self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]) -> dict[str, float | bool]:
        del total_reward, transition_metrics
        positive = bool(final_info.get("positive_termination", False))
        negative = bool(final_info.get("negative_termination", False))
        return {
            "success": positive,
            "positive_termination": positive,
            "negative_termination": negative,
            "survived_horizon": bool(final_info.get("truncated", False) and not (positive or negative)),
        }
