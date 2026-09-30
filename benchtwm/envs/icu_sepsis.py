"""ICU-Sepsis: a tabular MDP estimated from MIMIC-III sepsis trajectories.

The agent observes the 47-feature state-cluster centroid of the current MDP
state. Actions form a 5 x 5 grid of IV-fluid and vasopressor bins; only
admissible actions for the current state are unmasked.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import icu_sepsis  # noqa: F401  (registers Sepsis/ICU-Sepsis-v2)
import numpy as np

from benchtwm.envs import register_env
from benchtwm.envs.base import EnvAdapter, EnvSpec
from benchtwm.policies import Policy, greedy, masked_uniform, one_hot

ACTION_LEVELS = 5

# Column order of the upstream MDP construction (AI Clinician features in
# icu_sepsis_helpers/mdp_creation/create_rl_table.py: binary, z-scored, then
# log-transformed and z-scored columns).
FEATURE_NAMES = (
    "gender", "mechvent", "max_dose_vaso", "re_admission",
    "age", "Weight_kg", "GCS", "HR", "SysBP", "MeanBP", "DiaBP", "RR", "Temp_C", "FiO2_1",
    "Potassium", "Sodium", "Chloride", "Glucose", "Magnesium", "Calcium", "Hb", "WBC_count",
    "Platelets_count", "PTT", "PT", "Arterial_pH", "paO2", "paCO2", "Arterial_BE", "HCO3",
    "Arterial_lactate", "SOFA", "SIRS", "Shock_Index", "PaO2_FiO2", "cumulated_balance",
    "SpO2", "BUN", "Creatinine", "SGOT", "SGPT", "Total_bili", "INR",
    "input_total", "input_4hourly", "output_total", "output_4hourly",
)


def decode_action(action: int) -> tuple[int, int]:
    """Return the ``(fluid_bin, vasopressor_bin)`` of a flat action index."""

    return int(action) // ACTION_LEVELS, int(action) % ACTION_LEVELS


def value_iteration(tx_mat: np.ndarray, r_mat: np.ndarray, gamma: float, *, max_steps: int = 50_000, delta: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Tabular value iteration, identical to the upstream ICU-Sepsis helper.

    As upstream, the greedy policy is read from the last Bellman backup that
    was computed before the stopping test succeeded.
    """

    value = np.zeros((1, 1, tx_mat.shape[0]))
    for _ in range(int(max_steps)):
        backup = np.multiply(tx_mat, r_mat + gamma * value)
        new_value = np.max(np.sum(backup, axis=2), axis=1)
        if np.max(np.abs(new_value - value)) < delta:
            break
        value = new_value
    return np.argmax(np.sum(backup, axis=2), axis=1), value


@register_env
class ICUSepsis(EnvAdapter):
    env_id = "icu_sepsis"
    aliases = ("icu", "icu-sepsis")
    behavior_policy_ids = {"random_admissible": 0, "expert": 1, "optimal": 2, "epsilon_expert": 3}
    reference_policies = ("expert", "epsilon_expert", "optimal", "random_admissible")

    def __init__(self, *, seed: int | None = None, params: dict[str, Any] | None = None):
        self.env = gym.make("Sepsis/ICU-Sepsis-v2", params=params) if params else gym.make("Sepsis/ICU-Sepsis-v2")
        _, info = self.env.reset(seed=seed)
        self.spec = EnvSpec(
            env_id=self.env_id,
            num_actions=int(self.env.action_space.n),
            obs_dim=int(len(info["state_vector"])),
            feature_names=FEATURE_NAMES,
            action_labels=tuple(f"Fluid {fluid}; Vaso {vaso}" for fluid in range(ACTION_LEVELS) for vaso in range(ACTION_LEVELS)),
            metric_names=("sofa_score",),
            primary_metric="sofa_score",
            action_geometry={"type": "grid", "shape": [ACTION_LEVELS, ACTION_LEVELS], "axes": ["fluid_bin", "vaso_bin"]},
            num_states=int(self.env.observation_space.n),
        )
        if len(FEATURE_NAMES) != self.spec.obs_dim:
            raise RuntimeError(f"ICU-Sepsis exposes {self.spec.obs_dim} features, expected {len(FEATURE_NAMES)}")

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None) -> "ICUSepsis":
        return cls(seed=seed, params=dict(config.get("env_kwargs") or {}) or None)

    @property
    def mdp(self):
        return self.env.unwrapped

    def _observe(self, state_id: int, info: dict) -> tuple[np.ndarray, dict]:
        mask = np.zeros(self.num_actions, dtype=np.float32)
        mask[np.asarray(info["admissible_actions"], dtype=np.int64)] = 1.0
        enriched = {**info, "state_id": int(state_id), "action_mask": mask, "admissible_actions": list(info["admissible_actions"])}
        return np.asarray(info["state_vector"], dtype=np.float32), enriched

    def reset(self, seed: int | None = None):
        state_id, info = self.env.reset(seed=seed)
        return self._observe(state_id, info)

    def step(self, action: int):
        state_id, reward, terminated, truncated, info = self.env.step(int(action))
        obs, info = self._observe(state_id, info)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()

    def action_distance(self, previous: int, current: int) -> float:
        (pf, pv), (cf, cv) = decode_action(previous), decode_action(current)
        return float(abs(pf - cf) + abs(pv - cv))

    def action_is_abrupt(self, previous: int, current: int) -> bool:
        (pf, pv), (cf, cv) = decode_action(previous), decode_action(current)
        return bool(abs(pf - cf) > 1 or abs(pv - cv) > 1)

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {"sofa_score": float(info["sofa_score"])}

    def episode_metrics(self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]) -> dict[str, float | bool]:
        del final_info, transition_metrics
        return {"success": bool(total_reward > 0.0)}

    def export_columns(self, info: dict, action: int) -> dict[str, float | int]:
        fluid, vaso = decode_action(action)
        return {"fluid_bin": fluid, "vaso_bin": vaso}

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        options = dict(options or {})
        expert = np.asarray(self.mdp.expert_policy, dtype=np.float64)
        if name == "random_admissible":
            return RandomAdmissible(seed=seed)
        if name == "expert":
            return EpsilonExpert(expert, epsilon=0.0, seed=seed)
        if name == "epsilon_expert":
            return EpsilonExpert(expert, epsilon=float(options.get("epsilon", 0.2)), seed=seed)
        if name == "optimal":
            dynamics = self.mdp.dynamics
            optimal, _ = value_iteration(
                dynamics["tx_mat"],
                dynamics["r_mat"],
                float(options.get("gamma", 1.0)),
                max_steps=int(options.get("optimal_value_iteration_steps", 50_000)),
                delta=float(options.get("optimal_value_iteration_delta", 1e-6)),
            )
            return Optimal(np.asarray(optimal, dtype=np.int64), seed=seed)
        return super().make_policy(name, seed=seed, options=options)


class RandomAdmissible(Policy):
    """Uniform over the admissible actions of the current state."""

    def action_probabilities(self, obs, info, deterministic=True):
        probabilities = masked_uniform(info["action_mask"])
        return greedy(probabilities) if deterministic else probabilities


class EpsilonExpert(Policy):
    """Clinician action frequencies per state, mixed with ``epsilon`` uniform exploration.

    ``epsilon=0`` is the packaged expert policy. States whose admissible
    clinician mass is zero fall back to the uniform admissible policy.
    """

    def __init__(self, expert_policy: np.ndarray, *, epsilon: float, seed: int = 0):
        super().__init__(seed=seed)
        self.expert_policy = expert_policy
        self.epsilon = float(epsilon)

    def action_probabilities(self, obs, info, deterministic=True):
        mask = np.asarray(info["action_mask"], dtype=np.float64)
        expert = self.expert_policy[int(info["state_id"])] * (mask > 0)
        total = float(expert.sum())
        expert = masked_uniform(mask) if total <= 0.0 else expert / total
        if self.epsilon > 0.0:
            mixed = (1.0 - self.epsilon) * expert + self.epsilon * masked_uniform(mask)
            expert = mixed / mixed.sum()
        return greedy(expert) if deterministic else expert


class Optimal(Policy):
    """Deterministic value-iteration policy of the packaged MDP."""

    def __init__(self, actions: np.ndarray, *, seed: int = 0):
        super().__init__(seed=seed)
        self.actions = actions

    def action_probabilities(self, obs, info, deterministic=True):
        mask = np.asarray(info["action_mask"])
        action = int(self.actions[int(info["state_id"])])
        if mask[action] <= 0:
            action = int(np.flatnonzero(mask > 0)[0])
        return one_hot(len(mask), action)
