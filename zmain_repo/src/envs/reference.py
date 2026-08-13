"""Reference-policy adapters exposed by individual environments."""

from __future__ import annotations

import numpy as np


class EpiCareReferencePolicy:
    def __init__(self, env, name: str, *, seed: int = 0):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.bind_env(env)

    def bind_env(self, env) -> None:
        from epicare.policies import ClinicalTrial, Oracle, Random, StandardOfCare

        factories = {"random": Random, "smart": ClinicalTrial, "soc": StandardOfCare, "oracle": Oracle}
        if self.name not in factories:
            raise ValueError(f"Unknown EpiCare reference policy {self.name!r}")
        self.policy = factories[self.name](env.env)

    def reset_episode(self) -> None:
        self.policy.reset()

    def action_probabilities(self, obs, info, deterministic: bool = True) -> np.ndarray:
        probabilities = np.asarray(self.policy.get_treatment_probs(obs), dtype=np.float64)
        probabilities /= probabilities.sum()
        if not deterministic:
            return probabilities
        result = np.zeros_like(probabilities)
        result[int(np.argmax(probabilities))] = 1.0
        return result

    def act(self, obs, info, deterministic: bool = True) -> int:
        probabilities = self.action_probabilities(obs, info, deterministic=deterministic)
        return int(np.argmax(probabilities) if deterministic else self.rng.choice(len(probabilities), p=probabilities))

    def observe(self, action: int, reward: float) -> None:
        self.policy.update(int(action), float(reward))


class ICUReferencePolicy:
    """Direct ICU-Sepsis reference policies for the baseline report.

    Unlike an offline policy checkpoint this adapter exposes the admissible
    action mask and can therefore be evaluated with the same action-validity
    and stability instrumentation as learned policies.
    """

    def __init__(self, env, name: str, *, seed: int = 0, config: dict | None = None):
        self.env = env
        self.name = str(name)
        self.config = dict(config or {})
        self.rng = np.random.default_rng(seed)
        self.current_policy = self.name
        self.expert_policy = np.asarray(env.env.unwrapped.expert_policy, dtype=np.float64)
        self.optimal_policy = None
        if self.name in {"optimal", "mixed"} or "optimal" in self.config.get("mixed_weights", {}):
            from src.offline.generate_dataset import _build_optimal_policy

            self.optimal_policy = _build_optimal_policy(env, self.config)
        if self.name not in {"random_admissible", "expert", "optimal", "mixed", "epsilon_expert"}:
            raise ValueError(f"Unknown ICU reference policy {self.name!r}")

    def reset_episode(self) -> None:
        if self.name != "mixed":
            self.current_policy = self.name
            return
        weights = self.config.get("mixed_weights", {"expert": 0.4, "optimal": 0.2, "random_admissible": 0.4})
        names = list(weights)
        probabilities = np.asarray([weights[item] for item in names], dtype=np.float64)
        probabilities /= probabilities.sum()
        self.current_policy = str(self.rng.choice(names, p=probabilities))

    def _stochastic_probabilities(self, info: dict) -> np.ndarray:
        admissible = np.asarray(info["admissible_actions"], dtype=np.int64)
        result = np.zeros(self.env.num_actions, dtype=np.float64)
        if self.current_policy == "random_admissible":
            result[admissible] = 1.0 / max(1, len(admissible))
        elif self.current_policy == "expert":
            result[admissible] = self.expert_policy[int(info["state_id"]), admissible]
            total = result.sum()
            result[admissible] = 1.0 / max(1, len(admissible)) if total <= 0 else result[admissible] / total
        elif self.current_policy == "epsilon_expert":
            expert = np.zeros(self.env.num_actions, dtype=np.float64)
            expert[admissible] = self.expert_policy[int(info["state_id"]), admissible]
            total = expert.sum()
            if total <= 0:
                expert[admissible] = 1.0 / max(1, len(admissible))
            else:
                expert /= total
            random = np.zeros(self.env.num_actions, dtype=np.float64)
            random[admissible] = 1.0 / max(1, len(admissible))
            epsilon = float(self.config.get("epsilon", 0.2))
            result = (1.0 - epsilon) * expert + epsilon * random
        elif self.current_policy == "optimal":
            if self.optimal_policy is None:
                raise RuntimeError("optimal ICU reference policy was not initialized")
            action = int(self.optimal_policy[int(info["state_id"])])
            if action not in admissible:
                action = int(admissible[0])
            result[action] = 1.0
        else:
            raise ValueError(f"Unsupported ICU reference policy {self.current_policy!r}")
        return result / max(float(result.sum()), 1e-12)

    def action_probabilities(self, obs, info, deterministic: bool = True) -> np.ndarray:
        del obs
        probabilities = self._stochastic_probabilities(info)
        if not deterministic:
            return probabilities
        result = np.zeros_like(probabilities)
        result[int(np.argmax(probabilities))] = 1.0
        return result

    def act(self, obs, info, deterministic: bool = True) -> int:
        probabilities = self.action_probabilities(obs, info, deterministic=deterministic)
        return int(np.argmax(probabilities) if deterministic else self.rng.choice(len(probabilities), p=probabilities))

    def observe(self, action: int, reward: float) -> None:
        del action, reward


def make_reference_policy(env, name: str, *, seed: int = 0, config: dict | None = None):
    if env.spec.env_id == "epicare":
        return EpiCareReferencePolicy(env, name, seed=seed)
    if env.spec.env_id == "icu_sepsis":
        return ICUReferencePolicy(env, name, seed=seed, config=config)
    raise ValueError(f"Environment {env.spec.env_id!r} has no generic reference policy {name!r}")
