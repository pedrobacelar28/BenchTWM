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


def make_reference_policy(env, name: str, *, seed: int = 0):
    if env.spec.env_id == "epicare":
        return EpiCareReferencePolicy(env, name, seed=seed)
    raise ValueError(f"Environment {env.spec.env_id!r} has no generic reference policy {name!r}")
