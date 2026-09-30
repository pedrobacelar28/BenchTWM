"""The policy contract shared by behavior policies, baselines, and learned agents.

A policy maps ``(obs, info)`` to a probability vector over the discrete
actions. Everything else (greedy or stochastic action selection, behavior
propensities for logged datasets, and importance ratios for OPE) is derived
from that single method, so a policy written once can collect data, be
evaluated as a reference baseline, and serve as an OPE target.
"""

from __future__ import annotations

import math

import numpy as np


EXACT_PROBABILITIES = {
    "status": "exact",
    "message": "The policy exposes action probabilities directly for the requested target mode.",
}


def masked_uniform(action_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=np.float64)
    return mask / max(float(mask.sum()), 1.0)


def one_hot(num_actions: int, action: int) -> np.ndarray:
    probabilities = np.zeros(int(num_actions), dtype=np.float64)
    probabilities[int(action)] = 1.0
    return probabilities


def greedy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    return one_hot(len(probabilities), int(np.argmax(probabilities))).astype(probabilities.dtype, copy=False)


def sample_action(rng: np.random.Generator, probabilities: np.ndarray) -> tuple[int, float]:
    """Sample one action and return it with its exact log propensity."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    action = int(rng.choice(len(probabilities), p=probabilities))
    return action, float(math.log(max(probabilities[action], 1e-12)))


class Policy:
    """Base class for every policy in the benchmark.

    Subclasses implement :meth:`action_probabilities`. Stateful policies
    (recurrent agents, adaptive clinical policies) also override
    :meth:`reset_episode` and :meth:`observe`.
    """

    ope_probability_contract = EXACT_PROBABILITIES

    def __init__(self, *, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def bind_env(self, env) -> None:
        """Called by simulator evaluation with the environment it will act in.

        Only policies that read simulator internals (e.g. privileged oracles)
        need to override this.
        """

    def reset_episode(self) -> None:
        """Called before the first step of every episode."""

    def observe(self, action: int, reward: float) -> None:
        """Called after every environment step with the executed action."""

    def action_probabilities(self, obs, info: dict, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    def act(self, obs, info: dict, deterministic: bool = True) -> int:
        probabilities = self.action_probabilities(obs, info, deterministic=deterministic)
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self.rng.choice(len(probabilities), p=np.asarray(probabilities, dtype=np.float64)))
