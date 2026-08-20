"""Action wrappers retained for provenance; BenchTWM adapters use them only as a reference."""

from __future__ import annotations

import numpy as np
from gymnasium import Wrapper, spaces


class DiscreteActionWrapper(Wrapper):
    """Map a scalar Box action to an evenly spaced discrete dose grid."""

    def __init__(self, env, n_act: int):
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("DiscreteActionWrapper requires a Box action space")
        super().__init__(env)
        self.n_act = int(n_act)
        self.action_space = spaces.Discrete(self.n_act)
        if hasattr(self, "env_info"):
            self.env_info["action_type"] = "discrete"

    def step(self, action: int):
        bins = np.linspace(self.env.action_space.low, self.env.action_space.high, self.n_act)
        return self.env.step(np.asarray(bins[int(action)], dtype=np.float32))


class DiscreteDoubleActionWrapper(DiscreteActionWrapper):
    """Map a two-dimensional Box action to a square row-major grid."""

    def __init__(self, env, n_act: int = 25):
        side = int(np.sqrt(int(n_act)))
        if side * side != int(n_act):
            raise ValueError("n_act must be a perfect square")
        super().__init__(env, int(n_act))
        self.side = side

    def step(self, action: int):
        row, col = divmod(int(action), self.side)
        bins = np.linspace(self.env.action_space.low, self.env.action_space.high, self.side)
        return self.env.step(np.asarray([bins[row, 0], bins[col, 1]], dtype=np.float32))
