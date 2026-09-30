"""Wrappers for converting between 1D and 2D action spaces."""

import gymnasium
import numpy as np
from gymnasium import ActionWrapper, spaces


class FlattenActionWrapper(ActionWrapper):
    """Wrapper for flattening multi-discrete actions into a single discrete
    action.

    This wrapper is applied by default to the ICU-Sepsis environment,
    which converts its multi-discrete action space into a single discrete
    action space.
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.action_levels = super().action_space.nvec
        self.action_space = spaces.Discrete(np.prod(self.action_levels))

    def action(self, action):
        return action // self.action_levels[1], action % self.action_levels[1]
