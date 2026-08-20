"""Small subset of DTRGym's base helpers required by the two vendored ODEs."""

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Union

import numpy as np


class BaseSimulator:
    @abstractmethod
    def activate(self, *args, **kwargs) -> OrderedDict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def update(self, action: Union[dict, float], state: dict, *args, **kwargs) -> OrderedDict[str, Any]:
        raise NotImplementedError


class BaseReward:
    @abstractmethod
    def count_reward(self, *args, **kwargs) -> float:
        raise NotImplementedError


def uniform_random(mean, width, absolute=False):
    """Port of DTRGym's parameter perturbation helper."""

    def single_random(value, amount):
        if absolute:
            return float(np.random.uniform(value - amount, value + amount))
        return float(np.random.uniform(value - value * amount, value + value * amount))

    if isinstance(mean, (list, tuple, np.ndarray)):
        return [single_random(value, width) for value in mean]
    if isinstance(mean, (int, float, np.floating, np.integer)):
        return single_random(mean, width)
    raise NotImplementedError(f"mean type {type(mean)} not supported")
