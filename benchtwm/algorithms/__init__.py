"""Algorithm registry.

An algorithm is one module that defines an :class:`OfflineAlgorithm` subclass
decorated with :func:`register_algorithm`, and is imported at the bottom of
this file. Its default hyperparameters live in ``configs/algorithms/<name>.json``.
"""

from __future__ import annotations

from typing import Any

from benchtwm.algorithms.base import MODEL_BASED, MODEL_FREE, DataSpec, OfflineAlgorithm

_ALGORITHMS: dict[str, type[OfflineAlgorithm]] = {}


def register_algorithm(cls: type[OfflineAlgorithm]) -> type[OfflineAlgorithm]:
    for key in {cls.name, cls.method_id}:
        if key in _ALGORITHMS:
            raise ValueError(f"Algorithm {key!r} is already registered")
        _ALGORITHMS[key] = cls
    return cls


def get_algorithm(name: str) -> type[OfflineAlgorithm]:
    """Look an algorithm up by its ``algo`` name or public method id."""

    try:
        return _ALGORITHMS[str(name)]
    except KeyError as exc:
        raise KeyError(f"Unknown algorithm {name!r}; registered: {algorithm_names()}") from exc


def algorithm_names() -> list[str]:
    return sorted({cls.name for cls in _ALGORITHMS.values()})


def algorithms() -> list[type[OfflineAlgorithm]]:
    return [get_algorithm(name) for name in algorithm_names()]


def load_policy(config: dict[str, Any], checkpoint: dict[str, Any]):
    """Rebuild the deployed policy of a trained run from its config and checkpoint."""

    from benchtwm.envs import make_env
    from benchtwm.utils.torch_utils import get_torch_device

    env = make_env(config)
    try:
        spec = DataSpec.from_env(env)
    finally:
        env.close()
    algorithm = get_algorithm(config["algo"]).from_checkpoint(config, spec, get_torch_device(), checkpoint)
    return algorithm.make_policy()


# Built-in algorithms register themselves on import.
from benchtwm.algorithms import bc, cql, dqn, dreamer, iql, world_models  # noqa: E402,F401

__all__ = [
    "MODEL_BASED",
    "MODEL_FREE",
    "DataSpec",
    "OfflineAlgorithm",
    "algorithm_names",
    "algorithms",
    "get_algorithm",
    "load_policy",
    "register_algorithm",
]
