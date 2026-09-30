"""The environment contract every benchmark adapter implements."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np

from benchtwm.policies import Policy


@dataclass(frozen=True)
class EnvSpec:
    """Static description persisted with every dataset and evaluation."""

    env_id: str
    num_actions: int
    obs_dim: int
    feature_names: tuple[str, ...]
    action_labels: tuple[str, ...]
    metric_names: tuple[str, ...] = ()
    primary_metric: str | None = None
    action_geometry: dict[str, Any] = field(default_factory=lambda: {"type": "categorical"})
    # Size of the simulator's diagnostic state space, when it has one. The
    # state id is logged for analysis only and is never an agent input.
    num_states: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("feature_names", "action_labels", "metric_names"):
            payload[key] = list(payload[key])
        return payload


class EnvAdapter:
    """Gymnasium-style environment with the benchmark's semantic surface.

    ``reset``/``step`` return a ``float32`` observation vector and an ``info``
    dict that always contains ``action_mask`` (``float32[num_actions]``),
    ``admissible_actions`` (list of ints), and ``state_id`` (diagnostic int).

    Subclasses are registered with :func:`benchtwm.envs.register_env` and
    declare their behavior/reference policies in ``policies``.
    """

    env_id: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    # Stable integer ids written to ``behavior_policy_ids`` in datasets.
    behavior_policy_ids: ClassVar[dict[str, int]] = {}
    # Policies evaluated by ``bench baselines`` when none are requested.
    reference_policies: ClassVar[tuple[str, ...]] = ()

    spec: EnvSpec

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None) -> "EnvAdapter":
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def make_policy(self, name: str, *, seed: int = 0, options: dict[str, Any] | None = None) -> Policy:
        """Build one of the environment's declared behavior/reference policies."""

        raise ValueError(f"{self.spec.env_id} declares no policy {name!r}; available: {sorted(self.behavior_policy_ids)}")

    @property
    def num_actions(self) -> int:
        return self.spec.num_actions

    @property
    def obs_dim(self) -> int:
        return self.spec.obs_dim

    def action_label(self, action: int) -> str:
        labels = self.spec.action_labels
        return labels[int(action)] if 0 <= int(action) < len(labels) else str(int(action))

    def action_distance(self, previous: int, current: int) -> float:
        return float(int(previous) != int(current))

    def action_is_abrupt(self, previous: int, current: int) -> bool:
        return bool(self.action_distance(previous, current) > 0.0)

    def transition_metrics(self, info: dict) -> dict[str, float]:
        """Per-step metrics, ordered as ``spec.metric_names``."""

        return {}

    def episode_metrics(self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]) -> dict[str, float | bool]:
        """Episode outcomes. Must contain ``success``."""

        del final_info, transition_metrics
        return {"success": bool(total_reward > 0.0)}

    def export_columns(self, info: dict, action: int) -> dict[str, float | int]:
        """Extra environment-specific columns for exported trajectories."""

        del info, action
        return {}
