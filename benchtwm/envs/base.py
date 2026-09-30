"""Common environment-adapter contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    num_actions: int
    obs_dim: int
    state_vector_dim: int
    obs_backend: str
    num_states: int | None = None
    supported_obs_backends: tuple[str, ...] = ("state_vector",)
    action_labels: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    metric_names: tuple[str, ...] = ()
    primary_metric: str | None = None
    action_geometry: dict[str, Any] = field(default_factory=lambda: {"type": "categorical"})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("supported_obs_backends", "action_labels", "feature_names", "metric_names"):
            payload[key] = list(payload[key])
        return payload


class EnvAdapter:
    """Small common surface consumed by trainers, datasets and reports."""

    spec: EnvSpec

    @property
    def num_actions(self) -> int:
        return self.spec.num_actions

    @property
    def num_states(self) -> int:
        if self.spec.num_states is None:
            raise ValueError(f"{self.spec.env_id} does not expose a discrete observation backend")
        return self.spec.num_states

    @property
    def obs_dim(self) -> int:
        return self.spec.obs_dim

    def action_label(self, action_idx: int) -> str:
        labels = self.spec.action_labels
        return labels[int(action_idx)] if 0 <= int(action_idx) < len(labels) else str(int(action_idx))

    def action_distance(self, previous: int, current: int) -> float:
        return float(int(previous) != int(current))

    def action_is_abrupt(self, previous: int, current: int) -> bool:
        return bool(self.action_distance(previous, current) > 0.0)

    def transition_metrics(self, info: dict) -> dict[str, float]:
        return {}

    def episode_metrics(self, *, total_reward: float, final_info: dict, transition_metrics: list[dict[str, float]]) -> dict[str, float | bool]:
        del final_info, transition_metrics
        return {"success": bool(total_reward > 0.0)}
