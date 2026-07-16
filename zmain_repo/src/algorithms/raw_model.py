"""Direct observation-space model-based baseline."""

from __future__ import annotations

from src.algorithms.cem_world_models import load_policy as _load_policy
from src.algorithms.cem_world_models import train as _train


def train(config: dict, output_dir):
    return _train(config, output_dir, model_family="raw_model")


def load_policy(config: dict, checkpoint: dict):
    return _load_policy(config, checkpoint, model_family="raw_model")
