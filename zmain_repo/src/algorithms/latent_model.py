"""Latent transition model-based baseline."""

from __future__ import annotations

from src.algorithms.cem_world_models import load_policy as _load_policy


def load_policy(config: dict, checkpoint: dict):
    return _load_policy(config, checkpoint, model_family="latent_model")
