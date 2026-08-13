"""Algorithm registry."""

from __future__ import annotations

from src.algorithms import bc, cql, dqn, iql, latent_model, ppo, raw_model, reconstructive_model
from src.offline.trainers import get_offline_trainer


TRAINERS = {
    "raw_model": raw_model.train,
    "latent_model": latent_model.train,
    "reconstructive_model": reconstructive_model.train,
    "dqn": dqn.train,
    "ppo": ppo.train,
}


POLICY_LOADERS = {
    "raw_model": raw_model.load_policy,
    "latent_model": latent_model.load_policy,
    "reconstructive_model": reconstructive_model.load_policy,
    "dqn": dqn.load_policy,
    "ppo": ppo.load_policy,
    "bc": bc.load_policy,
    "iql": iql.load_policy,
    "cql": cql.load_policy,
}


def get_trainer(algo: str):
    if algo not in TRAINERS:
        raise KeyError(f"Unknown algorithm: {algo}")
    return TRAINERS[algo]


def get_trainer_for_mode(algo: str, train_mode: str = "online"):
    if train_mode == "offline":
        return get_offline_trainer(algo)
    if train_mode != "online":
        raise ValueError(f"Unsupported train_mode={train_mode!r}")
    return get_trainer(algo)


def get_policy_loader(algo: str):
    if algo not in POLICY_LOADERS:
        raise KeyError(f"Unknown algorithm: {algo}")
    return POLICY_LOADERS[algo]
