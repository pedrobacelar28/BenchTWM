"""Algorithm registry."""

from __future__ import annotations

from benchtwm.algorithms import bc, cql, dqn, dreamer, iql, latent_model, raw_model, reconstructive_model
from benchtwm.offline.trainers import get_offline_trainer


POLICY_LOADERS = {
    "raw_model": raw_model.load_policy,
    "latent_model": latent_model.load_policy,
    "reconstructive_model": reconstructive_model.load_policy,
    "dqn": dqn.load_policy,
    "bc": bc.load_policy,
    "iql": iql.load_policy,
    "cql": cql.load_policy,
    "dreamer": dreamer.load_policy,
}


def get_trainer(algo: str):
    return get_offline_trainer(algo)


def get_policy_loader(algo: str):
    if algo not in POLICY_LOADERS:
        raise KeyError(f"Unknown algorithm: {algo}")
    return POLICY_LOADERS[algo]
