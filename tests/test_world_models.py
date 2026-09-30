"""Unit tests for the CEM world-model families."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from benchtwm.algorithms import DataSpec, get_algorithm
from benchtwm.algorithms.world_models import CEMWorldModelPolicy
from benchtwm.planners.cem import CategoricalCEMPlanner

from conftest import TINY_HYPERPARAMETERS

FAMILIES = ["raw_model", "latent_model", "reconstructive_model"]


def _algorithm(name: str, **overrides):
    config = {**TINY_HYPERPARAMETERS["common"], **TINY_HYPERPARAMETERS[name], "learning_rate": 1e-2, **overrides}
    return get_algorithm(name)(config, DataSpec(obs_dim=6, num_actions=3), torch.device("cpu"))


def _batch():
    obs = torch.tensor(
        [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0], [0.2, 0.2, 0.1, 0.7, 0.3, 0.1], [0.9, 0.2, 0.2, 0.2, 0.1, 0.3]]
    )
    actions = torch.tensor([0, 1, 2, 1])
    shift = torch.nn.functional.one_hot(actions, num_classes=3).float()
    next_obs = obs + torch.cat([shift, shift], dim=1) * 0.1
    return SimpleNamespace(obs=obs, next_obs=next_obs, action=actions, reward=next_obs[:, :2].sum(dim=1), done=torch.tensor([0.0, 0.0, 1.0, 0.0]))


@pytest.mark.parametrize("name", FAMILIES)
def test_heads_read_the_joint_state_action_input(name):
    model = _algorithm(name).model
    state_dim = 6 if name == "raw_model" else 8
    assert model.reward_head[0].in_features == state_dim + 4
    assert model.continue_head[0].in_features == state_dim + 4


@pytest.mark.parametrize("name", FAMILIES)
def test_training_loss_decreases(name):
    algorithm = _algorithm(name)
    batch = _batch()
    losses = [algorithm.update(batch, step)["loss"] for step in range(1, 31)]
    assert losses[-1] < losses[0]


def test_latent_model_updates_its_ema_target_encoder():
    algorithm = _algorithm("latent_model")
    before = [parameter.detach().clone() for parameter in algorithm.model.target_encoder.parameters()]
    algorithm.update(_batch(), 1)
    assert any(not torch.allclose(old, new) for old, new in zip(before, algorithm.model.target_encoder.parameters()))


def test_latent_model_rejects_invalid_target_decay():
    with pytest.raises(ValueError, match="target_encoder_decay"):
        _algorithm("latent_model", target_encoder_decay=1.0)


@pytest.mark.parametrize("name", FAMILIES)
def test_reward_and_continuation_heads_drive_cem_scores(name):
    model = _algorithm(name).model
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    model.reward_head[-1].bias.data.fill_(3.0)
    model.continue_head[-1].bias.data.fill_(8.0)
    policy = CEMWorldModelPolicy(model, CategoricalCEMPlanner(horizon=1, num_actions=3, samples=4, elites=2, iterations=1), gamma=1.0)
    start = model.rollout_state_from_obs(torch.zeros(1, 6)).squeeze(0)
    scores = policy._score_sequences(start, torch.tensor([[0], [1]]))
    assert torch.allclose(scores, torch.full_like(scores, 3.0), atol=1e-4)
