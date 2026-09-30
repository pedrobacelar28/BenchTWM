"""Simple MLP modules used across methods."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    if action_mask is None:
        return logits
    return logits.masked_fill(action_mask <= 0, -1e9)


class MLPQNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: list[int]):
        super().__init__()
        self.network = build_mlp(obs_dim, hidden_dims, num_actions)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        return masked_logits(self.network(obs), action_mask)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: list[int]):
        super().__init__()
        self.actor = build_mlp(obs_dim, hidden_dims, num_actions)
        self.critic = build_mlp(obs_dim, hidden_dims, 1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def policy_logits(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        return masked_logits(self.actor(obs), action_mask)

    def distribution(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> Categorical:
        return Categorical(logits=self.policy_logits(obs, action_mask))

    def sample_action(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        dist = self.distribution(obs, action_mask)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor | None = None):
        dist = self.distribution(obs, action_mask)
        return dist.log_prob(actions), dist.entropy(), self.critic(obs)

    def greedy_action(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.argmax(self.policy_logits(obs, action_mask), dim=-1)


def policy_hidden_dims(config: dict) -> list[int]:
    width = int(config.get("policy_hidden_dim", 256))
    layers = int(config.get("policy_hidden_layers", 2))
    if width <= 0 or layers <= 0:
        raise ValueError("policy_hidden_dim and policy_hidden_layers must be > 0 for MLP policies")
    return [width] * layers


def build_q_network(config: dict, *, obs_dim: int, num_actions: int) -> nn.Module:
    return MLPQNetwork(obs_dim, num_actions, policy_hidden_dims(config))


def build_actor_critic(config: dict, *, obs_dim: int, num_actions: int) -> nn.Module:
    return MLPActorCritic(obs_dim, num_actions, policy_hidden_dims(config))


def select_greedy_q(q_values: torch.Tensor) -> torch.Tensor:
    return torch.argmax(q_values, dim=-1)


def select_greedy_policy(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)
