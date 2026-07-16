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


class TabularQNetwork(nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        self.network = nn.Linear(num_states, num_actions, bias=False)
        nn.init.constant_(self.network.weight, 0.0)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.network(obs)
        return masked_logits(logits, action_mask)


class TabularActorCritic(nn.Module):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        self.actor = nn.Linear(num_states, num_actions, bias=False)
        self.critic = nn.Linear(num_states, 1, bias=False)
        nn.init.constant_(self.actor.weight, 0.0)
        nn.init.constant_(self.critic.weight, 0.0)

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


def select_greedy_q(q_values: torch.Tensor) -> torch.Tensor:
    return torch.argmax(q_values, dim=-1)


def select_greedy_policy(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)
