"""Small neural modules shared by the offline RL baselines."""

from __future__ import annotations

import torch.nn as nn

from benchtwm.models.mlp import build_actor_critic, build_mlp, build_q_network, policy_hidden_dims


class IQLAgent(nn.Module):
    """Discrete IQL actor, twin Q-functions and value function."""

    def __init__(self, config: dict, *, obs_dim: int, num_actions: int) -> None:
        super().__init__()
        self.actor = build_actor_critic(config, obs_dim=obs_dim, num_actions=num_actions)
        self.q1 = build_q_network(config, obs_dim=obs_dim, num_actions=num_actions)
        self.q2 = build_q_network(config, obs_dim=obs_dim, num_actions=num_actions)
        self.value = build_mlp(obs_dim, policy_hidden_dims(config), 1)


class CQLAgent(nn.Module):
    """Discrete CQL twin Q-functions."""

    def __init__(self, config: dict, *, obs_dim: int, num_actions: int) -> None:
        super().__init__()
        self.q1 = build_q_network(config, obs_dim=obs_dim, num_actions=num_actions)
        self.q2 = build_q_network(config, obs_dim=obs_dim, num_actions=num_actions)


def build_iql_agent(config: dict, *, obs_dim: int, num_actions: int) -> IQLAgent:
    return IQLAgent(config, obs_dim=obs_dim, num_actions=num_actions)


def build_cql_agent(config: dict, *, obs_dim: int, num_actions: int) -> CQLAgent:
    return CQLAgent(config, obs_dim=obs_dim, num_actions=num_actions)
