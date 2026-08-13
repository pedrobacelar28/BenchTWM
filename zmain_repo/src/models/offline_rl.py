"""Small neural modules shared by the v0 offline RL baselines.

The benchmark has a single observation contract (``state_vector``) and a
categorical action contract.  Keeping the IQL/CQL containers here makes the
trainers and policy loaders use exactly the same networks and checkpoint
format.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.mlp import build_actor_critic, build_mlp, build_q_network, policy_hidden_dims


class IQLAgent(nn.Module):
    """Discrete IQL actor, twin Q-functions and value function."""

    def __init__(
        self,
        config: dict,
        *,
        obs_dim: int,
        num_states: int | None,
        num_actions: int,
        obs_backend: str,
    ) -> None:
        super().__init__()
        self.actor = build_actor_critic(
            config,
            obs_dim=obs_dim,
            num_states=num_states,
            num_actions=num_actions,
            obs_backend=obs_backend,
        )
        self.q1 = build_q_network(
            config,
            obs_dim=obs_dim,
            num_states=num_states,
            num_actions=num_actions,
            obs_backend=obs_backend,
        )
        self.q2 = build_q_network(
            config,
            obs_dim=obs_dim,
            num_states=num_states,
            num_actions=num_actions,
            obs_backend=obs_backend,
        )
        # The v0 leaderboard uses MLP/state_vector recipes.  For the retained
        # tabular compatibility backend the actor-critic's critic is a valid
        # value head as well, so this stays a small independent MLP only for
        # vector observations.
        if obs_backend == "discrete_id":
            self.value = nn.Linear(num_states or obs_dim, 1, bias=False)
        else:
            self.value = build_mlp(obs_dim, policy_hidden_dims(config), 1)


class CQLAgent(nn.Module):
    """Discrete CQL twin Q-functions."""

    def __init__(
        self,
        config: dict,
        *,
        obs_dim: int,
        num_states: int | None,
        num_actions: int,
        obs_backend: str,
    ) -> None:
        super().__init__()
        self.q1 = build_q_network(
            config,
            obs_dim=obs_dim,
            num_states=num_states,
            num_actions=num_actions,
            obs_backend=obs_backend,
        )
        self.q2 = build_q_network(
            config,
            obs_dim=obs_dim,
            num_states=num_states,
            num_actions=num_actions,
            obs_backend=obs_backend,
        )


def build_iql_agent(config: dict, *, obs_dim: int, num_states: int | None, num_actions: int, obs_backend: str) -> IQLAgent:
    return IQLAgent(
        config,
        obs_dim=obs_dim,
        num_states=num_states,
        num_actions=num_actions,
        obs_backend=obs_backend,
    )


def build_cql_agent(config: dict, *, obs_dim: int, num_states: int | None, num_actions: int, obs_backend: str) -> CQLAgent:
    return CQLAgent(
        config,
        obs_dim=obs_dim,
        num_states=num_states,
        num_actions=num_actions,
        obs_backend=obs_backend,
    )
