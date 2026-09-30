"""Shared model-based architectures for raw_model, latent_model, and reconstructive_model."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from benchtwm.models.mlp import build_mlp


class RawDynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int,
        action_embed_dim: int,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.action_embed_dim = int(action_embed_dim)
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        self.predictor = build_mlp(obs_dim + action_embed_dim, [hidden_dim, hidden_dim], obs_dim)
        self.reward_head = build_mlp(obs_dim + action_embed_dim, [hidden_dim], 1)
        self.continue_head = build_mlp(obs_dim + action_embed_dim, [hidden_dim], 1)

    def rollout_state_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def predict_from_obs(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        action_embed = self.action_embedding(action)
        joint = torch.cat([obs, action_embed], dim=-1)
        prediction = self.predictor(joint)
        return {
            "prediction": prediction,
            "next_state": prediction,
            "reward": self.reward_head(joint).squeeze(-1),
            "continue_logit": self.continue_head(joint).squeeze(-1),
        }

    def rollout_step(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.predict_from_obs(state, action)


class LatentDynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int,
        latent_dim: int,
        action_embed_dim: int,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.encoder = build_mlp(obs_dim, [hidden_dim, hidden_dim], latent_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        self.transition = build_mlp(latent_dim + action_embed_dim, [hidden_dim, hidden_dim], latent_dim)
        self.reward_head = build_mlp(latent_dim + action_embed_dim, [hidden_dim], 1)
        self.continue_head = build_mlp(latent_dim + action_embed_dim, [hidden_dim], 1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    @torch.no_grad()
    def target_encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(obs)

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        decay = float(decay)
        for target_param, source_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)

    def rollout_state_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encode(obs)

    def predict_from_latent(self, latent: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        action_embed = self.action_embedding(action)
        joint = torch.cat([latent, action_embed], dim=-1)
        next_latent = self.transition(joint)
        return {
            "prediction": next_latent,
            "next_state": next_latent,
            "reward": self.reward_head(joint).squeeze(-1),
            "continue_logit": self.continue_head(joint).squeeze(-1),
        }

    def predict_from_obs(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(obs)
        result = self.predict_from_latent(latent, action)
        result["latent"] = latent
        return result

    def rollout_step(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.predict_from_latent(state, action)


class ReconstructiveLatentModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int,
        latent_dim: int,
        action_embed_dim: int,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.encoder = build_mlp(obs_dim, [hidden_dim, hidden_dim], latent_dim)
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        self.transition = build_mlp(latent_dim + action_embed_dim, [hidden_dim, hidden_dim], latent_dim)
        self.decoder = build_mlp(latent_dim, [hidden_dim, hidden_dim], obs_dim)
        self.reward_head = build_mlp(latent_dim + action_embed_dim, [hidden_dim], 1)
        self.continue_head = build_mlp(latent_dim + action_embed_dim, [hidden_dim], 1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def rollout_state_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encode(obs)

    def predict_from_latent(self, latent: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        action_embed = self.action_embedding(action)
        joint = torch.cat([latent, action_embed], dim=-1)
        next_latent = self.transition(joint)
        reconstruction = self.decode(next_latent)
        return {
            "prediction": next_latent,
            "next_state": next_latent,
            "reconstruction": reconstruction,
            "reward": self.reward_head(joint).squeeze(-1),
            "continue_logit": self.continue_head(joint).squeeze(-1),
        }

    def predict_from_obs(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(obs)
        result = self.predict_from_latent(latent, action)
        result["latent"] = latent
        return result

    def rollout_step(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.predict_from_latent(state, action)
