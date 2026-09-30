"""Learned dynamics models with categorical CEM planning (MPC).

Three model families share one planner and the same reward and continuation
heads; they differ only in the state the dynamics operate on:

``raw-model``            predicts the next observation directly.
``latent-model``         predicts the next latent of an EMA target encoder.
``reconstructive-model`` predicts the next latent and decodes the observation.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchtwm.algorithms import register_algorithm
from benchtwm.algorithms.base import MODEL_BASED, OfflineAlgorithm
from benchtwm.algorithms.networks import build_mlp
from benchtwm.planners.cem import CategoricalCEMPlanner
from benchtwm.policies import Policy, one_hot


class _Heads(nn.Module):
    # Modules are created in a fixed order per family so a seed always gives
    # the same initial weights.
    def _init_heads(self, state_dim: int, hidden_dim: int, action_embed_dim: int) -> None:
        self.reward_head = build_mlp(state_dim + action_embed_dim, [hidden_dim], 1)
        self.continue_head = build_mlp(state_dim + action_embed_dim, [hidden_dim], 1)

    def _joint(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([state, self.action_embedding(action)], dim=-1)

    def _outputs(self, joint: torch.Tensor, next_state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "prediction": next_state,
            "next_state": next_state,
            "reward": self.reward_head(joint).squeeze(-1),
            "continue_logit": self.continue_head(joint).squeeze(-1),
        }


class RawDynamicsModel(_Heads):
    def __init__(self, *, obs_dim: int, num_actions: int, hidden_dim: int, action_embed_dim: int):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        self.predictor = build_mlp(obs_dim + action_embed_dim, [hidden_dim, hidden_dim], obs_dim)
        self._init_heads(obs_dim, hidden_dim, action_embed_dim)

    def rollout_state_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def rollout_step(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        joint = self._joint(state, action)
        return self._outputs(joint, self.predictor(joint))

    def predict_from_obs(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.rollout_step(obs, action)


class LatentDynamicsModel(_Heads):
    def __init__(self, *, obs_dim: int, num_actions: int, hidden_dim: int, latent_dim: int, action_embed_dim: int, ema_target: bool = True):
        super().__init__()
        self.encoder = build_mlp(obs_dim, [hidden_dim, hidden_dim], latent_dim)
        if ema_target:
            self.target_encoder = copy.deepcopy(self.encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        self.transition = build_mlp(latent_dim + action_embed_dim, [hidden_dim, hidden_dim], latent_dim)
        if ema_target:
            self._init_heads(latent_dim, hidden_dim, action_embed_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    @torch.no_grad()
    def target_encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(obs)

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        for target, source in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.data.mul_(decay).add_(source.data, alpha=1.0 - decay)

    def rollout_state_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encode(obs)

    def rollout_step(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        joint = self._joint(state, action)
        return self._outputs(joint, self.transition(joint))

    def predict_from_obs(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.rollout_step(self.encode(obs), action)


class ReconstructiveLatentModel(LatentDynamicsModel):
    """Latent dynamics regularized by decoding the next observation (no EMA target)."""

    def __init__(self, *, obs_dim: int, num_actions: int, hidden_dim: int, latent_dim: int, action_embed_dim: int):
        super().__init__(obs_dim=obs_dim, num_actions=num_actions, hidden_dim=hidden_dim, latent_dim=latent_dim, action_embed_dim=action_embed_dim, ema_target=False)
        self.decoder = build_mlp(latent_dim, [hidden_dim, hidden_dim], obs_dim)
        self._init_heads(latent_dim, hidden_dim, action_embed_dim)

    def rollout_step(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().rollout_step(state, action)
        outputs["reconstruction"] = self.decoder(outputs["next_state"])
        return outputs


class CEMWorldModelPolicy(Policy):
    """Model-predictive control: plan with CEM, execute the first action."""

    ope_probability_contract = {
        "status": "diagnostic",
        "message": (
            "CEM action probabilities depend on sampled planning sequences. "
            "They are useful as a diagnostic, but do not represent an exact fixed stochastic deployment policy."
        ),
    }

    def __init__(self, model: nn.Module, planner: CategoricalCEMPlanner, gamma: float):
        super().__init__()
        self.model = model.eval()
        self.planner = planner
        self.gamma = float(gamma)
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def _score_sequences(self, start_state: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        state = start_state.unsqueeze(0).repeat(sequences.shape[0], 1)
        scores = torch.zeros(sequences.shape[0], dtype=torch.float32, device=self.device)
        discounts = torch.ones_like(scores)
        for t in range(sequences.shape[1]):
            rollout = self.model.rollout_step(state, sequences[:, t])
            scores = scores + discounts * rollout["reward"]
            discounts = discounts * torch.sigmoid(rollout["continue_logit"]) * self.gamma
            state = rollout["next_state"]
        return scores

    @torch.no_grad()
    def _plan(self, obs, info) -> tuple[int, np.ndarray]:
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        start_state = self.model.rollout_state_from_obs(obs_tensor).squeeze(0)
        mask = torch.as_tensor(np.asarray(info["action_mask"], dtype=np.float32), device=self.device)
        action, stats = self.planner.plan(lambda sequences: self._score_sequences(start_state, sequences), first_action_mask=mask)
        return int(action), np.asarray(stats["first_action_probs"], dtype=np.float64)

    def action_probabilities(self, obs, info, deterministic: bool = True) -> np.ndarray:
        action, probabilities = self._plan(obs, info)
        if deterministic:
            return one_hot(len(probabilities), action)
        return probabilities / probabilities.sum()

    def act(self, obs, info, deterministic: bool = True) -> int:
        action, probabilities = self._plan(obs, info)
        if deterministic:
            return action
        return int(self.rng.choice(len(probabilities), p=probabilities / probabilities.sum()))


class _CEMWorldModel(OfflineAlgorithm):
    family = MODEL_BASED

    def model_kwargs(self) -> dict:
        return {
            "obs_dim": self.spec.obs_dim,
            "num_actions": self.spec.num_actions,
            "hidden_dim": int(self.config["hidden_dim"]),
            "action_embed_dim": int(self.config["action_embed_dim"]),
        }

    def state_loss(self, rollout: dict, batch) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def state_error(self, rollout: dict, target_obs: torch.Tensor) -> torch.Tensor:
        """Per-sequence state error used by the multi-step rollout diagnostics."""

        raise NotImplementedError

    def loss(self, batch):
        rollout = self.model.predict_from_obs(batch.obs, batch.action)
        total, info = self.state_loss(rollout, batch)
        reward_loss = F.mse_loss(rollout["reward"], batch.reward)
        continue_loss = F.binary_cross_entropy_with_logits(rollout["continue_logit"], 1.0 - batch.done)
        total = total + float(self.config.get("reward_loss_weight", 1.0)) * reward_loss
        total = total + float(self.config.get("continue_loss_weight", 1.0)) * continue_loss
        return total, {**info, "reward_loss": float(reward_loss.item()), "continue_loss": float(continue_loss.item())}

    @torch.no_grad()
    def validate(self, dataset):
        loss, info = super().validate(dataset)
        return loss, {**info, **self.rollout_metrics(dataset)}

    @torch.no_grad()
    def rollout_metrics(self, dataset) -> dict[str, float]:
        """Open-loop k-step prediction errors on logged sequences (model-fit diagnostics)."""

        gamma = float(self.config.get("gamma", 1.0))
        batch_size = int(self.config["offline_batch_size"])
        metrics: dict[str, float] = {}
        for k in sorted({1, 3, 5, min(int(self.config.get("cem_horizon", 1)), 10)}):
            totals = dict.fromkeys(("state_error", "reward_error", "continue_loss", "discounted_return_error", "cumulative_state_error"), 0.0)
            sequences = 0
            for batch in dataset.iter_sequence_batches(k, batch_size, device=self.device):
                count = int(batch.start_obs.shape[0])
                state = self.model.rollout_state_from_obs(batch.start_obs)
                pred_discount = torch.ones(count, device=self.device)
                true_discount = torch.ones(count, device=self.device)
                pred_return = torch.zeros(count, device=self.device)
                true_return = torch.zeros(count, device=self.device)
                cumulative = torch.zeros(count, device=self.device)
                reward_error = torch.zeros(count, device=self.device)
                continue_loss = torch.zeros(count, device=self.device)
                final = torch.zeros(count, device=self.device)
                for t in range(k):
                    rollout = self.model.rollout_step(state, batch.actions[:, t])
                    error = self.state_error(rollout, batch.next_obs[:, t, :])
                    reward, done = batch.rewards[:, t], batch.dones[:, t]
                    cumulative, final = cumulative + error, error
                    reward_error = reward_error + (rollout["reward"] - reward).pow(2)
                    continue_loss = continue_loss + F.binary_cross_entropy_with_logits(rollout["continue_logit"], 1.0 - done, reduction="none")
                    pred_return = pred_return + pred_discount * rollout["reward"]
                    true_return = true_return + true_discount * reward
                    pred_discount = pred_discount * torch.sigmoid(rollout["continue_logit"]) * gamma
                    true_discount = true_discount * (1.0 - done) * gamma
                    state = rollout["next_state"]
                totals["state_error"] += float(final.sum())
                totals["reward_error"] += float((reward_error / k).sum())
                totals["continue_loss"] += float((continue_loss / k).sum())
                totals["discounted_return_error"] += float((pred_return - true_return).pow(2).sum())
                totals["cumulative_state_error"] += float(cumulative.sum())
                sequences += count
            for name, value in totals.items():
                metrics[f"rollout{k}_{name}"] = value / sequences if sequences else float("nan")
        return metrics

    def make_policy(self):
        planner = CategoricalCEMPlanner(
            horizon=int(self.config["cem_horizon"]),
            num_actions=self.spec.num_actions,
            samples=int(self.config["cem_samples"]),
            elites=int(self.config["cem_elites"]),
            iterations=int(self.config["cem_iterations"]),
            alpha=float(self.config.get("cem_alpha", 0.2)),
            device=str(self.device),
        )
        return CEMWorldModelPolicy(self.model, planner, gamma=float(self.config.get("gamma", 1.0)))

    @staticmethod
    def tuning_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "cem_horizon": trial.suggest_int("cem_horizon", 3, 12),
        }


@register_algorithm
class RawModel(_CEMWorldModel):
    name = "raw_model"
    method_id = "raw-model"
    description = "Observation-space dynamics model with CEM planning."
    validation_objective = "validation weighted world-model objective"

    def build(self):
        return RawDynamicsModel(**self.model_kwargs())

    def state_loss(self, rollout, batch):
        loss = F.mse_loss(rollout["prediction"], batch.next_obs)
        return float(self.config.get("dynamics_loss_weight", 1.0)) * loss, {"dynamics_loss": float(loss.item())}

    def state_error(self, rollout, target_obs):
        return (rollout["prediction"] - target_obs).pow(2).mean(dim=-1)


@register_algorithm
class LatentModel(_CEMWorldModel):
    name = "latent_model"
    method_id = "latent-model"
    description = "Latent dynamics model trained against an EMA target encoder, with CEM planning."
    validation_objective = "validation weighted latent-dynamics objective"

    def build(self):
        decay = float(self.config["target_encoder_decay"])
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"target_encoder_decay must satisfy 0 <= decay < 1, got {decay}")
        return LatentDynamicsModel(**self.model_kwargs(), latent_dim=int(self.config["latent_dim"]))

    def state_loss(self, rollout, batch):
        loss = F.mse_loss(rollout["prediction"], self.model.target_encode(batch.next_obs))
        return float(self.config.get("dynamics_loss_weight", 1.0)) * loss, {"dynamics_loss": float(loss.item())}

    def state_error(self, rollout, target_obs):
        return (rollout["prediction"] - self.model.target_encode(target_obs)).pow(2).mean(dim=-1)

    def after_update(self, step: int) -> None:
        self.model.update_target_encoder(float(self.config["target_encoder_decay"]))


@register_algorithm
class ReconstructiveModel(_CEMWorldModel):
    name = "reconstructive_model"
    method_id = "reconstructive-model"
    description = "Latent dynamics model with next-observation reconstruction, with CEM planning."
    validation_objective = "validation weighted reconstruction-and-dynamics objective"

    def build(self):
        return ReconstructiveLatentModel(**self.model_kwargs(), latent_dim=int(self.config["latent_dim"]))

    def state_loss(self, rollout, batch):
        latent_loss = F.mse_loss(rollout["prediction"], self.model.encode(batch.next_obs))
        reconstruction_loss = F.mse_loss(rollout["reconstruction"], batch.next_obs)
        total = float(self.config.get("latent_loss_weight", 1.0)) * latent_loss
        total = total + float(self.config.get("reconstruction_loss_weight", 1.0)) * reconstruction_loss
        return total, {"dynamics_loss": float(latent_loss.item()), "reconstruction_loss": float(reconstruction_loss.item())}

    def state_error(self, rollout, target_obs):
        return (rollout["reconstruction"] - target_obs).pow(2).mean(dim=-1)
