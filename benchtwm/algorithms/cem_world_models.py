"""Shared CEM world-model losses, rollout diagnostics, and policies."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from benchtwm.envs.factory import make_env
from benchtwm.models.cem_world_models import RawDynamicsModel, ReconstructiveLatentModel, LatentDynamicsModel
from benchtwm.planners.cem import CategoricalCEMPlanner
from benchtwm.utils.torch_utils import as_device_string, get_torch_device


VALID_MODEL_FAMILIES = {"raw_model", "latent_model", "reconstructive_model"}


def _validate_model_family(model_family: str) -> str:
    model_family = str(model_family).lower()
    if model_family not in VALID_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family={model_family!r}")
    return model_family


def build_model(model_family: str, *, obs_dim: int, num_actions: int, hidden_dim: int, action_embed_dim: int, latent_dim: int | None = None):
    model_family = _validate_model_family(model_family)
    if model_family == "raw_model":
        return RawDynamicsModel(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            action_embed_dim=action_embed_dim,
        )
    if latent_dim is None:
        raise ValueError(f"model_family={model_family!r} requires latent_dim")
    if model_family == "latent_model":
        return LatentDynamicsModel(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_embed_dim=action_embed_dim,
        )
    return ReconstructiveLatentModel(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        action_embed_dim=action_embed_dim,
    )


class CEMWorldModelPolicy:
    ope_probability_contract = {
        "status": "diagnostic",
        "message": (
            "CEM action probabilities depend on sampled planning sequences. "
            "They are useful as a diagnostic, but do not represent an exact fixed stochastic deployment policy."
        ),
    }

    def __init__(self, model, planner: CategoricalCEMPlanner, gamma: float):
        self.model = model.eval()
        self.planner = planner
        self.gamma = float(gamma)
        self.device = next(model.parameters()).device
        self.last_plan_stats: dict = {}

    def _score_sequences(self, rollout_state: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            state = rollout_state.unsqueeze(0).repeat(sequences.shape[0], 1)
            scores = torch.zeros(sequences.shape[0], dtype=torch.float32, device=self.device)
            discounts = torch.ones(sequences.shape[0], dtype=torch.float32, device=self.device)
            for t in range(sequences.shape[1]):
                action = sequences[:, t]
                rollout = self.model.rollout_step(state, action)
                scores = scores + discounts * rollout["reward"]
                discounts = discounts * torch.sigmoid(rollout["continue_logit"]) * self.gamma
                state = rollout["next_state"]
            return scores

    def act(self, obs, info, deterministic: bool = True) -> int:
        action, probabilities = self._plan(obs, info)
        if deterministic:
            return action
        return int(torch.multinomial(torch.as_tensor(probabilities, device=self.device), 1).item())

    def _plan(self, obs, info) -> tuple[int, np.ndarray]:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        rollout_state = self.model.rollout_state_from_obs(obs_tensor).squeeze(0)
        action_mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=self.device)
        scorer = lambda sequences: self._score_sequences(rollout_state, sequences)
        action, self.last_plan_stats = self.planner.plan(scorer, first_action_mask=action_mask)
        return int(action), np.asarray(self.last_plan_stats["first_action_probs"], dtype=np.float64)

    def action_probabilities(self, obs, info, deterministic: bool = True) -> np.ndarray:
        action, probabilities = self._plan(obs, info)
        if deterministic:
            out = np.zeros_like(probabilities)
            out[action] = 1.0
            return out
        return probabilities / probabilities.sum()


def _latent_model_target_encoder_decay(args) -> float:
    if not hasattr(args, "target_encoder_decay"):
        raise ValueError("latent_model requires target_encoder_decay when using EMA latent targets")
    decay = float(args.target_encoder_decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"target_encoder_decay must satisfy 0.0 <= decay < 1.0, got {decay}")
    return decay


def _latent_model_target_latent(model: LatentDynamicsModel, next_obs: torch.Tensor, args) -> torch.Tensor:
    _latent_model_target_encoder_decay(args)
    return model.target_encode(next_obs)


@torch.no_grad()
def _state_error_vector(prediction: torch.Tensor, target_obs: torch.Tensor) -> torch.Tensor:
    return (prediction - target_obs).pow(2).mean(dim=-1)


def batch_loss(model, batch, args, *, model_family: str):
    model_family = _validate_model_family(model_family)
    reward_loss_weight = float(getattr(args, "reward_loss_weight", 1.0))
    continue_loss_weight = float(getattr(args, "continue_loss_weight", 1.0))

    if model_family == "raw_model":
        rollout = model.predict_from_obs(batch.obs, batch.action)
        dynamics_loss = F.mse_loss(rollout["prediction"], batch.next_obs)
        prediction_error = dynamics_loss
        total = float(getattr(args, "dynamics_loss_weight", 1.0)) * dynamics_loss
        info = {
            "dynamics_loss": float(dynamics_loss.item()),
            "prediction_error": float(prediction_error.item()),
        }
    elif model_family == "latent_model":
        rollout = model.predict_from_obs(batch.obs, batch.action)
        target_latent = _latent_model_target_latent(model, batch.next_obs, args)
        dynamics_loss = F.mse_loss(rollout["prediction"], target_latent)
        prediction_error = F.mse_loss(rollout["prediction"], target_latent)
        total = float(getattr(args, "dynamics_loss_weight", 1.0)) * dynamics_loss
        info = {
            "dynamics_loss": float(dynamics_loss.item()),
            "prediction_error": float(prediction_error.item()),
        }
    else:
        rollout = model.predict_from_obs(batch.obs, batch.action)
        target_latent = model.encode(batch.next_obs)
        latent_loss = F.mse_loss(rollout["prediction"], target_latent)
        reconstruction_loss = F.mse_loss(rollout["reconstruction"], batch.next_obs)
        prediction_error = reconstruction_loss
        total = float(getattr(args, "latent_loss_weight", 1.0)) * latent_loss
        total = total + float(getattr(args, "reconstruction_loss_weight", 1.0)) * reconstruction_loss
        info = {
            "dynamics_loss": float(latent_loss.item()),
            "reconstruction_loss": float(reconstruction_loss.item()),
            "prediction_error": float(prediction_error.item()),
        }

    reward_loss = F.mse_loss(rollout["reward"], batch.reward)
    continue_loss = F.binary_cross_entropy_with_logits(rollout["continue_logit"], 1.0 - batch.done)
    total = total + reward_loss_weight * reward_loss + continue_loss_weight * continue_loss
    info["reward_loss"] = float(reward_loss.item())
    info["continue_loss"] = float(continue_loss.item())
    return total, info


@torch.no_grad()
def rollout_validation_metrics(model, dataset, args, *, model_family: str) -> dict[str, float]:
    model_family = _validate_model_family(model_family)
    horizon = int(getattr(args, "cem_horizon", 1))
    rollout_steps = sorted({1, 3, 5, min(horizon, 10)})
    batch_size = int(getattr(args, "offline_batch_size", 256))
    gamma = float(getattr(args, "gamma", 1.0))
    metrics: dict[str, float] = {}
    device = next(model.parameters()).device

    for k in rollout_steps:
        totals = {
            "state_error": 0.0,
            "reward_error": 0.0,
            "continue_loss": 0.0,
            "discounted_return_error": 0.0,
            "cumulative_state_error": 0.0,
        }
        total_sequences = 0
        for batch in dataset.iter_sequence_batches(k, batch_size, device=device):
            state = model.rollout_state_from_obs(batch.start_obs)
            pred_discount = torch.ones(batch.start_obs.shape[0], dtype=torch.float32, device=device)
            true_discount = torch.ones(batch.start_obs.shape[0], dtype=torch.float32, device=device)
            pred_return = torch.zeros_like(pred_discount)
            true_return = torch.zeros_like(true_discount)
            cumulative_state_error = torch.zeros_like(pred_discount)
            mean_reward_error = torch.zeros_like(pred_discount)
            mean_continue_loss = torch.zeros_like(pred_discount)
            final_state_error = torch.zeros_like(pred_discount)

            for step_idx in range(k):
                action = batch.actions[:, step_idx]
                rollout = model.rollout_step(state, action)
                target_next_obs = batch.next_obs[:, step_idx, :]
                target_reward = batch.rewards[:, step_idx]
                target_done = batch.dones[:, step_idx]

                if model_family == "latent_model":
                    target_latent = _latent_model_target_latent(model, target_next_obs, args)
                    state_error = (rollout["prediction"] - target_latent).pow(2).mean(dim=-1)
                elif model_family == "reconstructive_model":
                    state_error = _state_error_vector(rollout["reconstruction"], target_next_obs)
                else:
                    state_error = _state_error_vector(rollout["prediction"], target_next_obs)

                reward_error = (rollout["reward"] - target_reward).pow(2)
                continue_loss = F.binary_cross_entropy_with_logits(
                    rollout["continue_logit"],
                    1.0 - target_done,
                    reduction="none",
                )

                cumulative_state_error = cumulative_state_error + state_error
                mean_reward_error = mean_reward_error + reward_error
                mean_continue_loss = mean_continue_loss + continue_loss
                final_state_error = state_error

                pred_return = pred_return + pred_discount * rollout["reward"]
                true_return = true_return + true_discount * target_reward
                pred_discount = pred_discount * torch.sigmoid(rollout["continue_logit"]) * gamma
                true_discount = true_discount * (1.0 - target_done) * gamma
                state = rollout["next_state"]

            sequence_count = int(batch.start_obs.shape[0])
            if sequence_count == 0:
                continue
            totals["state_error"] += float(final_state_error.sum().item())
            totals["reward_error"] += float((mean_reward_error / k).sum().item())
            totals["continue_loss"] += float((mean_continue_loss / k).sum().item())
            totals["discounted_return_error"] += float((pred_return - true_return).pow(2).sum().item())
            totals["cumulative_state_error"] += float(cumulative_state_error.sum().item())
            total_sequences += sequence_count

        denominator = float(total_sequences) if total_sequences > 0 else float("nan")
        for metric_name, total_value in totals.items():
            metrics[f"validation_rollout{k}_{metric_name}"] = total_value / denominator
    return metrics


def make_policy(model, config: dict) -> CEMWorldModelPolicy:
    env = make_env(config)
    planner = CategoricalCEMPlanner(
        horizon=config["cem_horizon"],
        num_actions=env.num_actions,
        samples=config["cem_samples"],
        elites=config["cem_elites"],
        iterations=config["cem_iterations"],
        alpha=config.get("cem_alpha", 0.2),
        device=as_device_string(next(model.parameters()).device),
    )
    env.close()
    return CEMWorldModelPolicy(model, planner, gamma=config.get("gamma", 1.0))


def load_policy(config: dict, checkpoint: dict, *, model_family: str):
    env = make_env(config)
    model = build_model(
        model_family,
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        hidden_dim=config["hidden_dim"],
        action_embed_dim=config["action_embed_dim"],
        latent_dim=config.get("latent_dim"),
    ).to(get_torch_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    env.close()
    return make_policy(model, config)
