"""Behavior cloning policy for categorical offline actions."""

from __future__ import annotations

import torch

from benchtwm.envs import make_env
from benchtwm.policies import Policy
from benchtwm.models.mlp import build_actor_critic
from benchtwm.utils.torch_utils import get_torch_device


class BCPolicy(Policy):
    def __init__(self, actor, *, temperature: float = 1.0):
        super().__init__()
        self.actor = actor.eval()
        self.device = next(actor.parameters()).device
        self.temperature = float(temperature)

    def action_probabilities(self, obs, info, deterministic: bool = True):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(info["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor.policy_logits(obs_tensor, mask_tensor).squeeze(0)
            if deterministic:
                probs = torch.zeros_like(logits)
                probs[torch.argmax(logits)] = 1.0
            else:
                probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        return probs.cpu().numpy()


def load_policy(config: dict, checkpoint: dict):
    env = make_env(config)
    actor = build_actor_critic(
        config,
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
    ).to(get_torch_device())
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()
    env.close()
    return BCPolicy(actor, temperature=float(config.get("policy_temperature", 1.0)))
