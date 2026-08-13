"""Behavior cloning policy for categorical offline actions."""

from __future__ import annotations

import numpy as np
import torch

from src.envs.factory import make_env
from src.models.mlp import build_actor_critic
from src.utils.torch_utils import get_torch_device


class BCPolicy:
    def __init__(self, actor, *, temperature: float = 1.0):
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

    def act(self, obs, info, deterministic: bool = True) -> int:
        probs = self.action_probabilities(obs, info, deterministic=deterministic)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))


def load_policy(config: dict, checkpoint: dict):
    env = make_env(config, obs_backend=config["obs_backend"])
    actor = build_actor_critic(
        config,
        obs_dim=env.obs_dim,
        num_states=env.spec.num_states,
        num_actions=env.num_actions,
        obs_backend=config["obs_backend"],
    ).to(get_torch_device())
    # BC checkpoints contain only the actor logits.  Loading a full
    # actor-critic state is also accepted for compatibility with old runs.
    state = checkpoint["model_state_dict"]
    if any(key.startswith("actor.") for key in state):
        actor.load_state_dict(state)
    else:
        actor.actor.load_state_dict(state)
    actor.eval()
    env.close()
    return BCPolicy(actor, temperature=float(config.get("policy_temperature", 1.0)))
