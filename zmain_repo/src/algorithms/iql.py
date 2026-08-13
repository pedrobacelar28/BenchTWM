"""Discrete Implicit Q-Learning policy loader."""

from __future__ import annotations

import numpy as np
import torch

from src.envs.factory import make_env
from src.models.offline_rl import build_iql_agent
from src.utils.torch_utils import get_torch_device


class IQLPolicy:
    def __init__(self, agent, *, temperature: float = 1.0):
        self.agent = agent.eval()
        self.device = next(agent.parameters()).device
        self.temperature = float(temperature)

    def action_probabilities(self, obs, info, deterministic: bool = True):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(info["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.agent.actor.policy_logits(obs_tensor, mask_tensor).squeeze(0)
            if deterministic:
                probs = torch.zeros_like(logits)
                probs[torch.argmax(logits)] = 1.0
            else:
                probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        return probs.cpu().numpy()

    def act(self, obs, info, deterministic: bool = True) -> int:
        probs = self.action_probabilities(obs, info, deterministic=deterministic)
        return int(np.argmax(probs) if deterministic else np.random.choice(len(probs), p=probs))


def load_policy(config: dict, checkpoint: dict):
    env = make_env(config, obs_backend=config["obs_backend"])
    agent = build_iql_agent(
        config,
        obs_dim=env.obs_dim,
        num_states=env.spec.num_states,
        num_actions=env.num_actions,
        obs_backend=config["obs_backend"],
    ).to(get_torch_device())
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    env.close()
    return IQLPolicy(agent, temperature=float(config.get("policy_temperature", 1.0)))
