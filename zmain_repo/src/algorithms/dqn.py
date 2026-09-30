"""Offline DQN policy loader."""

from __future__ import annotations

import numpy as np
import torch

from src.envs.factory import make_env
from src.models.mlp import build_q_network
from src.utils.torch_utils import get_torch_device


class DQNPolicy:
    def __init__(self, network, *, temperature: float = 1.0):
        self.network = network.eval()
        self.device = next(network.parameters()).device
        self.temperature = float(temperature)

    def action_probabilities(self, obs, info, deterministic: bool = True):
        with torch.no_grad():
            q_values = self.network(
                torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(info["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0),
            ).squeeze(0)
            if deterministic:
                probs = torch.zeros_like(q_values)
                probs[torch.argmax(q_values)] = 1.0
            else:
                probs = torch.softmax(q_values / max(self.temperature, 1e-6), dim=-1)
        return probs.cpu().numpy()

    def act(self, obs, info, deterministic: bool = True) -> int:
        probs = self.action_probabilities(obs, info, deterministic=deterministic)
        if deterministic:
            return int(probs.argmax())
        return int(np.random.choice(len(probs), p=probs))


def load_policy(config: dict, checkpoint: dict):
    env = make_env(config, obs_backend=config["obs_backend"])
    network = build_q_network(
        config,
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
    ).to(get_torch_device())
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    env.close()
    return DQNPolicy(network, temperature=float(config.get("policy_temperature", 1.0)))
