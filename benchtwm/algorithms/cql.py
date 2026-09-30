"""Discrete Conservative Q-Learning policy loader."""

from __future__ import annotations

import torch

from benchtwm.envs import make_env
from benchtwm.policies import Policy
from benchtwm.models.offline_rl import build_cql_agent
from benchtwm.utils.torch_utils import get_torch_device


class CQLPolicy(Policy):
    def __init__(self, agent, *, temperature: float = 1.0):
        super().__init__()
        self.agent = agent.eval()
        self.device = next(agent.parameters()).device
        self.temperature = float(temperature)

    def _q_values(self, obs_tensor, mask_tensor):
        q1 = self.agent.q1(obs_tensor, mask_tensor)
        q2 = self.agent.q2(obs_tensor, mask_tensor)
        return torch.minimum(q1, q2)

    def action_probabilities(self, obs, info, deterministic: bool = True):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(info["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self._q_values(obs_tensor, mask_tensor).squeeze(0)
            if deterministic:
                probs = torch.zeros_like(q_values)
                probs[torch.argmax(q_values)] = 1.0
            else:
                probs = torch.softmax(q_values / max(self.temperature, 1e-6), dim=-1)
        return probs.cpu().numpy()


def load_policy(config: dict, checkpoint: dict):
    env = make_env(config)
    agent = build_cql_agent(
        config,
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
    ).to(get_torch_device())
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    env.close()
    return CQLPolicy(agent, temperature=float(config.get("policy_temperature", 1.0)))
