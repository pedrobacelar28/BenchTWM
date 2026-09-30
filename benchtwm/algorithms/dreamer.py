"""DreamerV3-style recurrent policy integration for BenchTWM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from benchtwm.envs import make_env
from benchtwm.policies import Policy
from benchtwm.models.dreamer import DreamerAgent, DreamerConfig
from benchtwm.utils.torch_utils import get_torch_device


class DreamerPolicy(Policy):
    ope_probability_contract = {
        "status": "exact",
        "message": "Exact masked categorical probabilities from the recurrent Dreamer actor.",
    }

    def __init__(self, agent: DreamerAgent):
        super().__init__()
        self.agent = agent.eval()
        self.device = next(agent.parameters()).device
        self.obs_dim = int(agent.config.obs_dim)
        self.num_actions = int(agent.config.num_actions)
        self.reset_episode()

    def reset_episode(self) -> None:
        self._state = self.agent.initial(1, device=self.device)
        self._previous_action = torch.zeros(1, dtype=torch.long, device=self.device)
        self._is_first = True

    def _distribution(self, obs, info) -> torch.Tensor:
        observation = np.asarray(obs, dtype=np.float32)
        if observation.shape != (self.obs_dim,):
            raise ValueError(f"Dreamer expected state_vector shape {(self.obs_dim,)}, got {observation.shape}")
        mask = np.asarray(info["action_mask"], dtype=np.float32)
        if mask.shape != (self.num_actions,):
            raise ValueError(f"Dreamer expected action_mask shape {(self.num_actions,)}, got {mask.shape}")
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)[None]
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.agent.config.autocast_enabled(self.device),
            ):
                token = self.agent.encoder(tensor)
                previous = F.one_hot(self._previous_action, self.num_actions).float()
                reset = torch.tensor([self._is_first], dtype=torch.bool, device=self.device)
                self._state, feature = self.agent.rssm.observe_step(self._state, token, previous, reset)
                logits = self.agent.actor_logits(
                    self.agent.feature(feature),
                    torch.as_tensor(mask, dtype=torch.float32, device=self.device)[None],
                ).squeeze(0)
            probabilities = torch.softmax(logits.float(), dim=-1)
        self._is_first = False
        return probabilities

    def action_probabilities(self, obs, info, deterministic: bool = True):
        probabilities = self._distribution(obs, info)
        if deterministic:
            deployed = torch.zeros_like(probabilities)
            deployed[torch.argmax(probabilities)] = 1.0
            probabilities = deployed
        return probabilities.cpu().numpy()


    def observe(self, action: int, reward: float) -> None:
        del reward  # The reference policy carry uses the previous action, not reward.
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Dreamer received invalid action {action}")
        self._previous_action = torch.tensor([int(action)], dtype=torch.long, device=self.device)


def load_policy(config: dict, checkpoint: dict) -> DreamerPolicy:
    if str(config.get("obs_backend", "state_vector")) != "state_vector":
        raise ValueError("Dreamer only supports obs_backend='state_vector'")
    env = make_env(config)
    try:
        checkpoint_config = checkpoint.get("dreamer_config")
        if checkpoint_config:
            model_config = DreamerConfig(**checkpoint_config)
            if model_config.obs_dim != env.obs_dim or model_config.num_actions != env.num_actions:
                raise ValueError("Dreamer checkpoint dimensions do not match the requested environment")
        else:
            model_config = DreamerConfig.from_mapping(config, obs_dim=env.obs_dim, num_actions=env.num_actions)
        agent = DreamerAgent(model_config).to(get_torch_device())
        agent.load_state_dict(checkpoint["model_state_dict"])
        agent.eval()
        return DreamerPolicy(agent)
    finally:
        env.close()
