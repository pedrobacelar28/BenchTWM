"""Replay buffers."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    action_mask: torch.Tensor
    next_action_mask: torch.Tensor
    state_id: torch.Tensor
    next_state_id: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: list[dict] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        *,
        obs,
        action,
        reward,
        next_obs,
        done,
        action_mask,
        next_action_mask,
        state_id,
        next_state_id,
    ) -> None:
        item = {
            "obs": np.asarray(obs, dtype=np.float32),
            "action": int(action),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32),
            "done": float(done),
            "action_mask": np.asarray(action_mask, dtype=np.float32),
            "next_action_mask": np.asarray(next_action_mask, dtype=np.float32),
            "state_id": int(state_id),
            "next_state_id": int(next_state_id),
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: str = "cpu") -> ReplayBatch:
        transitions = random.sample(self.buffer, batch_size)
        obs = torch.tensor(np.stack([item["obs"] for item in transitions]), dtype=torch.float32, device=device)
        next_obs = torch.tensor(np.stack([item["next_obs"] for item in transitions]), dtype=torch.float32, device=device)
        action = torch.tensor([item["action"] for item in transitions], dtype=torch.int64, device=device)
        reward = torch.tensor([item["reward"] for item in transitions], dtype=torch.float32, device=device)
        done = torch.tensor([item["done"] for item in transitions], dtype=torch.float32, device=device)
        action_mask = torch.tensor(np.stack([item["action_mask"] for item in transitions]), dtype=torch.float32, device=device)
        next_action_mask = torch.tensor(np.stack([item["next_action_mask"] for item in transitions]), dtype=torch.float32, device=device)
        state_id = torch.tensor([item["state_id"] for item in transitions], dtype=torch.int64, device=device)
        next_state_id = torch.tensor([item["next_state_id"] for item in transitions], dtype=torch.int64, device=device)
        return ReplayBatch(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
            state_id=state_id,
            next_state_id=next_state_id,
        )
