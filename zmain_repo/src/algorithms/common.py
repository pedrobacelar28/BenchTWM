"""Shared helpers for training loops."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional at import time
    SummaryWriter = None

from src.utils.metrics import action_stability, summarize_training


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    if duration <= 0:
        return start_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def random_admissible_action(info: dict) -> int:
    return int(random.choice(info["admissible_actions"]))


def recent_window_mean(values: list[float], window: int = 20) -> float:
    if not values:
        return float("-inf")
    arr = np.asarray(values[-window:], dtype=np.float32)
    return float(arr.mean())


@dataclass
class TrainLogger:
    gamma: float = 1.0
    success_threshold: float = 0.85
    writer: SummaryWriter | None = None
    algo_name: str = "train"
    max_episodes: int | None = None
    show_progress: bool = True
    returns: list[float] = field(default_factory=list)
    discounted_returns: list[float] = field(default_factory=list)
    num_steps: list[int] = field(default_factory=list)
    total_steps: list[int] = field(default_factory=list)
    survival_rate: list[float] = field(default_factory=list)
    action_stability_l1: list[float] = field(default_factory=list)
    abrupt_jump_rate: list[float] = field(default_factory=list)
    extra_metrics: dict[str, list[float]] = field(default_factory=dict)
    progress: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.max_episodes is None:
            return
        self.progress = tqdm(
            total=int(self.max_episodes),
            desc=str(self.algo_name),
            unit="ep",
            dynamic_ncols=True,
            leave=True,
            disable=not (self.show_progress and sys.stderr.isatty()),
        )

    def log_episode(self, episode_idx: int, total_reward: float, episode_length: int, total_env_steps: int, actions: list[int]) -> None:
        self.returns.append(float(total_reward))
        self.discounted_returns.append(float(total_reward * (self.gamma ** max(0, episode_length - 1))))
        self.num_steps.append(int(episode_length))
        self.total_steps.append(int(total_env_steps))
        self.survival_rate.append(float(total_reward > 0))
        stability_l1, abrupt_rate = action_stability(actions)
        self.action_stability_l1.append(stability_l1)
        self.abrupt_jump_rate.append(abrupt_rate)
        if self.writer is not None:
            self.writer.add_scalar("charts/episodic_return", total_reward, episode_idx)
            self.writer.add_scalar("charts/episodic_length", episode_length, episode_idx)
            self.writer.add_scalar("charts/total_steps", total_env_steps, episode_idx)
            self.writer.add_scalar("charts/action_stability_l1", stability_l1, episode_idx)
            self.writer.add_scalar("charts/abrupt_jump_rate", abrupt_rate, episode_idx)
        if self.progress is not None:
            self.progress.update(1)
            self.progress.set_postfix(
                ret=f"{total_reward:.3f}",
                avg20=f"{recent_window_mean(self.returns):.3f}",
                length=int(episode_length),
                steps=int(total_env_steps),
            )

    def log_extra(self, name: str, value: float, step: int | None = None) -> None:
        self.extra_metrics.setdefault(name, []).append(float(value))
        if self.writer is not None and step is not None:
            self.writer.add_scalar(f"extra/{name}", float(value), step)

    def metrics_payload(self) -> dict:
        summary = summarize_training(
            returns=self.returns,
            episode_lengths=self.num_steps,
            total_steps=self.total_steps,
            action_stability_l1=self.action_stability_l1,
            abrupt_jump_rate=self.abrupt_jump_rate,
            threshold=self.success_threshold,
        )
        return {
            "returns": np.asarray(self.returns, dtype=np.float32),
            "discounted_returns": np.asarray(self.discounted_returns, dtype=np.float32),
            "num_steps": np.asarray(self.num_steps, dtype=np.int32),
            "episode_length": np.asarray(self.num_steps, dtype=np.int32),
            "total_steps": np.asarray(self.total_steps, dtype=np.int32),
            "survival_rate": np.asarray(self.survival_rate, dtype=np.float32),
            "action_stability_l1": np.asarray(self.action_stability_l1, dtype=np.float32),
            "abrupt_jump_rate": np.asarray(self.abrupt_jump_rate, dtype=np.float32),
            "extra_metrics": {
                key: np.asarray(values, dtype=np.float32)
                for key, values in self.extra_metrics.items()
            },
            "summary": summary,
        }

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress = None


def make_writer(enabled: bool, log_dir: str | Path):
    if not enabled or SummaryWriter is None:
        return None
    return SummaryWriter(log_dir=str(log_dir))
