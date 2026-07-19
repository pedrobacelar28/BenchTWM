"""Metric helpers."""

from __future__ import annotations

import numpy as np


ACTION_LEVELS = 5


def decode_action(action_idx: int, action_levels: int = ACTION_LEVELS) -> tuple[int, int]:
    return action_idx // action_levels, action_idx % action_levels


def action_stability(actions: list[int], distance_fn=None, abrupt_threshold: float = 1.0, abrupt_fn=None) -> tuple[float, float]:
    if len(actions) <= 1:
        return 0.0, 0.0
    if distance_fn is not None:
        deltas = [float(distance_fn(previous, current)) for previous, current in zip(actions, actions[1:])]
        abrupt = (
            [float(bool(abrupt_fn(previous, current))) for previous, current in zip(actions, actions[1:])]
            if abrupt_fn is not None
            else [float(delta > float(abrupt_threshold)) for delta in deltas]
        )
        return float(np.mean(deltas)), float(np.mean(abrupt))
    deltas = []
    abrupt = []
    prev_fluid, prev_vaso = decode_action(actions[0])
    for action in actions[1:]:
        fluid, vaso = decode_action(action)
        diff_fluid = abs(fluid - prev_fluid)
        diff_vaso = abs(vaso - prev_vaso)
        deltas.append(diff_fluid + diff_vaso)
        abrupt.append(float(diff_fluid > 1 or diff_vaso > 1))
        prev_fluid, prev_vaso = fluid, vaso
    return float(np.mean(deltas)), float(np.mean(abrupt))


def auc(values) -> float:
    values = np.asarray(values, dtype=np.float32)
    return float(np.mean(values)) if values.size else 0.0


def episodes_to_threshold(values, threshold: float) -> int:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return -1
    running = np.cumsum(values) / np.arange(1, len(values) + 1)
    idx = np.where(running >= threshold)[0]
    return int(idx[0] + 1) if idx.size else -1


def summarize_training(returns, episode_lengths, total_steps, action_stability_l1, abrupt_jump_rate, threshold: float, success_rates=None) -> dict:
    returns = np.asarray(returns, dtype=np.float32)
    episode_lengths = np.asarray(episode_lengths, dtype=np.float32)
    total_steps = np.asarray(total_steps, dtype=np.float32)
    action_stability_l1 = np.asarray(action_stability_l1, dtype=np.float32)
    abrupt_jump_rate = np.asarray(abrupt_jump_rate, dtype=np.float32)
    success_rates = returns > 0.0 if success_rates is None else np.asarray(success_rates, dtype=np.float32)
    if returns.size == 0:
        return {
            "auc_return": 0.0,
            "final_return": 0.0,
            "final_survival_rate": 0.0,
            "final_success_rate": 0.0,
            "mean_episode_length": 0.0,
            "total_env_steps": 0,
            "episodes_to_threshold": -1,
            "mean_action_stability_l1": 0.0,
            "mean_abrupt_jump_rate": 0.0,
        }
    tail = max(1, min(100, returns.size))
    final_success = float(np.mean(success_rates[-tail:])) if success_rates.size else 0.0
    return {
        "auc_return": auc(returns),
        "final_return": float(np.mean(returns[-tail:])),
        "final_survival_rate": final_success,
        "final_success_rate": final_success,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths.size else 0.0,
        "total_env_steps": int(total_steps[-1]) if total_steps.size else 0,
        "episodes_to_threshold": episodes_to_threshold(returns, threshold),
        "mean_action_stability_l1": float(np.mean(action_stability_l1)) if action_stability_l1.size else 0.0,
        "mean_abrupt_jump_rate": float(np.mean(abrupt_jump_rate)) if abrupt_jump_rate.size else 0.0,
    }
