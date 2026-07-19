"""Trajectory-wise importance-sampling off-policy evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.offline.dataset import OfflineTransitionDataset
from src.utils.io import ensure_dir, save_json


ESTIMATORS = ("IS", "WIS", "PDIS", "WPDIS")


def _episodes(dataset: OfflineTransitionDataset, policy, *, deterministic: bool) -> tuple[list[dict], dict[str, float]]:
    data = dataset.data
    records: list[dict] = []
    target_probs: list[float] = []
    for indices in dataset._episode_indices:
        order = indices[np.argsort(data["timesteps"][indices], kind="stable")]
        rewards = []
        ratios = []
        for index in order:
            obs = dataset._obs_array("obs")[index]
            info = {
                "action_mask": data["action_masks"][index],
                "state_id": int(data["state_ids"][index]),
            }
            action = int(data["actions"][index])
            all_probs = np.asarray(policy.action_probabilities(obs, info, deterministic=deterministic), dtype=np.float64)
            if all_probs.shape != (dataset.num_actions,):
                raise ValueError(f"Policy returned action probabilities with shape {all_probs.shape}; expected {(dataset.num_actions,)}")
            probability = float(all_probs[action])
            if probability < 0.0 or not np.isfinite(probability):
                raise ValueError("Policy returned invalid action probabilities for OPE")
            behavior_logprob = float(data["behavior_logprobs"][index])
            if not np.isfinite(behavior_logprob):
                raise ValueError("OPE requires finite behavior_logprobs")
            ratio = 0.0 if probability == 0.0 else float(np.exp(np.log(probability) - behavior_logprob))
            rewards.append(float(data["rewards"][index]))
            ratios.append(ratio)
            target_probs.append(probability)
        records.append({"rewards": np.asarray(rewards, dtype=np.float64), "ratios": np.asarray(ratios, dtype=np.float64)})
    probs = np.asarray(target_probs, dtype=np.float64)
    diagnostics = {
        "num_episodes": float(len(records)),
        "num_transitions": float(len(probs)),
        "support_rate": float(np.mean(probs > 0.0)) if probs.size else 0.0,
        "out_of_support_rate": float(np.mean(probs <= 0.0)) if probs.size else 0.0,
        "mean_target_action_probability": float(probs.mean()) if probs.size else 0.0,
    }
    return records, diagnostics


def _estimate(episodes: list[dict], gamma: float) -> tuple[dict[str, float], dict[str, float]]:
    if not episodes:
        return {name: 0.0 for name in ESTIMATORS}, {"trajectory_ess": 0.0, "zero_weight_rate": 0.0, "max_log_weight": float("-inf")}
    trajectory_weights = []
    trajectory_returns = []
    pdis = []
    per_step_weights: list[list[float]] = []
    per_step_rewards: list[list[float]] = []
    for episode in episodes:
        rewards = episode["rewards"]
        ratios = episode["ratios"]
        discounts = np.power(gamma, np.arange(len(rewards), dtype=np.float64))
        cumulative = np.cumprod(ratios)
        weight = float(cumulative[-1]) if cumulative.size else 1.0
        trajectory_weights.append(weight)
        trajectory_returns.append(float(np.sum(discounts * rewards)))
        pdis.append(float(np.sum(discounts * cumulative * rewards)))
        for step, (step_weight, step_reward) in enumerate(zip(cumulative, rewards)):
            while len(per_step_weights) <= step:
                per_step_weights.append([])
                per_step_rewards.append([])
            per_step_weights[step].append(float(step_weight))
            per_step_rewards[step].append(float(step_reward))
    weights = np.asarray(trajectory_weights, dtype=np.float64)
    returns = np.asarray(trajectory_returns, dtype=np.float64)
    denom = float(weights.sum())
    wpdis = 0.0
    for step, (weights_t, rewards_t) in enumerate(zip(per_step_weights, per_step_rewards)):
        weights_array = np.asarray(weights_t, dtype=np.float64)
        if weights_array.sum() > 0.0:
            wpdis += (gamma**step) * float(np.dot(weights_array / weights_array.sum(), np.asarray(rewards_t, dtype=np.float64)))
    estimates = {
        "IS": float(np.mean(weights * returns)),
        "WIS": float(np.dot(weights, returns) / denom) if denom > 0.0 else 0.0,
        "PDIS": float(np.mean(pdis)),
        "WPDIS": float(wpdis),
    }
    diagnostics = {
        "trajectory_ess": float((weights.sum() ** 2) / np.square(weights).sum()) if np.square(weights).sum() > 0.0 else 0.0,
        "zero_weight_rate": float(np.mean(weights == 0.0)),
        "max_log_weight": float(np.log(weights[weights > 0.0]).max()) if np.any(weights > 0.0) else float("-inf"),
    }
    return estimates, diagnostics


def evaluate_ope(
    policy,
    *,
    dataset_path: str | Path,
    obs_backend: str,
    gamma: float = 1.0,
    bootstrap_replicates: int = 200,
    seed: int = 0,
    deterministic: bool = True,
) -> tuple[dict, pd.DataFrame]:
    dataset = OfflineTransitionDataset(dataset_path, obs_backend=obs_backend, seed=seed)
    episodes, diagnostics = _episodes(dataset, policy, deterministic=deterministic)
    estimates, weight_diagnostics = _estimate(episodes, float(gamma))
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    for replicate in range(int(bootstrap_replicates)):
        indices = rng.integers(0, len(episodes), size=len(episodes))
        sampled = [episodes[int(index)] for index in indices]
        bootstrap_estimates, _ = _estimate(sampled, float(gamma))
        bootstrap_rows.append({"replicate": replicate, **bootstrap_estimates})
    bootstrap = pd.DataFrame(bootstrap_rows, columns=["replicate", *ESTIMATORS])
    estimator_summary = {}
    for name in ESTIMATORS:
        samples = bootstrap[name].to_numpy(dtype=np.float64) if not bootstrap.empty else np.asarray([], dtype=np.float64)
        estimator_summary[name] = {
            "estimate": estimates[name],
            "bootstrap_mean": float(samples.mean()) if samples.size else estimates[name],
            "ci95_low": float(np.quantile(samples, 0.025)) if samples.size else estimates[name],
            "ci95_high": float(np.quantile(samples, 0.975)) if samples.size else estimates[name],
        }
    return {
        "estimators": estimator_summary,
        "diagnostics": {**diagnostics, **weight_diagnostics},
        "config": {
            "dataset_path": str(dataset.path),
            "obs_backend": obs_backend,
            "gamma": float(gamma),
            "bootstrap_replicates": int(bootstrap_replicates),
            "seed": int(seed),
            "deterministic": bool(deterministic),
        },
    }, bootstrap


def export_ope(summary: dict, bootstrap: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    save_json(summary, Path(output_dir) / "ope_summary.json")
    bootstrap.to_csv(Path(output_dir) / "ope_bootstrap.csv", index=False)
