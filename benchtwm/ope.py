"""Trajectory-wise importance-sampling off-policy evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchtwm.data.dataset import OfflineTransitionDataset
from benchtwm.utils.progress import tqdm
from benchtwm.utils.io import ensure_dir, save_json


ESTIMATORS = ("IS", "WIS", "PDIS", "WPDIS")


def _episodes(
    dataset: OfflineTransitionDataset,
    policy,
    *,
    deterministic: bool,
    show_progress: bool,
    progress_desc: str,
) -> tuple[list[dict], dict[str, float]]:
    data = dataset.data
    records: list[dict] = []
    target_probs: list[float] = []
    behavior_probs: list[float] = []
    transition_bar = tqdm(
        total=len(data["actions"]),
        desc=f"{progress_desc}: target ratios",
        unit="transition",
        leave=True,
        disable=not show_progress,
    )
    try:
        for indices in dataset._episode_indices:
            policy.reset_episode()
            order = indices[np.argsort(data["timesteps"][indices], kind="stable")]
            rewards = []
            ratios = []
            for index in order:
                obs = dataset._obs_array("obs")[index]
                mask = data["action_masks"][index]
                info = {"action_mask": mask, "admissible_actions": np.flatnonzero(mask > 0).tolist(), "state_id": int(data["state_ids"][index])}
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
                behavior_probs.append(float(np.exp(behavior_logprob)))
                policy.observe(action, float(data["rewards"][index]))
                transition_bar.update(1)
            records.append({"rewards": np.asarray(rewards, dtype=np.float64), "ratios": np.asarray(ratios, dtype=np.float64)})
    finally:
        transition_bar.close()
    probs = np.asarray(target_probs, dtype=np.float64)
    behavior = np.asarray(behavior_probs, dtype=np.float64)
    target_support = float(np.mean(probs > 0.0)) if probs.size else 0.0
    diagnostics = {
        "num_episodes": float(len(records)),
        "num_transitions": float(len(probs)),
        "behavior_action_probability_min": float(behavior.min()) if behavior.size else 0.0,
        "behavior_action_probability_max": float(behavior.max()) if behavior.size else 0.0,
        "target_logged_action_support_rate": target_support,
        "target_zero_probability_rate": float(np.mean(probs <= 0.0)) if probs.size else 0.0,
        "mean_target_action_probability": float(probs.mean()) if probs.size else 0.0,
    }
    return records, diagnostics


def _estimate(episodes: list[dict], gamma: float) -> tuple[dict[str, float], dict[str, float]]:
    if not episodes:
        return {name: 0.0 for name in ESTIMATORS}, {"trajectory_ess": 0.0, "zero_weight_rate": 0.0, "max_log_weight": float("-inf")}
    trajectory_weights = []
    trajectory_returns = []
    pdis = []
    max_horizon = max((len(episode["rewards"]) for episode in episodes), default=0)
    per_step_weights: list[list[float]] = [[] for _ in range(max_horizon)]
    per_step_rewards: list[list[float]] = [[] for _ in range(max_horizon)]
    for episode in episodes:
        rewards = episode["rewards"]
        ratios = episode["ratios"]
        discounts = np.power(gamma, np.arange(len(rewards), dtype=np.float64))
        cumulative = np.cumprod(ratios)
        weight = float(cumulative[-1]) if cumulative.size else 1.0
        trajectory_weights.append(weight)
        trajectory_returns.append(float(np.sum(discounts * rewards)))
        pdis.append(float(np.sum(discounts * cumulative * rewards)))
        # WPDIS needs every sampled trajectory at every time step. After a
        # trajectory terminates its reward is zero, while its likelihood ratio
        # remains the probability of its complete terminal history. Omitting
        # such rows normalizes only over survivors and biases WPDIS upward when
        # episode lengths differ.
        terminal_weight = float(cumulative[-1]) if cumulative.size else 1.0
        for step in range(max_horizon):
            if step < len(rewards):
                per_step_weights[step].append(float(cumulative[step]))
                per_step_rewards[step].append(float(rewards[step]))
            else:
                per_step_weights[step].append(terminal_weight)
                per_step_rewards[step].append(0.0)
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


def _reliability(diagnostics: dict[str, float]) -> dict[str, float | str]:
    episodes = max(0.0, float(diagnostics.get("num_episodes", 0.0)))
    ess = max(0.0, float(diagnostics.get("trajectory_ess", 0.0)))
    ess_fraction = ess / episodes if episodes else 0.0
    target_zero_rate = float(diagnostics.get("target_zero_probability_rate", 0.0))
    if ess <= 0.0:
        status = "unusable"
        message = "No nonzero trajectory weights; importance-sampling estimates cannot identify the target policy."
    elif ess_fraction >= 0.5 and episodes < 100.0:
        status = "limited"
        message = "Importance weights are well dispersed, but the held-out OPE split has fewer than 100 trajectories."
    elif ess_fraction >= 0.5:
        status = "usable"
        message = "Weight concentration is acceptable for a diagnostic OPE comparison, subject to matching target-policy semantics."
    elif ess < 10.0:
        status = "critical"
        message = "Effective sample size is below 10 trajectories; do not rank policies from these estimates."
    elif ess < 100.0 or ess_fraction < 0.01:
        status = "weak"
        message = "Effective sample size is too small for a reliable policy ranking; treat OPE as a diagnostic only."
    elif ess_fraction < 0.1:
        status = "limited"
        message = "Importance weights are concentrated; compare estimates with direct evaluation and bootstrap intervals."
    else:
        status = "usable"
        message = "Weight concentration is acceptable for a diagnostic OPE comparison, subject to matching target-policy semantics."
    return {
        "status": status,
        "message": message,
        "trajectory_ess_fraction": ess_fraction,
        "target_zero_probability_rate": target_zero_rate,
    }


def evaluate_ope(
    policy,
    *,
    dataset_path: str | Path,
    gamma: float = 1.0,
    bootstrap_replicates: int = 200,
    seed: int = 0,
    deterministic: bool = True,
    target_policy_mode: str | None = None,
    show_progress: bool = False,
    progress_desc: str = "OPE",
) -> tuple[dict, pd.DataFrame]:
    dataset = OfflineTransitionDataset(dataset_path, seed=seed)
    episodes, diagnostics = _episodes(
        dataset,
        policy,
        deterministic=deterministic,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    estimates, weight_diagnostics = _estimate(episodes, float(gamma))
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    bootstrap_iter = tqdm(
        range(int(bootstrap_replicates)),
        desc=f"{progress_desc}: bootstrap",
        unit="rep",
        leave=True,
        disable=not show_progress,
    )
    for replicate in bootstrap_iter:
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
    diagnostics = {**diagnostics, **weight_diagnostics}
    return {
        "schema_version": 2,
        "estimators": estimator_summary,
        "diagnostics": diagnostics,
        "reliability": _reliability(diagnostics),
        "policy_probability_contract": dict(policy.ope_probability_contract),
        "config": {
            "dataset_path": str(dataset.path),
            "gamma": float(gamma),
            "bootstrap_replicates": int(bootstrap_replicates),
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "target_policy_mode": target_policy_mode or ("deterministic" if deterministic else "stochastic"),
        },
    }, bootstrap


def export_ope(summary: dict, bootstrap: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    save_json(summary, Path(output_dir) / "ope_summary.json")
    bootstrap.to_csv(Path(output_dir) / "ope_bootstrap.csv", index=False)
