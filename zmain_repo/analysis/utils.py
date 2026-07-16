"""Analysis helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from src.bootstrap import bootstrap_icu_helpers
from src.utils.formatting import create_file_name, output_dir_for_experiment
from src.utils.io import load_pickle, save_pickle
from src.utils.json_handling import get_param_iterable, get_param_iterable_runs


def smoothen_runs(data, factor: float = 0.95):
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return data
    smooth = np.zeros_like(data)
    smooth[0] = data[0]
    for idx in range(1, data.size):
        smooth[idx] = smooth[idx - 1] * factor + (1.0 - factor) * data[idx]
    return smooth


def load_run_metrics(experiment: dict):
    output_dir = output_dir_for_experiment(experiment)
    metrics_path = output_dir / "metrics.pkl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return load_pickle(metrics_path)


def aggregate_arrays(arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    if not arrays:
        return {"mean": np.array([]), "stderr": np.array([])}
    min_len = min(len(np.asarray(arr)) for arr in arrays)
    trimmed = np.stack([np.asarray(arr)[:min_len] for arr in arrays], axis=0)
    return {
        "mean": np.mean(trimmed, axis=0),
        "stderr": np.std(trimmed, axis=0) / np.sqrt(trimmed.shape[0]),
    }


def aggregate_metric_dict(metric_dicts: list[dict]) -> dict:
    keys = sorted(metric_dicts[0].keys())
    aggregated = {}
    for key in keys:
        first_value = metric_dicts[0][key]
        if isinstance(first_value, np.ndarray):
            aggregated[key] = aggregate_arrays([metrics[key] for metrics in metric_dicts])
        elif isinstance(first_value, dict):
            aggregated[key] = aggregate_metric_dict([metrics[key] for metrics in metric_dicts])
        elif isinstance(first_value, str):
            unique_values = list(dict.fromkeys(str(metrics[key]) for metrics in metric_dicts))
            aggregated[key] = {
                "value": unique_values[0],
                "unique_values": unique_values,
            }
        else:
            values = np.asarray([metrics[key] for metrics in metric_dicts], dtype=np.float32)
            aggregated[key] = {
                "mean": float(values.mean()),
                "stderr": float(values.std() / np.sqrt(max(1, len(values)))),
            }
    return aggregated


def find_best(json_handle: dict, data: str = "returns", metric: str = "auc", minmax: str = "max"):
    best_score = -np.inf if minmax == "max" else np.inf
    best_params = None
    best_payload = None
    for params in get_param_iterable_runs(json_handle):
        folder, signature = create_file_name(params, sub_folder="processed")
        filename = Path(folder) / f"{signature}.pcsd"
        if not filename.exists():
            continue
        payload = load_pickle(filename)
        mean_curve = np.asarray(payload[data]["mean"])
        score = float(np.mean(mean_curve)) if metric == "auc" else float(np.mean(mean_curve[-max(1, mean_curve.shape[0] // 10):]))
        if (minmax == "max" and score > best_score) or (minmax == "min" and score < best_score):
            best_score = score
            best_params = params
            best_payload = payload
    return best_score, best_params, best_payload


def get_reference_stats(num_episodes: int = 5000, gamma: float = 1.0) -> dict:
    bootstrap_icu_helpers()
    import gymnasium as gym
    import icu_sepsis  # noqa: F401
    import numpy as np
    from icu_sepsis_helpers.utils.mdp import estimate_j_pi, value_iteration

    env = gym.make("Sepsis/ICU-Sepsis-v2")
    dynamics = env.unwrapped.dynamics
    expert_policy_arr = env.unwrapped.expert_policy
    pi_star_arr, _ = value_iteration(dynamics["tx_mat"], dynamics["r_mat"], gamma)

    def _rand_policy(_):
        return np.random.randint(0, env.unwrapped.env_metadata["n_actions"])

    def _expert_policy(state: int):
        return np.random.choice(expert_policy_arr.shape[1], p=expert_policy_arr[state])

    def _pi_star(state: int):
        return pi_star_arr[state]

    stats = {
        "pi_star": {},
        "rand_policy": {},
        "expert_policy": {},
    }
    j_star, j_dict_star = estimate_j_pi(env, _pi_star, num_episodes=num_episodes, gamma=gamma, desc="J_optm")
    j_rand, j_dict_rand = estimate_j_pi(env, _rand_policy, num_episodes=num_episodes, gamma=gamma, desc="J_rand")
    j_data, j_dict_data = estimate_j_pi(env, _expert_policy, num_episodes=num_episodes, gamma=gamma, desc="J_data")
    stats["pi_star"]["j_mean"] = j_star
    stats["pi_star"]["ep_len"] = j_dict_star["mean_ep_len"]
    stats["rand_policy"]["j_mean"] = j_rand
    stats["rand_policy"]["ep_len"] = j_dict_rand["mean_ep_len"]
    stats["expert_policy"]["j_mean"] = j_data
    stats["expert_policy"]["ep_len"] = j_dict_data["mean_ep_len"]
    env.close()
    return stats


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_processed(payload: dict, params: dict) -> Path:
    folder, signature = create_file_name(params, sub_folder="processed")
    path = Path(folder) / f"{signature}.pcsd"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(payload, path)
    return path
