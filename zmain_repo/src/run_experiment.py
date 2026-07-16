"""Unified experiment runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.algorithms.registry import get_policy_loader, get_trainer_for_mode
from src.evaluation import evaluate_policy
from src.utils.formatting import output_dir_for_experiment
from src.utils.io import ensure_dir, load_json, load_pickle, save_json, save_pickle
from src.utils.methods import method_id, train_mode


def _summarize_extra_metrics(extra_metrics: dict) -> dict:
    summary = {}
    for name, values in extra_metrics.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            summary[name] = {"num_points": 0}
            continue
        summary[name] = {
            "last": float(arr[-1]),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "num_points": int(arr.size),
        }
    return summary


def _flatten_summary(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _save_training_history(metrics: dict, output_dir: Path) -> None:
    columns = {
        "episode": np.arange(1, len(metrics["returns"]) + 1, dtype=np.int32),
        "return": np.asarray(metrics["returns"], dtype=np.float32),
        "discounted_return": np.asarray(metrics["discounted_returns"], dtype=np.float32),
        "episode_length": np.asarray(metrics["episode_length"], dtype=np.int32),
        "total_steps": np.asarray(metrics["total_steps"], dtype=np.int32),
        "survival_rate": np.asarray(metrics["survival_rate"], dtype=np.float32),
        "action_stability_l1": np.asarray(metrics["action_stability_l1"], dtype=np.float32),
        "abrupt_jump_rate": np.asarray(metrics["abrupt_jump_rate"], dtype=np.float32),
    }
    for optional_key in ["offline_updates", "offline_sampled_transitions"]:
        if optional_key in metrics:
            columns[optional_key] = np.asarray(metrics[optional_key], dtype=np.int32)
    history_df = pd.DataFrame(columns)
    history_df.to_csv(output_dir / "training_history.csv", index=False)


def _save_extra_metrics(extra_metrics: dict, output_dir: Path) -> None:
    if not extra_metrics:
        return
    max_len = max(len(values) for values in extra_metrics.values())
    columns = {"index": np.arange(1, max_len + 1, dtype=np.int32)}
    for name, values in extra_metrics.items():
        arr = np.asarray(values, dtype=np.float32)
        padded = np.full(max_len, np.nan, dtype=np.float32)
        padded[: arr.size] = arr
        columns[name] = padded
    pd.DataFrame(columns).to_csv(output_dir / "extra_metrics.csv", index=False)


def _save_history_dict(history: dict, output_dir: Path, filename: str) -> None:
    if not history:
        return
    columns = {}
    for key, values in history.items():
        arr = np.asarray(values)
        if arr.size:
            columns[key] = arr
    if columns:
        pd.DataFrame(columns).to_csv(output_dir / filename, index=False)


def _load_eval_summary_from_export(output_dir: Path) -> dict | None:
    eval_summary_path = output_dir / "eval_summary.json"
    if eval_summary_path.exists():
        return load_json(eval_summary_path)
    episodes_path = output_dir / "episodes_summary.csv"
    if not episodes_path.exists():
        return None
    episodes_df = pd.read_csv(episodes_path)
    if episodes_df.empty:
        return {
            "mean_return": 0.0,
            "mean_survival": 0.0,
            "mean_length": 0.0,
            "mean_action_stability_l1": 0.0,
            "mean_abrupt_jump_rate": 0.0,
        }
    return {
        "mean_return": float(episodes_df["return"].mean()),
        "mean_survival": float(episodes_df["survived"].mean()),
        "mean_length": float(episodes_df["length"].mean()),
        "mean_action_stability_l1": float(episodes_df["action_stability_l1"].mean()),
        "mean_abrupt_jump_rate": float(episodes_df["abrupt_jump_rate"].mean()),
    }


def _save_readable_metrics(params: dict, metrics: dict, output_dir: Path, eval_summary: dict | None) -> None:
    extra_summary = _summarize_extra_metrics(metrics["extra_metrics"])
    payload = {
        "algo": params["algo"],
        "method_id": method_id(params),
        "train_mode": train_mode(params),
        "seed": int(params["seed"]),
        "train_summary": metrics["summary"],
        "extra_metrics_summary": extra_summary,
    }
    if eval_summary is not None:
        payload["eval_summary"] = eval_summary
    save_json(payload, output_dir / "metrics_summary.json")

    flat_summary = {
        "algo": params["algo"],
        "method_id": method_id(params),
        "train_mode": train_mode(params),
        "seed": int(params["seed"]),
        **_flatten_summary("train", metrics["summary"]),
    }
    for name, values in extra_summary.items():
        for stat_name, stat_value in values.items():
            flat_summary[f"extra_{name}_{stat_name}"] = stat_value
    if eval_summary is not None:
        flat_summary.update(_flatten_summary("eval", eval_summary))
        save_json(eval_summary, output_dir / "eval_summary.json")

    pd.DataFrame([flat_summary]).to_csv(output_dir / "metrics_summary.csv", index=False)
    _save_training_history(metrics, output_dir)
    _save_extra_metrics(metrics["extra_metrics"], output_dir)
    _save_history_dict(metrics.get("offline_training_history", {}), output_dir, "offline_training_history.csv")
    _save_history_dict(metrics.get("offline_eval_history", {}), output_dir, "offline_eval_history.csv")


def _print_summary(summary_name: str, summary: dict) -> None:
    print(summary_name)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def run_experiment(params: dict, force_run: bool = False):
    params = dict(params)
    params.setdefault("train_mode", "online")
    params.setdefault("method_id", method_id(params))
    output_dir = output_dir_for_experiment(params)
    ensure_dir(output_dir)
    metrics_path = output_dir / "metrics.pkl"
    if metrics_path.exists() and not force_run:
        metrics = load_pickle(metrics_path)
        eval_summary = _load_eval_summary_from_export(output_dir)
        _save_readable_metrics(params, metrics, output_dir, eval_summary)
        print(f"Already done: {output_dir}")
        _print_summary("Train summary:", metrics["summary"])
        if eval_summary is not None:
            _print_summary("Eval summary:", eval_summary)
        return output_dir

    save_json(params, output_dir / "config.json")
    trainer = get_trainer_for_mode(params["algo"], train_mode(params))
    result = trainer(params, output_dir)

    save_pickle(result["metrics"], metrics_path)
    if params.get("save_checkpoints", True):
        torch.save(result["checkpoint_last"], output_dir / "checkpoint_last.pt")
        torch.save(result["checkpoint_best"], output_dir / "checkpoint_best.pt")

    eval_result = None
    should_run_final_eval = bool(params.get("save_trajectories", True)) or train_mode(params) == "offline"
    if should_run_final_eval:
        policy_loader = get_policy_loader(params["algo"])
        best_checkpoint = result["checkpoint_best"]
        policy = policy_loader(params, best_checkpoint)
        eval_result = evaluate_policy(
            policy,
            config=params,
            num_episodes=int(params["eval_episodes"]),
            eval_seed_base=int(params["eval_seed_base"]),
            output_dir=output_dir,
            checkpoint_name="best",
            export_csv=bool(params.get("save_trajectories", True)),
            show_progress=bool(params.get("show_progress", True)),
            progress_desc="final_eval",
        )
    eval_summary = None if eval_result is None else eval_result["summary"]
    _save_readable_metrics(params, result["metrics"], output_dir, eval_summary)
    print(f"Finished run: {output_dir}")
    _print_summary("Train summary:", result["metrics"]["summary"])
    if eval_summary is not None:
        _print_summary("Eval summary:", eval_summary)
    return output_dir
