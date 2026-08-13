"""Unified experiment runner."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.algorithms.registry import get_policy_loader, get_trainer_for_mode
from src.evaluation import evaluate_policy
from src.offline.ope import evaluate_ope, export_ope
from src.operations.artifacts import write_manifest
from src.utils.formatting import output_dir_for_experiment
from src.utils.io import ensure_dir, load_json, load_pickle, save_json, save_pickle
from src.utils.methods import benchmark_track, method_id, task_id, train_mode
from src.utils.progress import progress_write


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
    success_rate = np.asarray(metrics.get("success_rate", metrics["survival_rate"]), dtype=np.float32)
    columns = {
        "episode": np.arange(1, len(metrics["returns"]) + 1, dtype=np.int32),
        "return": np.asarray(metrics["returns"], dtype=np.float32),
        "discounted_return": np.asarray(metrics["discounted_returns"], dtype=np.float32),
        "episode_length": np.asarray(metrics["episode_length"], dtype=np.int32),
        "total_steps": np.asarray(metrics["total_steps"], dtype=np.int32),
        "success_rate": success_rate,
        # Backward-compatible ICU alias. New consumers should use success_rate.
        "survival_rate": np.asarray(metrics["survival_rate"], dtype=np.float32),
        "action_stability_l1": np.asarray(metrics["action_stability_l1"], dtype=np.float32),
        "abrupt_jump_rate": np.asarray(metrics["abrupt_jump_rate"], dtype=np.float32),
    }
    for optional_key in ["offline_updates", "offline_sampled_transitions"]:
        if optional_key in metrics:
            columns[optional_key] = np.asarray(metrics[optional_key], dtype=np.int32)
    for name, values in metrics.get("environment_metrics", {}).items():
        columns[f"env_{name}"] = np.asarray(values, dtype=np.float32)
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
            "mean_success": 0.0,
            "mean_survival": 0.0,
            "mean_length": 0.0,
            "mean_action_stability_l1": 0.0,
            "mean_abrupt_jump_rate": 0.0,
        }
    success_column = "success" if "success" in episodes_df else "survived"
    return {
        "mean_return": float(episodes_df["return"].mean()),
        "mean_success": float(episodes_df[success_column].mean()),
        "mean_survival": float(episodes_df[success_column].mean()),
        "mean_length": float(episodes_df["length"].mean()),
        "mean_action_stability_l1": float(episodes_df["action_stability_l1"].mean()),
        "mean_abrupt_jump_rate": float(episodes_df["abrupt_jump_rate"].mean()),
    }


def _save_readable_metrics(params: dict, metrics: dict, output_dir: Path, eval_summary: dict | None) -> None:
    extra_summary = _summarize_extra_metrics(metrics["extra_metrics"])
    environment_metric_summary = _summarize_extra_metrics(metrics.get("environment_metrics", {}))
    payload = {
        "algo": params["algo"],
        "method_id": method_id(params),
        "train_mode": train_mode(params),
        "task_id": str(params.get("task_id", task_id(params))),
        "track": str(params.get("track", benchmark_track(params))),
        "seed": int(params["seed"]),
        "train_summary": metrics["summary"],
        "extra_metrics_summary": extra_summary,
        "environment_metrics_summary": environment_metric_summary,
        "timing_summary": metrics.get("timing", {}),
    }
    if eval_summary is not None:
        payload["eval_summary"] = eval_summary
    save_json(payload, output_dir / "metrics_summary.json")

    flat_summary = {
        "algo": params["algo"],
        "method_id": method_id(params),
        "train_mode": train_mode(params),
        "task_id": str(params.get("task_id", task_id(params))),
        "track": str(params.get("track", benchmark_track(params))),
        "seed": int(params["seed"]),
        **_flatten_summary("train", metrics["summary"]),
    }
    for name, values in extra_summary.items():
        for stat_name, stat_value in values.items():
            flat_summary[f"extra_{name}_{stat_name}"] = stat_value
    for name, values in environment_metric_summary.items():
        for stat_name, stat_value in values.items():
            flat_summary[f"train_env_{name}_{stat_name}"] = stat_value
    for name, value in metrics.get("timing", {}).items():
        flat_summary[f"timing_{name}"] = value
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


def _phase(params: dict, message: str) -> None:
    if bool(params.get("show_progress", True)):
        progress_write(f"[phase] {message}")


def run_experiment(
    params: dict,
    force_run: bool = False,
    *,
    output_dir: str | Path | None = None,
    manifest: dict | None = None,
    quiet: bool = False,
):
    """Train, evaluate, and serialize one fully resolved experiment.

    The CLI supplies an explicit clean artifact directory. Library callers may
    omit it and receive an isolated ``artifacts/adhoc`` location.
    """

    params = dict(params)
    params.setdefault("train_mode", "online")
    params.setdefault("method_id", method_id(params))
    output_dir = Path(output_dir) if output_dir is not None else output_dir_for_experiment(params)
    ensure_dir(output_dir)
    metrics_path = output_dir / "metrics.pkl"
    if metrics_path.exists() and not force_run:
        metrics = load_pickle(metrics_path)
        eval_summary = _load_eval_summary_from_export(output_dir)
        _save_readable_metrics(params, metrics, output_dir, eval_summary)
        if manifest is not None:
            write_manifest(output_dir, manifest, status="completed")
        if not quiet:
            print(f"Already done: {output_dir}")
            _print_summary("Train summary:", metrics["summary"])
            if eval_summary is not None:
                _print_summary("Eval summary:", eval_summary)
        return output_dir

    save_json(params, output_dir / "config.json")
    if manifest is not None:
        write_manifest(output_dir, manifest, status="running")
    try:
        timing: dict[str, float] = {}
        optimization_started = time.perf_counter()
        _phase(params, f"optimization | algorithm={params['algo']} mode={train_mode(params)}")
        trainer = get_trainer_for_mode(params["algo"], train_mode(params))
        result = trainer(params, output_dir)
        timing["optimization_seconds"] = float(time.perf_counter() - optimization_started)

        training_summary = result["metrics"].get("summary", {})
        if bool(training_summary.get("early_stopped", False)):
            _phase(
                params,
                "optimization stopped early | "
                f"monitor=validation_loss(min) step={training_summary.get('early_stop_step')} "
                f"best_step={training_summary.get('best_validation_step')} "
                f"best_loss={float(training_summary.get('best_validation_loss', float('nan'))):.6f}",
            )
        else:
            _phase(params, f"optimization complete | reason={training_summary.get('early_stop_reason', 'completed_budget')}")

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
            evaluation_started = time.perf_counter()
            _phase(params, f"direct evaluation | checkpoint=best episodes={int(params['eval_episodes'])}")
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
            timing["evaluation_seconds"] = float(time.perf_counter() - evaluation_started)
            _phase(
                params,
                "direct evaluation complete | "
                f"return={eval_result['summary'].get('mean_return', 0.0):.6f} "
                f"success={eval_result['summary'].get('mean_success', 0.0):.6f}",
            )
            if bool(params.get("ope_enabled", False)) and params.get("ope_dataset_path"):
                ope_started = time.perf_counter()
                ope_deterministic = bool(params.get("ope_deterministic", params.get("eval_deterministic", True)))
                ope_mode = "deployed_deterministic" if ope_deterministic else "deployed_stochastic"
                _phase(
                    params,
                    "held-out OPE | "
                    f"target={ope_mode} bootstrap={int(params.get('ope_bootstrap_replicates', 200))}",
                )
                ope_summary, ope_bootstrap = evaluate_ope(
                    policy,
                    dataset_path=params["ope_dataset_path"],
                    obs_backend=params["obs_backend"],
                    gamma=float(params.get("gamma", 1.0)),
                    bootstrap_replicates=int(params.get("ope_bootstrap_replicates", 200)),
                    seed=int(params.get("seed", 0)),
                    deterministic=ope_deterministic,
                    target_policy_mode=ope_mode,
                    show_progress=bool(params.get("show_progress", True)),
                    progress_desc="OPE",
                )
                ope_summary["task_id"] = str(params.get("task_id", task_id(params)))
                ope_summary["track"] = str(params.get("track", benchmark_track(params)))
                export_ope(ope_summary, ope_bootstrap, output_dir)
                timing["ope_seconds"] = float(time.perf_counter() - ope_started)
                reliability = ope_summary["reliability"]
                _phase(
                    params,
                    "held-out OPE complete | "
                    f"reliability={reliability['status']} ESS={ope_summary['diagnostics']['trajectory_ess']:.2f} "
                    f"target_zero_rate={ope_summary['diagnostics']['target_zero_probability_rate']:.3f}",
                )
        eval_summary = None if eval_result is None else eval_result["summary"]
        if eval_result is not None:
            timing.update(
                {
                    key: float(value)
                    for key, value in eval_result.get("summary", {}).items()
                    if key in {"mean_action_latency_ms", "planner_calls", "planner_model_evaluations"}
                    and isinstance(value, (int, float, np.number))
                }
            )
        result["metrics"]["timing"] = timing
        # Persist the completed phase timings as part of the canonical metrics artifact.
        save_pickle(result["metrics"], metrics_path)
        _save_readable_metrics(params, result["metrics"], output_dir, eval_summary)
        if manifest is not None:
            write_manifest(output_dir, manifest, status="completed")
        if not quiet:
            print(f"Finished run: {output_dir}")
            _print_summary("Train summary:", result["metrics"]["summary"])
            if eval_summary is not None:
                _print_summary("Eval summary:", eval_summary)
        return output_dir
    except Exception as exc:
        if manifest is not None:
            write_manifest(output_dir, manifest, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
