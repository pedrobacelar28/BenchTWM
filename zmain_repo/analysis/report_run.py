"""Generate plots, tables, and summaries for a completed run manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from analysis.colors import agent_colors


METHOD_ORDER = [
    "dqn_offline",
    "ppo_logged_offline",
    "raw_model_discrete_cem_offline",
    "raw_model_continuous_cem_offline",
    "latent_model_discrete_cem_offline",
    "latent_model_continuous_cem_offline",
    "reconstructive_model_discrete_cem_offline",
    "reconstructive_model_continuous_cem_offline",
]

DISPLAY_NAMES = {
    "dqn_offline": "DQN",
    "ppo_logged_offline": "PPO",
    "raw_model_discrete_cem_offline": "Raw-D",
    "raw_model_continuous_cem_offline": "Raw-C",
    "latent_model_discrete_cem_offline": "Latent-D",
    "latent_model_continuous_cem_offline": "Latent-C",
    "reconstructive_model_discrete_cem_offline": "Reconstructive-D",
    "reconstructive_model_continuous_cem_offline": "Reconstructive-C",
}

PRIMARY_METRICS = {
    "dqn": ("td_loss", "validation_td_loss"),
    "ppo": ("loss", "validation_loss"),
    "raw_model": ("loss", "validation_loss"),
    "latent_model": ("loss", "validation_loss"),
    "reconstructive_model": ("loss", "validation_loss"),
}

WM_METHODS = {
    "raw_model_discrete_cem_offline",
    "raw_model_continuous_cem_offline",
    "latent_model_discrete_cem_offline",
    "latent_model_continuous_cem_offline",
    "reconstructive_model_discrete_cem_offline",
    "reconstructive_model_continuous_cem_offline",
}

DATASET_ORDER = ["5k", "50k", "100k"]
REQUIRED_ARTIFACTS = [
    "config.json",
    "metrics.pkl",
    "offline_training_history.csv",
    "offline_eval_history.csv",
    "episodes_summary.csv",
    "trajectories.csv",
]
OUTPUT_SUBDIRS = [
    "plots/core",
    "plots/losses",
    "plots/wm",
    "plots/trajectories/action_heatmaps",
    "tables",
    "data",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunArtifacts:
    manifest_row: dict
    config: dict
    metrics: dict
    train_history: pd.DataFrame
    eval_history: pd.DataFrame
    episodes: pd.DataFrame
    trajectories: pd.DataFrame


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def _ensure_dirs(root: Path) -> None:
    for subdir in OUTPUT_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, base_path: Path, formats: list[str]) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(base_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_reference_stats_fast() -> dict | None:
    readme_path = PROJECT_ROOT.parent / "icu-sepsis" / "README.md"
    if not readme_path.exists():
        return None
    text = readme_path.read_text(encoding="utf-8")
    return_match = re.search(
        r"<th>Avg\. return</th>\s*<td>(?P<random>[0-9.]+)</td>\s*<td>(?P<expert>[0-9.]+)</td>\s*<td>(?P<optimal>[0-9.]+)</td>",
        text,
        flags=re.MULTILINE,
    )
    length_match = re.search(
        r"<th>Avg\. episode length</th>\s*<td>(?P<random>[0-9.]+)</td>\s*<td>(?P<expert>[0-9.]+)</td>\s*<td>(?P<optimal>[0-9.]+)</td>",
        text,
        flags=re.MULTILINE,
    )
    if return_match is None or length_match is None:
        return None
    return {
        "rand_policy": {
            "j_mean": float(return_match.group("random")),
            "ep_len": float(length_match.group("random")),
        },
        "expert_policy": {
            "j_mean": float(return_match.group("expert")),
            "ep_len": float(length_match.group("expert")),
        },
        "pi_star": {
            "j_mean": float(return_match.group("optimal")),
            "ep_len": float(length_match.group("optimal")),
        },
    }


def _method_rank(method_id: str) -> int:
    try:
        return METHOD_ORDER.index(method_id)
    except ValueError:
        return len(METHOD_ORDER)


def _method_label(method_id: str) -> str:
    return DISPLAY_NAMES.get(method_id, method_id)


def _method_color(method_id: str, algo: str | None = None) -> str:
    if method_id in agent_colors:
        return agent_colors[method_id]
    if algo is not None and algo in agent_colors:
        return agent_colors[algo]
    return "#333333"


def _dataset_sort_key(dataset: str) -> tuple[int, str]:
    try:
        return (DATASET_ORDER.index(dataset), dataset)
    except ValueError:
        return (len(DATASET_ORDER), dataset)


def _mean_ci95(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean
    stderr = float(array.std(ddof=1) / math.sqrt(array.size))
    half_width = 1.96 * stderr
    return mean, mean - half_width, mean + half_width


def _binomial_ci95(successes: int, total: int) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
    return float(p), float(max(0.0, center - margin)), float(min(1.0, center + margin))


def _curve_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    baseline = array[0]
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return np.ones_like(array)
    return array / baseline


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _action_entropy(actions: pd.Series) -> float:
    counts = actions.value_counts().sort_index().to_numpy(dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _initial_sofa_bins(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    numeric = pd.Series(values).dropna().astype(float)
    if numeric.empty:
        edges = np.array([0.0, 1.0], dtype=np.float64)
        return edges, ["all"]
    quantiles = np.quantile(numeric, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges = np.unique(quantiles)
    if edges.size < 3:
        minimum = float(numeric.min())
        maximum = float(numeric.max())
        if minimum == maximum:
            edges = np.array([minimum, minimum + 1.0], dtype=np.float64)
        else:
            edges = np.linspace(minimum, maximum, 5)
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        labels.append(f"[{left:.2f}, {right:.2f}]")
    return edges, labels


def _load_artifacts(run_dir: Path) -> tuple[list[RunArtifacts], pd.DataFrame]:
    manifest = _read_json(run_dir / "manifest.json")
    artifacts: list[RunArtifacts] = []
    missing_rows: list[dict] = []
    for row in manifest:
        output_dir = Path(row["output_dir"])
        missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]
        if missing:
            missing_rows.append(
                {
                    "dataset_tag": row["dataset_tag"],
                    "algorithm": row["algorithm"],
                    "method_id": row["method_id"],
                    "output_dir": str(output_dir),
                    "missing_artifacts": ";".join(missing),
                }
            )
            _warn(f"Missing artifacts for {row['dataset_tag']}/{row['algorithm']}: {missing}")
            continue
        artifacts.append(
            RunArtifacts(
                manifest_row=row,
                config=_read_json(output_dir / "config.json"),
                metrics=_read_pickle(output_dir / "metrics.pkl"),
                train_history=_read_csv(output_dir / "offline_training_history.csv"),
                eval_history=_read_csv(output_dir / "offline_eval_history.csv"),
                episodes=_read_csv(output_dir / "episodes_summary.csv"),
                trajectories=_read_csv(output_dir / "trajectories.csv"),
            )
        )
    return artifacts, pd.DataFrame(missing_rows)


def _flatten_manifest(artifacts: list[RunArtifacts]) -> pd.DataFrame:
    rows = []
    for run in artifacts:
        row = dict(run.manifest_row)
        row["display_name"] = _method_label(row["method_id"])
        row["obs_backend"] = run.config.get("obs_backend")
        row["seed"] = run.config.get("seed")
        row["train_dataset_path"] = run.config.get("dataset_path")
        row["val_dataset_path"] = run.config.get("val_dataset_path")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_episode_enriched(run: RunArtifacts, sofa_edges: np.ndarray, sofa_labels: list[str]) -> pd.DataFrame:
    meta = run.manifest_row
    episodes = run.episodes.copy()
    trajectories = run.trajectories.copy()
    last_sofa = (
        trajectories.sort_values(["episode_id", "t"])
        .groupby("episode_id", as_index=False)
        .agg(final_sofa=("sofa_score", "last"))
    )
    episodes = episodes.merge(last_sofa, on="episode_id", how="left")
    episodes["final_sofa_delta"] = episodes["final_sofa"] - episodes["initial_sofa"]
    episodes["mean_sofa_delta"] = episodes["mean_sofa"] - episodes["initial_sofa"]
    episodes["dataset_tag"] = meta["dataset_tag"]
    episodes["algorithm"] = meta["algorithm"]
    episodes["method_id"] = meta["method_id"]
    episodes["display_name"] = _method_label(meta["method_id"])
    episodes["run_output_dir"] = meta["output_dir"]
    episodes["seed"] = run.config.get("seed", meta.get("seed"))
    if len(sofa_edges) > 1:
        episodes["initial_sofa_bin"] = pd.cut(
            episodes["initial_sofa"],
            bins=sofa_edges,
            labels=sofa_labels,
            include_lowest=True,
            duplicates="drop",
        ).astype(str)
    else:
        episodes["initial_sofa_bin"] = "all"
    return episodes


def _build_run_level_summary(run: RunArtifacts) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = run.manifest_row
    config = run.config
    metrics = run.metrics
    train_history = run.train_history.copy()
    eval_history = run.eval_history.copy()
    episodes = run.episodes.copy()
    trajectories = run.trajectories.copy()

    trajectories_sorted = trajectories.sort_values(["episode_id", "t"])
    final_sofa_by_episode = trajectories_sorted.groupby("episode_id")["sofa_score"].last()
    episode_final_sofa = episodes[["episode_id", "initial_sofa"]].merge(
        final_sofa_by_episode.rename("final_sofa"),
        on="episode_id",
        how="left",
    )
    action_counts = trajectories["action_idx"].value_counts().sort_index()
    action_total = int(action_counts.sum())
    fluid_stats = trajectories["fluid_bin"].astype(float)
    vaso_stats = trajectories["vaso_bin"].astype(float)

    best_idx = int(eval_history["eval_mean_return"].astype(float).idxmax())
    best_row = eval_history.loc[best_idx]
    last_train_row = train_history.iloc[-1]

    algo = str(config["algo"])
    primary_train_metric, primary_validation_metric = PRIMARY_METRICS[algo]
    primary_train_first = float(train_history[primary_train_metric].iloc[0])
    primary_train_last = float(train_history[primary_train_metric].iloc[-1])
    primary_validation_first = float(train_history[primary_validation_metric].iloc[0])
    primary_validation_last = float(train_history[primary_validation_metric].iloc[-1])

    final_eval_mean_return = float(episodes["return"].mean()) if not episodes.empty else 0.0
    final_eval_mean_survival = float(episodes["survived"].astype(float).mean()) if not episodes.empty else 0.0
    final_eval_mean_length = float(episodes["length"].mean()) if not episodes.empty else 0.0
    final_eval_mean_sofa = float(episodes["mean_sofa"].mean()) if not episodes.empty else 0.0
    final_eval_mean_sofa_delta = float((episodes["mean_sofa"] - episodes["initial_sofa"]).mean()) if not episodes.empty else 0.0
    final_eval_mean_initial_sofa = float(episodes["initial_sofa"].mean()) if not episodes.empty else 0.0
    final_eval_mean_final_sofa = float(episode_final_sofa["final_sofa"].mean()) if not episode_final_sofa.empty else 0.0
    final_eval_mean_final_sofa_delta = float((episode_final_sofa["final_sofa"] - episode_final_sofa["initial_sofa"]).mean()) if not episode_final_sofa.empty else 0.0
    final_eval_mean_stability = float(episodes["action_stability_l1"].mean()) if not episodes.empty else 0.0
    final_eval_mean_jump = float(episodes["abrupt_jump_rate"].mean()) if not episodes.empty else 0.0
    best_validation_step = metrics["summary"].get("best_validation_step")
    best_validation_sampled_transitions = metrics["summary"].get("best_validation_sampled_transitions")
    early_stop_step = metrics["summary"].get("early_stop_step")
    early_stop_sampled_transitions = metrics["summary"].get("early_stop_sampled_transitions")

    run_summary = {
        "dataset_tag": meta["dataset_tag"],
        "algorithm": meta["algorithm"],
        "method_id": meta["method_id"],
        "display_name": _method_label(meta["method_id"]),
        "run_output_dir": meta["output_dir"],
        "resolved_config": meta["resolved_config"],
        "seed": config.get("seed", meta.get("seed")),
        "obs_backend": config.get("obs_backend"),
        "offline_train_steps": float(config.get("offline_train_steps", metrics["summary"].get("offline_train_steps", 0))),
        "offline_eval_interval": float(config.get("offline_eval_interval", 0)),
        "offline_eval_episodes": float(config.get("offline_eval_episodes", 0)),
        "eval_episodes": float(config.get("eval_episodes", 0)),
        "offline_batch_size": float(config.get("offline_batch_size", metrics["summary"].get("offline_batch_size", 0))),
        "gamma": float(config.get("gamma", meta.get("gamma", 1.0))),
        "cem_horizon": config.get("cem_horizon", meta.get("cem_horizon")),
        "train_dataset_transitions": float(metrics["summary"].get("train_dataset_transitions", 0)),
        "train_dataset_episodes": float(metrics["summary"].get("train_dataset_episodes", 0)),
        "val_dataset_transitions": float(metrics["summary"].get("val_dataset_transitions", 0)),
        "val_dataset_episodes": float(metrics["summary"].get("val_dataset_episodes", 0)),
        "effective_epochs_total": float(metrics["summary"].get("effective_epochs", 0)),
        "best_eval_return": float(metrics["summary"].get("best_eval_return", best_row["eval_mean_return"])),
        "best_eval_step": float(best_row["step"]),
        "best_sampled_transitions": float(best_row["sampled_transitions"]),
        "best_fraction_of_budget": _safe_divide(float(best_row["step"]), float(config.get("offline_train_steps", 0))),
        "best_effective_epochs": _safe_divide(float(best_row["sampled_transitions"]), float(metrics["summary"].get("train_dataset_transitions", 0))),
        "best_validation_loss": float(metrics["summary"].get("best_validation_loss", np.nan)),
        "best_validation_step": float(best_validation_step) if best_validation_step is not None else np.nan,
        "best_validation_sampled_transitions": float(best_validation_sampled_transitions) if best_validation_sampled_transitions is not None else np.nan,
        "best_validation_fraction_of_budget": (
            _safe_divide(float(best_validation_step), float(config.get("offline_train_steps", 0)))
            if best_validation_step is not None
            else np.nan
        ),
        "best_validation_effective_epochs": (
            _safe_divide(float(best_validation_sampled_transitions), float(metrics["summary"].get("train_dataset_transitions", 0)))
            if best_validation_sampled_transitions is not None
            else np.nan
        ),
        "early_stopped": bool(metrics["summary"].get("early_stopped", False)),
        "early_stop_step": float(early_stop_step) if early_stop_step is not None else np.nan,
        "early_stop_sampled_transitions": float(early_stop_sampled_transitions) if early_stop_sampled_transitions is not None else np.nan,
        "early_stop_fraction_of_budget": (
            _safe_divide(float(early_stop_step), float(config.get("offline_train_steps", 0)))
            if early_stop_step is not None
            else np.nan
        ),
        "early_stop_reason": metrics["summary"].get("early_stop_reason"),
        "num_validation_checks": float(metrics["summary"].get("num_validation_checks", np.nan)),
        "num_checks_without_improvement": float(metrics["summary"].get("num_checks_without_improvement", np.nan)),
        "final_eval_mean_return": final_eval_mean_return,
        "final_eval_mean_survival": final_eval_mean_survival,
        "final_eval_mean_length": final_eval_mean_length,
        "final_eval_mean_initial_sofa": final_eval_mean_initial_sofa,
        "final_eval_mean_sofa": final_eval_mean_sofa,
        "final_eval_mean_sofa_delta": final_eval_mean_sofa_delta,
        "final_eval_mean_final_sofa": final_eval_mean_final_sofa,
        "final_eval_mean_final_sofa_delta": final_eval_mean_final_sofa_delta,
        "final_eval_mean_action_stability_l1": final_eval_mean_stability,
        "final_eval_mean_abrupt_jump_rate": final_eval_mean_jump,
        "final_eval_gap": float(metrics["summary"].get("best_eval_return", best_row["eval_mean_return"])) - final_eval_mean_return,
        "mean_action_entropy": _action_entropy(trajectories["action_idx"]),
        "fluid_mean": float(fluid_stats.mean()) if not fluid_stats.empty else 0.0,
        "fluid_std": float(fluid_stats.std(ddof=0)) if len(fluid_stats) > 1 else 0.0,
        "vaso_mean": float(vaso_stats.mean()) if not vaso_stats.empty else 0.0,
        "vaso_std": float(vaso_stats.std(ddof=0)) if len(vaso_stats) > 1 else 0.0,
        "primary_train_metric": primary_train_metric,
        "primary_validation_metric": primary_validation_metric,
        "primary_train_first": primary_train_first,
        "primary_train_last": primary_train_last,
        "primary_validation_first": primary_validation_first,
        "primary_validation_last": primary_validation_last,
        "num_eval_checkpoints": int(len(eval_history)),
        "num_eval_episodes": int(len(episodes)),
        "num_trajectory_steps": int(len(trajectories)),
        "is_wm": meta["method_id"] in WM_METHODS,
        "selection_metric": "validation_loss" if "best_validation_loss" in metrics["summary"] else "eval_mean_return",
    }

    component_candidates = [
        "loss",
        "validation_loss",
        "td_loss",
        "validation_td_loss",
        "policy_loss",
        "validation_policy_loss",
        "value_loss",
        "validation_value_loss",
        "bc_loss",
        "validation_bc_loss",
        "entropy",
        "validation_entropy",
        "dynamics_loss",
        "validation_dynamics_loss",
        "prediction_error",
        "validation_prediction_error",
        "reward_loss",
        "validation_reward_loss",
        "continue_loss",
        "validation_continue_loss",
        "reconstruction_loss",
        "validation_reconstruction_loss",
    ]
    for metric_name in component_candidates:
        if metric_name in train_history.columns:
            run_summary[f"last_{metric_name}"] = float(train_history[metric_name].iloc[-1])

    train_long = train_history.melt(id_vars=["step", "sampled_transitions"], var_name="metric", value_name="value")
    train_long["dataset_tag"] = meta["dataset_tag"]
    train_long["method_id"] = meta["method_id"]
    train_long["display_name"] = _method_label(meta["method_id"])
    train_long["algorithm"] = meta["algorithm"]
    train_long["seed"] = run_summary["seed"]
    train_long["split"] = np.where(train_long["metric"].str.startswith("validation_"), "validation", "train")
    train_long["metric_base"] = train_long["metric"].str.removeprefix("validation_")

    eval_long = eval_history.melt(id_vars=["step", "sampled_transitions"], var_name="metric", value_name="value")
    eval_long["dataset_tag"] = meta["dataset_tag"]
    eval_long["method_id"] = meta["method_id"]
    eval_long["display_name"] = _method_label(meta["method_id"])
    eval_long["algorithm"] = meta["algorithm"]
    eval_long["seed"] = run_summary["seed"]
    eval_long["metric_base"] = eval_long["metric"].str.removeprefix("eval_")

    action_usage = action_counts.rename_axis("action_idx").reset_index(name="count")
    if action_total > 0:
        action_usage["proportion"] = action_usage["count"] / action_total
    else:
        action_usage["proportion"] = 0.0
    action_usage["dataset_tag"] = meta["dataset_tag"]
    action_usage["method_id"] = meta["method_id"]
    action_usage["display_name"] = _method_label(meta["method_id"])
    action_usage["algorithm"] = meta["algorithm"]
    action_usage["seed"] = run_summary["seed"]

    heatmap_counts = (
        trajectories.groupby(["fluid_bin", "vaso_bin"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    total_heatmap = float(heatmap_counts["count"].sum())
    heatmap_counts["proportion"] = heatmap_counts["count"] / total_heatmap if total_heatmap > 0 else 0.0
    heatmap_counts["dataset_tag"] = meta["dataset_tag"]
    heatmap_counts["method_id"] = meta["method_id"]
    heatmap_counts["display_name"] = _method_label(meta["method_id"])

    return run_summary, train_long, eval_long, action_usage, heatmap_counts


def _aggregate_group_metrics(run_level_df: pd.DataFrame, episode_enriched_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metric_columns = [
        "final_eval_mean_return",
        "final_eval_mean_survival",
        "final_eval_mean_length",
        "final_eval_mean_sofa",
        "final_eval_mean_sofa_delta",
        "final_eval_mean_final_sofa",
        "final_eval_mean_final_sofa_delta",
        "final_eval_mean_action_stability_l1",
        "final_eval_mean_abrupt_jump_rate",
        "mean_action_entropy",
        "fluid_mean",
        "fluid_std",
        "vaso_mean",
        "vaso_std",
        "best_eval_return",
        "best_eval_step",
        "best_sampled_transitions",
        "best_fraction_of_budget",
        "best_effective_epochs",
        "best_validation_loss",
        "best_validation_step",
        "best_validation_sampled_transitions",
        "best_validation_fraction_of_budget",
        "best_validation_effective_epochs",
        "early_stop_step",
        "early_stop_sampled_transitions",
        "early_stop_fraction_of_budget",
        "num_validation_checks",
        "num_checks_without_improvement",
        "final_eval_gap",
        "primary_train_first",
        "primary_train_last",
        "primary_validation_first",
        "primary_validation_last",
    ]
    for (dataset_tag, method_id), group in run_level_df.groupby(["dataset_tag", "method_id"], sort=False):
        row: dict[str, object] = {
            "dataset_tag": dataset_tag,
            "method_id": method_id,
            "display_name": group["display_name"].iloc[0],
            "algorithm": group["algorithm"].iloc[0],
            "n_runs": int(len(group)),
            "n_seeds": int(group["seed"].nunique()),
            "obs_backend": group["obs_backend"].iloc[0],
            "selection_metric": group["selection_metric"].iloc[0],
            "early_stopped": bool(group["early_stopped"].any()),
            "early_stop_reason": group["early_stop_reason"].iloc[0],
        }
        episodes_group = episode_enriched_df[
            (episode_enriched_df["dataset_tag"] == dataset_tag) & (episode_enriched_df["method_id"] == method_id)
        ]
        error_mode = "seed_stderr" if row["n_seeds"] > 1 else "episode_ci95"
        row["error_mode"] = error_mode

        if row["n_seeds"] > 1:
            for metric in metric_columns:
                values = group[metric].astype(float).to_numpy(dtype=np.float64)
                mean = float(values.mean()) if values.size else 0.0
                stderr = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
                row[metric] = mean
                row[f"{metric}_error"] = stderr
                row[f"{metric}_ci_low"] = mean - 1.96 * stderr
                row[f"{metric}_ci_high"] = mean + 1.96 * stderr
        else:
            single = group.iloc[0]
            for metric in metric_columns:
                row[metric] = float(single[metric])
                row[f"{metric}_error"] = 0.0
                row[f"{metric}_ci_low"] = float(single[metric])
                row[f"{metric}_ci_high"] = float(single[metric])

            continuous_metric_map = {
                "final_eval_mean_length": "length",
                "final_eval_mean_sofa": "mean_sofa",
                "final_eval_mean_sofa_delta": "mean_sofa_delta",
                "final_eval_mean_final_sofa": "final_sofa",
                "final_eval_mean_final_sofa_delta": "final_sofa_delta",
                "final_eval_mean_action_stability_l1": "action_stability_l1",
                "final_eval_mean_abrupt_jump_rate": "abrupt_jump_rate",
            }
            for summary_metric, episode_column in continuous_metric_map.items():
                mean, ci_low, ci_high = _mean_ci95(episodes_group[episode_column].astype(float))
                row[summary_metric] = mean
                row[f"{summary_metric}_error"] = ci_high - mean
                row[f"{summary_metric}_ci_low"] = ci_low
                row[f"{summary_metric}_ci_high"] = ci_high

        return_mean, return_low, return_high = _binomial_ci95(
            int(episodes_group["return"].round().sum()),
            int(len(episodes_group)),
        )
        survive_mean, survive_low, survive_high = _binomial_ci95(
            int(episodes_group["survived"].astype(int).sum()),
            int(len(episodes_group)),
        )
        row["final_eval_mean_return"] = return_mean
        row["final_eval_mean_return_error"] = return_high - return_mean
        row["final_eval_mean_return_ci_low"] = return_low
        row["final_eval_mean_return_ci_high"] = return_high
        row["final_eval_mean_survival"] = survive_mean
        row["final_eval_mean_survival_error"] = survive_high - survive_mean
        row["final_eval_mean_survival_ci_low"] = survive_low
        row["final_eval_mean_survival_ci_high"] = survive_high
        rows.append(row)

    group_level_df = pd.DataFrame(rows)
    return group_level_df.sort_values(
        by=["dataset_tag", "method_id"],
        key=lambda series: series.map(lambda value: _dataset_sort_key(value)[0] if series.name == "dataset_tag" else _method_rank(value)),
    ).reset_index(drop=True)


def _trajectory_quality_summary(group_level_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_tag",
        "method_id",
        "display_name",
        "final_eval_mean_return",
        "final_eval_mean_survival",
        "final_eval_mean_length",
        "final_eval_mean_sofa",
        "final_eval_mean_sofa_delta",
        "final_eval_mean_final_sofa_delta",
        "final_eval_mean_action_stability_l1",
        "final_eval_mean_abrupt_jump_rate",
        "mean_action_entropy",
        "fluid_mean",
        "fluid_std",
        "vaso_mean",
        "vaso_std",
    ]
    return group_level_df[columns].copy()


def _survival_by_initial_sofa_bin(episode_enriched_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = episode_enriched_df.groupby(["dataset_tag", "method_id", "display_name", "initial_sofa_bin"], dropna=False)
    for (dataset_tag, method_id, display_name, sofa_bin), group in grouped:
        mean, low, high = _binomial_ci95(int(group["survived"].astype(int).sum()), int(len(group)))
        rows.append(
            {
                "dataset_tag": dataset_tag,
                "method_id": method_id,
                "display_name": display_name,
                "initial_sofa_bin": sofa_bin,
                "episodes": int(len(group)),
                "mean_survival": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_action_usage(action_usage_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset_tag, method_id, action_idx), group in action_usage_df.groupby(["dataset_tag", "method_id", "action_idx"], sort=False):
        rows.append(
            {
                "dataset_tag": dataset_tag,
                "method_id": method_id,
                "display_name": group["display_name"].iloc[0],
                "action_idx": int(action_idx),
                "count": float(group["count"].sum()),
                "proportion": 0.0,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    totals = result.groupby(["dataset_tag", "method_id"])["count"].transform("sum").replace(0, np.nan)
    result["proportion"] = (result["count"] / totals).fillna(0.0)
    return result


def _build_action_heatmap_table(heatmap_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = heatmap_df.groupby(["dataset_tag", "method_id", "display_name", "fluid_bin", "vaso_bin"], sort=False)
    for (dataset_tag, method_id, display_name, fluid_bin, vaso_bin), group in grouped:
        rows.append(
            {
                "dataset_tag": dataset_tag,
                "method_id": method_id,
                "display_name": display_name,
                "fluid_bin": int(fluid_bin),
                "vaso_bin": int(vaso_bin),
                "count": float(group["count"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    totals = result.groupby(["dataset_tag", "method_id"])["count"].transform("sum").replace(0, np.nan)
    result["proportion"] = (result["count"] / totals).fillna(0.0)
    return result


def _make_run_level_tables(run_manifest_df: pd.DataFrame, run_level_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    loss_snapshot_last_checkpoint = run_level_df[
        [
            "dataset_tag",
            "method_id",
            "display_name",
            "primary_train_metric",
            "primary_validation_metric",
            "primary_train_first",
            "primary_train_last",
            "primary_validation_first",
            "primary_validation_last",
        ]
    ].copy()
    return {
        "run_manifest": run_manifest_df,
        "run_level_summary": run_level_df,
        "loss_snapshot_last_checkpoint": loss_snapshot_last_checkpoint,
    }


def _build_report_leaderboards(group_level_df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "return": ("final_eval_mean_return", False),
        "stability": ("final_eval_mean_action_stability_l1", True),
        "jump": ("final_eval_mean_abrupt_jump_rate", True),
        "length": ("final_eval_mean_length", False),
        "sofa_delta": ("final_eval_mean_final_sofa_delta", True),
        "budget_to_best": ("best_fraction_of_budget", True),
    }
    rows: list[dict] = []
    for dataset_tag, dataset_group in group_level_df.groupby("dataset_tag", sort=False):
        for label, (column, ascending) in metrics.items():
            ranked = dataset_group.sort_values(column, ascending=ascending).reset_index(drop=True)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "dataset_tag": dataset_tag,
                        "leaderboard": label,
                        "rank": rank,
                        "method_id": row["method_id"],
                        "display_name": row["display_name"],
                        "value": float(row[column]),
                    }
                )
    return pd.DataFrame(rows)


def _write_tables(output_root: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(output_root / "tables" / f"{name}.csv", index=False)
    for name, df in tables.items():
        df.to_csv(output_root / "data" / f"{name}.csv", index=False)


def _plot_ranked_bars(group_level_df: pd.DataFrame, output_root: Path, formats: list[str], references: dict | None) -> None:
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 5), dpi=200, sharex=False)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    for axis, dataset_tag in zip(axes, DATASET_ORDER):
        dataset_group = group_level_df[group_level_df["dataset_tag"] == dataset_tag].sort_values("final_eval_mean_return", ascending=True)
        axis.barh(
            dataset_group["display_name"],
            dataset_group["final_eval_mean_return"],
            color=[_method_color(method_id) for method_id in dataset_group["method_id"]],
            xerr=dataset_group["final_eval_mean_return_error"],
            alpha=0.9,
        )
        axis.set_title(dataset_tag)
        axis.set_xlabel("Final Mean Return")
        if references:
            axis.axvline(references["rand_policy"]["j_mean"], linestyle="--", color="#999999", linewidth=1.0)
            axis.axvline(references["expert_policy"]["j_mean"], linestyle="--", color=agent_colors["expert"], linewidth=1.0)
            axis.axvline(references["pi_star"]["j_mean"], linestyle=":", color=agent_colors["optimal"], linewidth=1.0)
    _save_fig(fig, output_root / "plots/core/final_mean_return_by_dataset", formats)

    fig, ax = plt.subplots(1, figsize=(11, 5), dpi=200)
    x = np.arange(len(DATASET_ORDER))
    width = 0.1
    for index, method_id in enumerate(METHOD_ORDER):
        subset = group_level_df[group_level_df["method_id"] == method_id].set_index("dataset_tag").reindex(DATASET_ORDER)
        ax.bar(
            x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
            subset["final_eval_mean_return"],
            width=width,
            label=_method_label(method_id),
            color=_method_color(method_id),
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Final Mean Return")
    ax.legend(ncol=4, fontsize=8)
    _save_fig(fig, output_root / "plots/core/final_mean_return_grouped", formats)

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 5), dpi=200, sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    for axis, dataset_tag in zip(axes, DATASET_ORDER):
        dataset_group = group_level_df[group_level_df["dataset_tag"] == dataset_tag].sort_values("final_eval_mean_return", ascending=False)
        axis.bar(
            dataset_group["display_name"],
            dataset_group["final_eval_mean_return"],
            color=[_method_color(method_id) for method_id in dataset_group["method_id"]],
            alpha=0.9,
        )
        if references:
            axis.axhline(references["rand_policy"]["j_mean"], linestyle="--", color="#999999", linewidth=1.0, label="random")
            axis.axhline(references["expert_policy"]["j_mean"], linestyle="--", color=agent_colors["expert"], linewidth=1.0, label="expert")
            axis.axhline(references["pi_star"]["j_mean"], linestyle=":", color=agent_colors["optimal"], linewidth=1.0, label="optimal")
        axis.set_title(dataset_tag)
        axis.tick_params(axis="x", rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    _save_fig(fig, output_root / "plots/core/final_mean_return_with_refs", formats)


def _plot_eval_curves(eval_history_long_df: pd.DataFrame, run_manifest_df: pd.DataFrame, output_root: Path, formats: list[str]) -> None:
    return_df = eval_history_long_df[eval_history_long_df["metric"] == "eval_mean_return"].copy()
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 4.5), dpi=200, sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    for axis, dataset_tag in zip(axes, DATASET_ORDER):
        dataset_group = return_df[return_df["dataset_tag"] == dataset_tag]
        for method_id, method_group in dataset_group.groupby("method_id", sort=False):
            axis.plot(
                method_group["step"],
                method_group["value"],
                label=_method_label(method_id),
                color=_method_color(method_id),
                linewidth=1.5,
            )
        axis.set_title(dataset_tag)
        axis.set_xlabel("Offline Step")
        axis.set_ylabel("Diagnostic Eval Mean Return")
    axes[-1].legend(loc="lower right", fontsize=8)
    _save_fig(fig, output_root / "plots/core/eval_learning_curves_return", formats)

    normalized = return_df.merge(
        run_manifest_df[["dataset_tag", "method_id", "offline_train_steps"]].drop_duplicates(),
        on=["dataset_tag", "method_id"],
        how="left",
    )
    normalized["normalized_budget"] = normalized["step"] / normalized["offline_train_steps"].replace(0, np.nan)
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 4.5), dpi=200, sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    for axis, dataset_tag in zip(axes, DATASET_ORDER):
        dataset_group = normalized[normalized["dataset_tag"] == dataset_tag]
        for method_id, method_group in dataset_group.groupby("method_id", sort=False):
            axis.plot(
                method_group["normalized_budget"],
                method_group["value"],
                label=_method_label(method_id),
                color=_method_color(method_id),
                linewidth=1.5,
            )
        axis.set_title(dataset_tag)
        axis.set_xlabel("Normalized Budget")
        axis.set_ylabel("Diagnostic Eval Mean Return")
    axes[-1].legend(loc="lower right", fontsize=8)
    _save_fig(fig, output_root / "plots/core/eval_learning_curves_return_normalized_budget", formats)


def _plot_core_summaries(group_level_df: pd.DataFrame, output_root: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(1, figsize=(11, 5), dpi=200)
    x = np.arange(len(DATASET_ORDER))
    width = 0.1
    for index, method_id in enumerate(METHOD_ORDER):
        subset = group_level_df[group_level_df["method_id"] == method_id].set_index("dataset_tag").reindex(DATASET_ORDER)
        ax.bar(
            x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
            subset["final_eval_gap"],
            width=width,
            label=_method_label(method_id),
            color=_method_color(method_id),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Best Periodic Eval - Final Eval")
    ax.legend(ncol=4, fontsize=8)
    _save_fig(fig, output_root / "plots/core/best_vs_final_gap", formats)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=200)
    metrics = [
        ("best_eval_step", "Best Eval Step"),
        ("best_fraction_of_budget", "Best Budget Fraction"),
        ("best_effective_epochs", "Best Effective Epochs"),
    ]
    x = np.arange(len(DATASET_ORDER))
    for axis, (metric, label) in zip(axes, metrics):
        for index, method_id in enumerate(METHOD_ORDER):
            subset = group_level_df[group_level_df["method_id"] == method_id].set_index("dataset_tag").reindex(DATASET_ORDER)
            axis.bar(
                x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
                subset[metric],
                width=width,
                color=_method_color(method_id),
                label=_method_label(method_id) if metric == "best_eval_step" else None,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(DATASET_ORDER)
        axis.set_title(label)
    axes[0].legend(ncol=4, fontsize=8)
    _save_fig(fig, output_root / "plots/core/time_to_best", formats)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=200)
    validation_metrics = [
        ("best_validation_step", "Best Validation Step"),
        ("best_validation_fraction_of_budget", "Best Validation Budget Fraction"),
        ("early_stop_fraction_of_budget", "Early Stop Budget Fraction"),
    ]
    for axis, (metric, label) in zip(axes, validation_metrics):
        for index, method_id in enumerate(METHOD_ORDER):
            subset = group_level_df[group_level_df["method_id"] == method_id].set_index("dataset_tag").reindex(DATASET_ORDER)
            axis.bar(
                x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
                subset[metric],
                width=width,
                color=_method_color(method_id),
                label=_method_label(method_id) if metric == "best_validation_step" else None,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(DATASET_ORDER)
        axis.set_title(label)
    axes[0].legend(ncol=4, fontsize=8)
    _save_fig(fig, output_root / "plots/core/validation_checkpoint_timing", formats)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=200, sharex=True)
    for axis, metric, label in zip(
        axes,
        ["final_eval_mean_action_stability_l1", "final_eval_mean_abrupt_jump_rate"],
        ["Mean Action Stability L1", "Mean Abrupt Jump Rate"],
    ):
        for index, method_id in enumerate(METHOD_ORDER):
            subset = group_level_df[group_level_df["method_id"] == method_id].set_index("dataset_tag").reindex(DATASET_ORDER)
            axis.bar(
                x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
                subset[metric],
                width=width,
                color=_method_color(method_id),
                label=_method_label(method_id) if metric == "final_eval_mean_action_stability_l1" else None,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(DATASET_ORDER)
        axis.set_title(label)
    axes[0].legend(ncol=4, fontsize=8)
    _save_fig(fig, output_root / "plots/core/stability_jump_bars", formats)

    fig, ax = plt.subplots(1, figsize=(11, 5), dpi=200)
    for index, method_id in enumerate(METHOD_ORDER):
        subset = group_level_df[group_level_df["method_id"] == method_id].set_index("dataset_tag").reindex(DATASET_ORDER)
        ax.bar(
            x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
            subset["final_eval_mean_length"],
            width=width,
            color=_method_color(method_id),
            label=_method_label(method_id),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Mean Episode Length")
    ax.legend(ncol=4, fontsize=8)
    _save_fig(fig, output_root / "plots/core/mean_length_bars", formats)

    for metric, filename, ylabel in [
        ("final_eval_mean_action_stability_l1", "pareto_return_vs_stability", "Mean Action Stability L1"),
        ("final_eval_mean_abrupt_jump_rate", "pareto_return_vs_jump", "Mean Abrupt Jump Rate"),
    ]:
        fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 4.5), dpi=200, sharey=False)
        if len(DATASET_ORDER) == 1:
            axes = [axes]
        for axis, dataset_tag in zip(axes, DATASET_ORDER):
            dataset_group = group_level_df[group_level_df["dataset_tag"] == dataset_tag]
            for _, row in dataset_group.iterrows():
                axis.scatter(
                    row[metric],
                    row["final_eval_mean_return"],
                    color=_method_color(row["method_id"]),
                    s=45,
                )
                axis.text(row[metric], row["final_eval_mean_return"], row["display_name"], fontsize=8)
            axis.set_title(dataset_tag)
            axis.set_xlabel(ylabel)
            axis.set_ylabel("Final Mean Return")
        _save_fig(fig, output_root / f"plots/core/{filename}", formats)


def _plot_primary_losses(run_level_df: pd.DataFrame, train_history_long_df: pd.DataFrame, output_root: Path, formats: list[str]) -> None:
    primary_rows: list[dict] = []
    for _, run_row in run_level_df.iterrows():
        subset = train_history_long_df[
            (train_history_long_df["dataset_tag"] == run_row["dataset_tag"])
            & (train_history_long_df["method_id"] == run_row["method_id"])
            & (
                ((train_history_long_df["split"] == "train") & (train_history_long_df["metric"] == run_row["primary_train_metric"]))
                | ((train_history_long_df["split"] == "validation") & (train_history_long_df["metric"] == run_row["primary_validation_metric"]))
            )
        ].copy()
        primary_rows.append(subset)
    primary_df = pd.concat(primary_rows, ignore_index=True) if primary_rows else pd.DataFrame()
    if primary_df.empty:
        return
    primary_df["value_normalized"] = primary_df.groupby(["dataset_tag", "method_id", "split"])["value"].transform(_curve_normalize)

    for split, filename in [("validation", "validation_primary"), ("train", "train_primary")]:
        split_df = primary_df[primary_df["split"] == split]
        fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 4.5), dpi=200, sharey=False)
        if len(DATASET_ORDER) == 1:
            axes = [axes]
        for axis, dataset_tag in zip(axes, DATASET_ORDER):
            dataset_group = split_df[split_df["dataset_tag"] == dataset_tag]
            for method_id, method_group in dataset_group.groupby("method_id", sort=False):
                axis.plot(method_group["step"], method_group["value"], color=_method_color(method_id), label=_method_label(method_id), linewidth=1.4)
            axis.set_title(dataset_tag)
            axis.set_xlabel("Offline Step")
            axis.set_ylabel(f"{split.title()} Primary Metric")
            axis.set_yscale("log")
        axes[-1].legend(loc="best", fontsize=8)
        _save_fig(fig, output_root / f"plots/losses/{filename}_raw", formats)

        fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 4.5), dpi=200, sharey=False)
        if len(DATASET_ORDER) == 1:
            axes = [axes]
        for axis, dataset_tag in zip(axes, DATASET_ORDER):
            dataset_group = split_df[split_df["dataset_tag"] == dataset_tag]
            for method_id, method_group in dataset_group.groupby("method_id", sort=False):
                axis.plot(method_group["step"], method_group["value_normalized"], color=_method_color(method_id), label=_method_label(method_id), linewidth=1.4)
            axis.set_title(dataset_tag)
            axis.set_xlabel("Offline Step")
            axis.set_ylabel(f"{split.title()} Primary Metric / First")
        axes[-1].legend(loc="best", fontsize=8)
        _save_fig(fig, output_root / f"plots/losses/{filename}_normalized", formats)

    pivot = primary_df.pivot_table(
        index=["dataset_tag", "method_id", "display_name", "step"],
        columns="split",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot["gap"] = pivot["validation"] - pivot["train"]
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 4.5), dpi=200, sharey=False)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    for axis, dataset_tag in zip(axes, DATASET_ORDER):
        dataset_group = pivot[pivot["dataset_tag"] == dataset_tag]
        for method_id, method_group in dataset_group.groupby("method_id", sort=False):
            axis.plot(method_group["step"], method_group["gap"], color=_method_color(method_id), label=_method_label(method_id), linewidth=1.4)
        axis.axhline(0.0, color="#666666", linewidth=0.8)
        axis.set_title(dataset_tag)
        axis.set_xlabel("Offline Step")
        axis.set_ylabel("Validation - Train Gap")
    axes[-1].legend(loc="best", fontsize=8)
    _save_fig(fig, output_root / "plots/losses/train_vs_validation_gap", formats)


def _plot_wm_diagnostics(run_level_df: pd.DataFrame, train_history_long_df: pd.DataFrame, output_root: Path, formats: list[str]) -> None:
    wm_df = train_history_long_df[train_history_long_df["method_id"].isin(WM_METHODS)].copy()
    if wm_df.empty:
        return

    component_map = [
        ("validation_dynamics_loss", "Val Dynamics"),
        ("validation_prediction_error", "Val Prediction Error"),
        ("validation_reward_loss", "Val Reward Loss"),
        ("validation_continue_loss", "Val Continue Loss"),
        ("validation_reconstruction_loss", "Val Reconstruction Loss"),
    ]
    fig, axes = plt.subplots(len(DATASET_ORDER), len(component_map), figsize=(22, 10), dpi=200, sharex=False)
    for row_idx, dataset_tag in enumerate(DATASET_ORDER):
        dataset_group = wm_df[wm_df["dataset_tag"] == dataset_tag]
        for col_idx, (metric, title) in enumerate(component_map):
            axis = axes[row_idx, col_idx]
            metric_group = dataset_group[dataset_group["metric"] == metric]
            if metric_group.empty:
                axis.set_axis_off()
                continue
            for method_id, method_group in metric_group.groupby("method_id", sort=False):
                axis.plot(method_group["step"], method_group["value"], color=_method_color(method_id), linewidth=1.3)
            axis.set_title(f"{dataset_tag} · {title}")
    _save_fig(fig, output_root / "plots/wm/wm_validation_components", formats)

    rollout_rows = wm_df[wm_df["metric"].str.startswith("validation_rollout", na=False)].copy()
    if not rollout_rows.empty:
        extracted = rollout_rows["metric"].str.extract(r"validation_rollout(?P<k>\d+)_(?P<name>.+)")
        rollout_rows["rollout_k"] = extracted["k"].astype(int)
        rollout_rows["rollout_name"] = extracted["name"]
        rollout_metric_titles = [
            ("state_error", "Rollout Final State Error"),
            ("reward_error", "Rollout Reward Error"),
            ("continue_loss", "Rollout Continue Loss"),
            ("discounted_return_error", "Rollout Discounted Return Error"),
            ("cumulative_state_error", "Rollout Cumulative State Error"),
        ]
        line_styles = ["-", "--", ":", "-."]
        fig, axes = plt.subplots(len(DATASET_ORDER), len(rollout_metric_titles), figsize=(22, 10), dpi=200, sharex=False)
        for row_idx, dataset_tag in enumerate(DATASET_ORDER):
            dataset_group = rollout_rows[rollout_rows["dataset_tag"] == dataset_tag]
            for col_idx, (metric_name, title) in enumerate(rollout_metric_titles):
                axis = axes[row_idx, col_idx]
                metric_group = dataset_group[dataset_group["rollout_name"] == metric_name]
                if metric_group.empty:
                    axis.set_axis_off()
                    continue
                for method_id, method_group in metric_group.groupby("method_id", sort=False):
                    for style_idx, (rollout_k, k_group) in enumerate(sorted(method_group.groupby("rollout_k", sort=True), key=lambda item: item[0])):
                        axis.plot(
                            k_group["step"],
                            k_group["value"],
                            color=_method_color(method_id),
                            linewidth=1.3,
                            linestyle=line_styles[style_idx % len(line_styles)],
                            label=f"{_method_label(method_id)} · k={rollout_k}" if row_idx == 0 and col_idx == 0 else None,
                        )
                axis.set_title(f"{dataset_tag} · {title}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(loc="best", fontsize=7)
        _save_fig(fig, output_root / "plots/wm/wm_rollout_validation_components", formats)

    component_map_train = [
        ("dynamics_loss", "Train Dynamics"),
        ("prediction_error", "Train Prediction Error"),
        ("reward_loss", "Train Reward Loss"),
        ("continue_loss", "Train Continue Loss"),
        ("reconstruction_loss", "Train Reconstruction Loss"),
    ]
    fig, axes = plt.subplots(len(DATASET_ORDER), len(component_map_train), figsize=(22, 10), dpi=200, sharex=False)
    for row_idx, dataset_tag in enumerate(DATASET_ORDER):
        dataset_group = wm_df[wm_df["dataset_tag"] == dataset_tag]
        for col_idx, (metric, title) in enumerate(component_map_train):
            axis = axes[row_idx, col_idx]
            metric_group = dataset_group[dataset_group["metric"] == metric]
            if metric_group.empty:
                axis.set_axis_off()
                continue
            for method_id, method_group in metric_group.groupby("method_id", sort=False):
                axis.plot(method_group["step"], method_group["value"], color=_method_color(method_id), linewidth=1.3)
            axis.set_title(f"{dataset_tag} · {title}")
    _save_fig(fig, output_root / "plots/wm/wm_train_components", formats)

    snapshot_rows: list[dict] = []
    for _, row in run_level_df[run_level_df["method_id"].isin(WM_METHODS)].iterrows():
        snapshot_rows.append(
            {
                "dataset_tag": row["dataset_tag"],
                "method_id": row["method_id"],
                "display_name": row["display_name"],
                "final_eval_mean_return": row["final_eval_mean_return"],
                "validation_prediction_error": row.get("last_validation_prediction_error", np.nan),
                "validation_reward_loss": row.get("last_validation_reward_loss", np.nan),
                "validation_continue_loss": row.get("last_validation_continue_loss", np.nan),
            }
        )
    snapshot_df = pd.DataFrame(snapshot_rows)
    for metric, filename, xlabel in [
        ("validation_prediction_error", "final_return_vs_prediction_error", "Last Validation Prediction Error"),
        ("validation_reward_loss", "final_return_vs_reward_loss", "Last Validation Reward Loss"),
        ("validation_continue_loss", "final_return_vs_continue_loss", "Last Validation Continue Loss"),
    ]:
        fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(16, 4.5), dpi=200, sharey=True)
        if len(DATASET_ORDER) == 1:
            axes = [axes]
        for axis, dataset_tag in zip(axes, DATASET_ORDER):
            dataset_group = snapshot_df[(snapshot_df["dataset_tag"] == dataset_tag) & (snapshot_df[metric].notna())]
            for _, row in dataset_group.iterrows():
                axis.scatter(row[metric], row["final_eval_mean_return"], color=_method_color(row["method_id"]), s=45)
                axis.text(row[metric], row["final_eval_mean_return"], row["display_name"], fontsize=8)
            axis.set_title(dataset_tag)
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Final Mean Return")
        _save_fig(fig, output_root / f"plots/wm/{filename}", formats)


def _plot_episode_distributions(episode_enriched_df: pd.DataFrame, output_root: Path, formats: list[str]) -> None:
    metrics = [
        ("length", "Episode Length", "episode_length_distribution"),
        ("mean_sofa", "Mean SOFA", "mean_sofa_distribution"),
        ("final_sofa_delta", "Final SOFA Delta", "final_sofa_delta_distribution"),
    ]
    for column, ylabel, filename in metrics:
        fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(17, 4.8), dpi=200, sharey=False)
        if len(DATASET_ORDER) == 1:
            axes = [axes]
        for axis, dataset_tag in zip(axes, DATASET_ORDER):
            dataset_group = episode_enriched_df[episode_enriched_df["dataset_tag"] == dataset_tag]
            labels = []
            values = []
            colors = []
            for method_id in METHOD_ORDER:
                method_group = dataset_group[dataset_group["method_id"] == method_id]
                if method_group.empty:
                    continue
                labels.append(_method_label(method_id))
                values.append(method_group[column].astype(float).to_numpy())
                colors.append(_method_color(method_id))
            if values:
                boxplot = axis.boxplot(values, patch_artist=True, tick_labels=labels, showfliers=False)
                for patch, color in zip(boxplot["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            axis.set_title(dataset_tag)
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", rotation=45)
        _save_fig(fig, output_root / f"plots/trajectories/{filename}", formats)


def _plot_trajectory_summaries(group_level_df: pd.DataFrame, survival_by_bin_df: pd.DataFrame, action_heatmap_df: pd.DataFrame, output_root: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(1, figsize=(11, 5), dpi=200)
    x = np.arange(len(DATASET_ORDER))
    width = 0.1
    for index, method_id in enumerate(METHOD_ORDER):
        subset = group_level_df[group_level_df["method_id"] == method_id].set_index("dataset_tag").reindex(DATASET_ORDER)
        ax.bar(
            x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
            subset["mean_action_entropy"],
            width=width,
            color=_method_color(method_id),
            label=_method_label(method_id),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Action Entropy")
    ax.legend(ncol=4, fontsize=8)
    _save_fig(fig, output_root / "plots/trajectories/action_entropy_bars", formats)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200, sharex=True)
    for axis, metric, label in zip(axes, ["fluid_mean", "vaso_mean"], ["Mean Fluid Bin", "Mean Vaso Bin"]):
        for index, method_id in enumerate(METHOD_ORDER):
            subset = group_level_df[group_level_df["method_id"] == method_id].set_index("dataset_tag").reindex(DATASET_ORDER)
            axis.bar(
                x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
                subset[metric],
                width=width,
                color=_method_color(method_id),
                label=_method_label(method_id) if metric == "fluid_mean" else None,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(DATASET_ORDER)
        axis.set_title(label)
    axes[0].legend(ncol=4, fontsize=8)
    _save_fig(fig, output_root / "plots/trajectories/fluid_vaso_usage_bars", formats)

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(17, 4.8), dpi=200, sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    for axis, dataset_tag in zip(axes, DATASET_ORDER):
        dataset_group = survival_by_bin_df[survival_by_bin_df["dataset_tag"] == dataset_tag]
        bin_order = list(dict.fromkeys(dataset_group["initial_sofa_bin"].tolist()))
        x_positions = np.arange(len(bin_order))
        for method_id in METHOD_ORDER:
            method_group = dataset_group[dataset_group["method_id"] == method_id]
            if method_group.empty:
                continue
            ordered = method_group.set_index("initial_sofa_bin").reindex(bin_order)
            axis.plot(x_positions, ordered["mean_survival"], marker="o", linewidth=1.3, color=_method_color(method_id), label=_method_label(method_id))
        axis.set_xticks(x_positions)
        axis.set_xticklabels(bin_order, rotation=45, ha="right")
        axis.set_title(dataset_tag)
        axis.set_ylabel("Survival by Initial SOFA Bin")
    axes[-1].legend(loc="best", fontsize=8)
    _save_fig(fig, output_root / "plots/trajectories/survival_by_initial_sofa_bin", formats)

    for dataset_tag in DATASET_ORDER:
        dataset_group = action_heatmap_df[action_heatmap_df["dataset_tag"] == dataset_tag]
        for method_id in METHOD_ORDER:
            method_group = dataset_group[dataset_group["method_id"] == method_id]
            if method_group.empty:
                continue
            matrix = np.zeros((5, 5), dtype=np.float64)
            for _, row in method_group.iterrows():
                if 0 <= int(row["fluid_bin"]) < 5 and 0 <= int(row["vaso_bin"]) < 5:
                    matrix[int(row["fluid_bin"]), int(row["vaso_bin"])] = float(row["proportion"])
            fig, ax = plt.subplots(1, figsize=(5, 4.5), dpi=200)
            image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(1e-12, matrix.max()))
            ax.set_xlabel("Vaso Bin")
            ax.set_ylabel("Fluid Bin")
            ax.set_title(f"{dataset_tag} · {_method_label(method_id)}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            _save_fig(fig, output_root / f"plots/trajectories/action_heatmaps/{dataset_tag}__{method_id}", formats)


def _summary_json(
    run_dir: Path,
    group_level_df: pd.DataFrame,
    run_level_df: pd.DataFrame,
    survival_by_bin_df: pd.DataFrame,
    references: dict | None,
    missing_artifacts_df: pd.DataFrame,
) -> dict:
    rankings_by_dataset: dict[str, list[dict]] = {}
    top3_by_metric: dict[str, dict[str, list[dict]]] = {}
    alerts: list[dict] = []

    metrics_for_top3 = {
        "return": ("final_eval_mean_return", False),
        "stability": ("final_eval_mean_action_stability_l1", True),
        "jump": ("final_eval_mean_abrupt_jump_rate", True),
        "length": ("final_eval_mean_length", False),
        "sofa_delta": ("final_eval_mean_final_sofa_delta", True),
        "budget_to_best": ("best_fraction_of_budget", True),
    }

    for dataset_tag, dataset_group in group_level_df.groupby("dataset_tag", sort=False):
        ranked = dataset_group.sort_values("final_eval_mean_return", ascending=False)
        rankings_by_dataset[dataset_tag] = [
            {
                "rank": index + 1,
                "method_id": row["method_id"],
                "display_name": row["display_name"],
                "final_eval_mean_return": float(row["final_eval_mean_return"]),
                "best_eval_return": float(row["best_eval_return"]),
                "final_eval_gap": float(row["final_eval_gap"]),
            }
            for index, (_, row) in enumerate(ranked.iterrows())
        ]

        ref_deltas = {}
        if references:
            for ref_key, ref_field in [("random", "rand_policy"), ("expert", "expert_policy"), ("optimal", "pi_star")]:
                ref_value = float(references[ref_field]["j_mean"])
                ref_deltas[ref_key] = {
                    row["method_id"]: float(row["final_eval_mean_return"] - ref_value) for _, row in ranked.iterrows()
                }

        top3_by_metric[dataset_tag] = {}
        for label, (metric, ascending) in metrics_for_top3.items():
            top_group = dataset_group.sort_values(metric, ascending=ascending).head(3)
            top3_by_metric[dataset_tag][label] = [
                {
                    "method_id": row["method_id"],
                    "display_name": row["display_name"],
                    "value": float(row[metric]),
                }
                for _, row in top_group.iterrows()
            ]

        wm_rows = run_level_df[
            (run_level_df["dataset_tag"] == dataset_tag)
            & (run_level_df["method_id"].isin(WM_METHODS))
            & (run_level_df["last_validation_prediction_error"].notna())
        ]
        prediction_threshold = (
            np.nanmedian(wm_rows["last_validation_prediction_error"].to_numpy(dtype=np.float64))
            if not wm_rows.empty
            else np.nan
        )
        return_median = float(dataset_group["final_eval_mean_return"].median())
        for _, row in dataset_group.iterrows():
            flags: list[str] = []
            if float(row["final_eval_gap"]) > 0.03:
                flags.append("best_vs_final_gap_high")
            if float(row["primary_validation_last"]) >= float(row["primary_validation_first"]):
                flags.append("validation_loss_not_improving")
            if float(row["final_eval_mean_abrupt_jump_rate"]) > 0.20 and float(row["final_eval_mean_return"]) < return_median:
                flags.append("high_jump_low_return")
            if float(row["final_eval_mean_return"]) >= return_median and float(row["final_eval_mean_final_sofa_delta"]) > 0.25:
                flags.append("good_return_bad_sofa")
            if row["method_id"] in WM_METHODS and np.isfinite(prediction_threshold) and float(row.get("last_validation_prediction_error", np.nan)) > prediction_threshold:
                flags.append("wm_prediction_error_high")
            if flags:
                alerts.append(
                    {
                        "dataset_tag": dataset_tag,
                        "method_id": row["method_id"],
                        "display_name": row["display_name"],
                        "flags": flags,
                    }
                )

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "offline_v4_validation_loss_early_stopping",
        "completed_runs": int(len(run_level_df)),
        "missing_runs": int(len(missing_artifacts_df)),
        "references": references,
        "rankings_by_dataset": rankings_by_dataset,
        "top3_by_metric": top3_by_metric,
        "survival_by_initial_sofa_bin_rows": int(len(survival_by_bin_df)),
        "alerts": alerts,
    }


def _write_summary_json(output_root: Path, payload: dict) -> None:
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _environment_spec_for_run(run: RunArtifacts) -> dict:
    path = Path(run.manifest_row["output_dir"]) / "environment_spec.json"
    if path.exists():
        return _read_json(path)
    return {"env_id": run.config.get("env_id", "icu_sepsis"), "metric_names": [], "action_geometry": {"type": "categorical"}}


def _generate_generic_report(
    run_dir: Path,
    output_root: Path,
    formats: list[str],
    artifacts: list[RunArtifacts],
    missing_artifacts_df: pd.DataFrame,
) -> Path:
    """Report canonical exports without ICU-specific SOFA or action-grid assumptions."""
    rows: list[dict] = []
    action_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    ope_rows: list[dict] = []
    for artifact in artifacts:
        spec = _environment_spec_for_run(artifact)
        episodes = artifact.episodes.copy()
        trajectories = artifact.trajectories.copy()
        success_column = "success" if "success" in episodes else "survived"
        row = {
            "env_id": spec["env_id"],
            "dataset_tag": artifact.manifest_row["dataset_tag"],
            "algorithm": artifact.manifest_row["algorithm"],
            "method_id": artifact.manifest_row["method_id"],
            "display_name": _method_label(artifact.manifest_row["method_id"]),
            "seed": artifact.config.get("seed"),
            "mean_return": float(episodes["return"].mean()) if not episodes.empty else 0.0,
            "mean_success": float(episodes[success_column].mean()) if not episodes.empty else 0.0,
            "mean_length": float(episodes["length"].mean()) if not episodes.empty else 0.0,
            "mean_action_stability_l1": float(episodes["action_stability_l1"].mean()) if not episodes.empty else 0.0,
            "mean_abrupt_jump_rate": float(episodes["abrupt_jump_rate"].mean()) if not episodes.empty else 0.0,
        }
        for metric_name in spec.get("metric_names", []):
            column = f"mean_{metric_name}"
            if column in episodes:
                row[column] = float(episodes[column].mean())
                metric_frames.append(
                    pd.DataFrame(
                        {
                            "env_id": spec["env_id"],
                            "dataset_tag": artifact.manifest_row["dataset_tag"],
                            "method_id": artifact.manifest_row["method_id"],
                            "metric": metric_name,
                            "value": episodes[column],
                        }
                    )
                )
        rows.append(row)
        if not trajectories.empty:
            action_counts = trajectories.groupby(["action_idx", "action_label"], as_index=False).size().rename(columns={"size": "count"})
            action_counts["proportion"] = action_counts["count"] / action_counts["count"].sum()
            action_counts["env_id"] = spec["env_id"]
            action_counts["dataset_tag"] = artifact.manifest_row["dataset_tag"]
            action_counts["method_id"] = artifact.manifest_row["method_id"]
            action_frames.append(action_counts)
        ope_path = Path(artifact.manifest_row["output_dir"]) / "ope_summary.json"
        if ope_path.exists():
            payload = _read_json(ope_path)
            for estimator, values in payload.get("estimators", {}).items():
                ope_rows.append(
                    {
                        "env_id": spec["env_id"],
                        "dataset_tag": artifact.manifest_row["dataset_tag"],
                        "method_id": artifact.manifest_row["method_id"],
                        "estimator": estimator,
                        **values,
                        **payload.get("diagnostics", {}),
                    }
                )
    run_summary = pd.DataFrame(rows)
    action_usage = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    metric_values = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    ope_summary = pd.DataFrame(ope_rows)
    _write_tables(
        output_root,
        {
            "generic_run_summary": run_summary,
            "generic_action_usage": action_usage,
            "generic_metric_values": metric_values,
            "ope_summary": ope_summary,
            "missing_artifacts": missing_artifacts_df,
        },
    )
    if not run_summary.empty:
        grouped = run_summary.groupby("method_id", as_index=False).agg(mean_return=("mean_return", "mean"), mean_success=("mean_success", "mean"))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
        axes[0].bar(grouped["method_id"], grouped["mean_return"], color=[_method_color(method) for method in grouped["method_id"]])
        axes[0].set_title("Mean return")
        axes[1].bar(grouped["method_id"], grouped["mean_success"], color=[_method_color(method) for method in grouped["method_id"]])
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Mean success")
        for axis in axes:
            axis.tick_params(axis="x", rotation=35)
        _save_fig(fig, output_root / "plots/core/generic_return_success", formats)
    if not action_usage.empty:
        aggregate_actions = action_usage.groupby("action_label", as_index=False)["proportion"].mean().sort_values("proportion", ascending=False)
        fig, axis = plt.subplots(figsize=(9, 4), dpi=180)
        axis.bar(aggregate_actions["action_label"], aggregate_actions["proportion"], color="#4c78a8")
        axis.set_title("Action usage")
        axis.set_ylabel("Mean proportion")
        axis.tick_params(axis="x", rotation=45)
        _save_fig(fig, output_root / "plots/trajectories/generic_action_usage", formats)
    if not metric_values.empty:
        metric_means = metric_values.groupby(["method_id", "metric"], as_index=False)["value"].mean()
        fig, axis = plt.subplots(figsize=(9, 4), dpi=180)
        for metric_name, frame in metric_means.groupby("metric", sort=True):
            axis.plot(frame["method_id"], frame["value"], marker="o", label=metric_name)
        axis.set_title("Environment metrics")
        axis.legend()
        axis.tick_params(axis="x", rotation=35)
        _save_fig(fig, output_root / "plots/core/generic_environment_metrics", formats)
    _write_summary_json(
        output_root,
        {
            "report_type": "generic",
            "run_dir": str(run_dir),
            "num_runs": int(len(run_summary)),
            "environments": sorted(run_summary["env_id"].dropna().unique().tolist()) if not run_summary.empty else [],
            "missing_runs": int(len(missing_artifacts_df)),
        },
    )
    return output_root


def generate_report(run_dir: Path, output_root: Path, formats: list[str]) -> Path:
    _ensure_dirs(output_root)
    artifacts, missing_artifacts_df = _load_artifacts(run_dir)
    if not artifacts:
        raise RuntimeError(f"No complete runs found under {run_dir}")

    if any(_environment_spec_for_run(artifact).get("env_id") != "icu_sepsis" for artifact in artifacts):
        return _generate_generic_report(run_dir, output_root, formats, artifacts, missing_artifacts_df)

    reference_stats = _load_reference_stats_fast()
    if reference_stats is None:
        _warn("Could not load reference stats from local benchmark README")

    run_manifest_df = _flatten_manifest(artifacts)
    sofa_edges, sofa_labels = _initial_sofa_bins(
        pd.concat([artifact.episodes["initial_sofa"] for artifact in artifacts], ignore_index=True)
    )

    run_level_rows: list[dict] = []
    train_long_frames: list[pd.DataFrame] = []
    eval_long_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    action_usage_frames: list[pd.DataFrame] = []
    heatmap_frames: list[pd.DataFrame] = []

    for artifact in artifacts:
        run_summary, train_long, eval_long, action_usage, heatmap_counts = _build_run_level_summary(artifact)
        run_level_rows.append(run_summary)
        train_long_frames.append(train_long)
        eval_long_frames.append(eval_long)
        action_usage_frames.append(action_usage)
        heatmap_frames.append(heatmap_counts)
        episode_frames.append(_build_episode_enriched(artifact, sofa_edges, sofa_labels))

    run_level_df = pd.DataFrame(run_level_rows)
    train_history_long_df = pd.concat(train_long_frames, ignore_index=True)
    eval_history_long_df = pd.concat(eval_long_frames, ignore_index=True)
    episode_enriched_df = pd.concat(episode_frames, ignore_index=True)
    action_usage_df = pd.concat(action_usage_frames, ignore_index=True)
    action_heatmap_df = pd.concat(heatmap_frames, ignore_index=True)

    group_level_df = _aggregate_group_metrics(run_level_df, episode_enriched_df)
    trajectory_quality_df = _trajectory_quality_summary(group_level_df)
    survival_by_bin_df = _survival_by_initial_sofa_bin(episode_enriched_df)
    action_usage_summary_df = _aggregate_action_usage(action_usage_df)
    action_heatmap_summary_df = _build_action_heatmap_table(action_heatmap_df)

    base_tables = _make_run_level_tables(run_manifest_df, run_level_df)
    base_tables["group_level_summary"] = group_level_df
    base_tables["final_eval_mean_return"] = (
        group_level_df[["dataset_tag", "method_id", "display_name", "final_eval_mean_return", "final_eval_mean_return_error", "final_eval_mean_return_ci_low", "final_eval_mean_return_ci_high"]]
        .sort_values(["dataset_tag", "final_eval_mean_return"], ascending=[True, False])
        .reset_index(drop=True)
    )
    base_tables["final_eval_full_metrics"] = group_level_df[
        [
            "dataset_tag",
            "method_id",
            "display_name",
            "final_eval_mean_return",
            "final_eval_mean_survival",
            "final_eval_mean_length",
            "final_eval_mean_action_stability_l1",
            "final_eval_mean_abrupt_jump_rate",
            "final_eval_mean_sofa",
            "final_eval_mean_final_sofa_delta",
            "mean_action_entropy",
            "fluid_mean",
            "vaso_mean",
        ]
    ].copy()
    base_tables["best_checkpoint_summary"] = group_level_df[
        [
            "dataset_tag",
            "method_id",
            "display_name",
            "best_eval_return",
            "best_eval_step",
            "best_fraction_of_budget",
            "best_effective_epochs",
            "final_eval_gap",
        ]
    ].copy()
    base_tables["validation_checkpoint_summary"] = group_level_df[
        [
            "dataset_tag",
            "method_id",
            "display_name",
            "best_validation_loss",
            "best_validation_step",
            "best_validation_sampled_transitions",
            "best_validation_fraction_of_budget",
            "best_validation_effective_epochs",
        ]
    ].copy()
    base_tables["early_stopping_summary"] = group_level_df[
        [
            "dataset_tag",
            "method_id",
            "display_name",
            "early_stopped",
            "early_stop_reason",
            "early_stop_step",
            "early_stop_sampled_transitions",
            "early_stop_fraction_of_budget",
            "num_validation_checks",
            "num_checks_without_improvement",
        ]
    ].copy()
    base_tables["eval_history_long"] = eval_history_long_df
    base_tables["train_history_long"] = train_history_long_df
    base_tables["episode_summary_enriched"] = episode_enriched_df
    base_tables["trajectory_quality_summary"] = trajectory_quality_df
    base_tables["time_to_best"] = group_level_df[
        [
            "dataset_tag",
            "method_id",
            "display_name",
            "best_eval_step",
            "best_sampled_transitions",
            "best_fraction_of_budget",
            "best_effective_epochs",
        ]
    ].copy()
    base_tables["time_to_best_validation"] = group_level_df[
        [
            "dataset_tag",
            "method_id",
            "display_name",
            "best_validation_step",
            "best_validation_sampled_transitions",
            "best_validation_fraction_of_budget",
            "best_validation_effective_epochs",
        ]
    ].copy()
    base_tables["survival_by_initial_sofa_bin"] = survival_by_bin_df
    base_tables["action_usage_summary"] = action_usage_summary_df
    base_tables["action_heatmap_summary"] = action_heatmap_summary_df
    base_tables["wm_component_snapshot"] = run_level_df[run_level_df["method_id"].isin(WM_METHODS)].copy()
    base_tables["trajectory_quality"] = trajectory_quality_df
    base_tables["report_leaderboards"] = _build_report_leaderboards(group_level_df)
    base_tables["missing_artifacts"] = missing_artifacts_df
    _write_tables(output_root, base_tables)

    _plot_ranked_bars(group_level_df, output_root, formats, reference_stats)
    _plot_eval_curves(eval_history_long_df, run_manifest_df, output_root, formats)
    _plot_core_summaries(group_level_df, output_root, formats)
    _plot_primary_losses(run_level_df, train_history_long_df, output_root, formats)
    _plot_wm_diagnostics(run_level_df, train_history_long_df, output_root, formats)
    _plot_episode_distributions(episode_enriched_df, output_root, formats)
    _plot_trajectory_summaries(group_level_df, survival_by_bin_df, action_heatmap_summary_df, output_root, formats)

    summary_payload = _summary_json(
        run_dir=run_dir,
        group_level_df=group_level_df,
        run_level_df=run_level_df,
        survival_by_bin_df=survival_by_bin_df,
        references=reference_stats,
        missing_artifacts_df=missing_artifacts_df,
    )
    _write_summary_json(output_root, summary_payload)
    return output_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report artifacts for a run manifest.")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run_name> containing manifest.json")
    parser.add_argument("--output-root", default="reports", help="Root folder under which the report will be saved")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Figure formats to emit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    report_root = Path(args.output_root).resolve() / run_dir.name
    generated = generate_report(run_dir, report_root, [fmt.lower() for fmt in args.formats])
    print(f"Saved report under {generated}")


if __name__ == "__main__":
    main()
