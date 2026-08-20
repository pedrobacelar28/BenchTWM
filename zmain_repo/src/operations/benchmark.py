"""Aggregation helpers for the BenchTWM v0 pilot.

The aggregator intentionally keeps a task's environment-specific metrics
separate.  It never averages ICU-Sepsis and EpiCare rows into one number.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.operations.artifacts import artifact_root, discover_runs
from src.utils.io import ensure_dir, load_json, save_json
from src.utils.methods import benchmark_track, task_id


PILOT_TASKS = (
    "epicare/smart/dev|obs=state_vector|action=discrete|mode=offline",
    "icu-sepsis/expert/dev|obs=state_vector|action=discrete|mode=offline",
)
PILOT_METHODS = {
    "direct_rl": {"bc", "dqn", "iql", "cql"},
    "provisional_planning": {
        "raw-model",
        "latent-model",
        "reconstructive-model",
        "raw-model/continuous-cem",
        "latent-model/continuous-cem",
        "reconstructive-model/continuous-cem",
        "dreamer",
    },
}

NUMERIC_FIELDS = (
    "mean_return",
    "mean_success",
    "mean_survival",
    "mean_length",
    "mean_action_stability_l1",
    "mean_abrupt_jump_rate",
    "mean_action_latency_ms",
    "planner_calls",
    "planner_model_evaluations",
    "optimization_seconds",
    "evaluation_seconds",
    "ope_seconds",
)


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "" or (isinstance(value, str) and value.lower() == "nan"):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _run_row(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    if manifest.get("status") != "completed":
        return None
    metrics_path = run_dir / "metrics_summary.json"
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    config = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    dataset_id = str(manifest.get("dataset_id", config.get("dataset_id", "")))
    resolved_task = str(manifest.get("task_id") or config.get("task_id") or task_id({**config, "dataset_id": dataset_id}, dataset_id))
    method = str(manifest.get("method_id") or metrics.get("method_id", ""))
    track = str(manifest.get("track") or metrics.get("track") or benchmark_track({**config, "method_id": method}))
    evaluation = metrics.get("eval_summary", {})
    training = metrics.get("train_summary", {})
    timing = metrics.get("timing_summary", {})
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "task_id": resolved_task,
        "dataset_id": dataset_id,
        "environment": manifest.get("environment", config.get("env_id")),
        "method_id": method,
        "track": track,
        "seed": int(manifest.get("seed", config.get("seed", 0))),
    }
    for key in NUMERIC_FIELDS:
        value = evaluation.get(key)
        if value is None:
            value = timing.get(key)
        if value is None and key in {"mean_return", "mean_success", "mean_survival", "mean_length"}:
            fallback = {
                "mean_return": "final_return",
                "mean_success": "final_success_rate",
                "mean_survival": "final_survival_rate",
                "mean_length": "mean_episode_length",
            }[key]
            value = training.get(fallback)
        row[key] = _float(value)
    # Keep all environment-specific evaluation fields without using them to
    # rank unrelated tasks.
    for key, value in evaluation.items():
        if key not in row:
            numeric = _float(value)
            if numeric is not None:
                row[key] = numeric
    return row


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["task_id"]), str(row["method_id"]), str(row["track"]))].append(row)
    aggregate: list[dict[str, Any]] = []
    for (task, method, track), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "task_id": task,
            "method_id": method,
            "track": track,
            "num_seeds": len(group),
            "seeds": sorted(int(row["seed"]) for row in group),
        }
        for field in NUMERIC_FIELDS:
            values = [row[field] for row in group if row.get(field) is not None]
            mean, std = _mean_std([float(value) for value in values])
            item[f"{field}_mean"] = mean
            item[f"{field}_std"] = std
        aggregate.append(item)
    return aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["task_id", "method_id", "track"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latest_baseline(root: Path, dataset_id: str) -> dict[str, Any] | None:
    directory = artifact_root(root) / "reports" / "baselines" / dataset_id
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("*/baseline_summary.json"), reverse=True)
    return load_json(candidates[0]) if candidates else None


def summarize_pilot(root: str | Path | None = None, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    root_path = artifact_root(root)
    rows = [row for run_dir, manifest in discover_runs(root_path) if (row := _run_row(run_dir, manifest)) is not None]
    rows = [row for row in rows if row["task_id"] in PILOT_TASKS and row["track"] in {"direct_rl", "provisional_planning"}]
    rows.sort(key=lambda row: (row["task_id"], row["track"], row["method_id"], int(row["seed"])))
    aggregate = _aggregate(rows)

    best_direct: dict[str, dict[str, Any]] = {}
    for item in aggregate:
        if item["track"] != "direct_rl" or item.get("mean_return_mean") is None:
            continue
        task = item["task_id"]
        if task not in best_direct or float(item["mean_return_mean"]) > float(best_direct[task]["mean_return_mean"]):
            best_direct[task] = item

    planning_gain: list[dict[str, Any]] = []
    for item in aggregate:
        if item["track"] != "provisional_planning":
            continue
        baseline = best_direct.get(item["task_id"])
        gain_return = None
        gain_success = None
        if baseline is not None:
            if item.get("mean_return_mean") is not None:
                gain_return = float(item["mean_return_mean"]) - float(baseline["mean_return_mean"])
            if item.get("mean_success_mean") is not None and baseline.get("mean_success_mean") is not None:
                gain_success = float(item["mean_success_mean"]) - float(baseline["mean_success_mean"])
        planning_gain.append(
            {
                "task_id": item["task_id"],
                "method_id": item["method_id"],
                "track": item["track"],
                "best_direct_method": None if baseline is None else baseline["method_id"],
                "best_direct_return": None if baseline is None else baseline.get("mean_return_mean"),
                "best_direct_success": None if baseline is None else baseline.get("mean_success_mean"),
                "planning_return": item.get("mean_return_mean"),
                "planning_success": item.get("mean_success_mean"),
                "planning_gain_return": gain_return,
                "planning_gain_success": gain_success,
                "evaluation_seconds_mean": item.get("evaluation_seconds_mean"),
                "optimization_seconds_mean": item.get("optimization_seconds_mean"),
                "mean_action_latency_ms_mean": item.get("mean_action_latency_ms_mean"),
                "planner_calls_mean": item.get("planner_calls_mean"),
                "planner_model_evaluations_mean": item.get("planner_model_evaluations_mean"),
            }
        )

    destination = ensure_dir(output_dir or (root_path / "reports" / "v0_pilot"))
    _write_csv(destination / "v0_pilot_runs.csv", rows)
    _write_csv(destination / "v0_pilot_aggregate.csv", aggregate)
    _write_csv(destination / "v0_pilot_planning_gain.csv", planning_gain)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": list(PILOT_TASKS),
        "direct_methods": sorted(PILOT_METHODS["direct_rl"]),
        "provisional_planning_methods": sorted(PILOT_METHODS["provisional_planning"]),
        "rows": rows,
        "aggregate": aggregate,
        "best_direct_baselines": best_direct,
        "planning_gain": planning_gain,
        "reference_baselines": {task: _latest_baseline(root_path, task.rsplit("|", 1)[0]) for task in PILOT_TASKS},
        "output_dir": str(destination),
    }
    save_json(payload, destination / "v0_pilot_summary.json")
    lines = [
        "# BenchTWM v0 Pilot Summary",
        "",
        f"Runs: {len(rows)} | task/method aggregates: {len(aggregate)}",
        "",
        "## Best direct baseline per task",
        "",
        "| Task | Method | Return | Success | Seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, item in sorted(best_direct.items()):
        lines.append(
            f"| `{task}` | `{item['method_id']}` | {item.get('mean_return_mean')} | "
            f"{item.get('mean_success_mean')} | {','.join(str(seed) for seed in item['seeds'])} |"
        )
    lines.extend(
        [
            "",
            "## Planning gain and cost",
            "",
            "See `v0_pilot_planning_gain.csv` for return/success gains against the best direct baseline on the same task and planner cost fields.",
            "",
            "OPE is stored per run and is not averaged into direct simulator metrics.",
        ]
    )
    (destination / "v0_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
