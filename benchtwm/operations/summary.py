"""Aggregate completed runs: per-method statistics and the planning gain.

The planning gain of a model-based method on a dataset is its mean return
(and success) minus that of the best model-free method on the same dataset.
Datasets are never averaged together: their returns have different meanings.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchtwm.algorithms.base import MODEL_BASED, MODEL_FREE
from benchtwm.operations.artifacts import artifact_root, discover_runs, manifest_family
from benchtwm.utils.io import ensure_dir, load_json, save_json

TIMING_FIELDS = ("optimization_seconds", "evaluation_seconds", "ope_seconds")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read(path: Path) -> dict[str, Any]:
    try:
        return load_json(path)
    except (OSError, ValueError):
        return {}


def run_row(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """One flat row for a completed run, or ``None``."""

    if manifest.get("status") != "completed":
        return None
    metrics = _read(run_dir / "metrics_summary.json")
    evaluation = metrics.get("eval_summary") or _read(run_dir / "eval_summary.json")
    if not evaluation:
        return None
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "dataset_id": manifest.get("dataset_id"),
        "env_id": manifest.get("env_id") or manifest.get("environment"),
        "method_id": manifest.get("method_id"),
        "family": manifest_family(manifest),
        "seed": int(manifest.get("seed", 0)),
    }
    for key, value in evaluation.items():
        if (number := _number(value)) is not None:
            row[key] = number
    timing = metrics.get("timing_summary") or {}
    for key in TIMING_FIELDS:
        row[key] = _number(timing.get(key))
    for key, value in (metrics.get("ope_summary") or {}).items():
        if key.endswith("_estimate"):
            row[f"ope_{key}"] = _number(value)
    return row


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset_id"], row["method_id"], row["family"])].append(row)
    numeric = sorted({key for row in rows for key, value in row.items() if isinstance(value, float) and key != "seed"})
    result = []
    for (dataset_id, method_id, family), group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        item: dict[str, Any] = {"dataset_id": dataset_id, "method_id": method_id, "family": family, "num_seeds": len(group), "seeds": sorted(row["seed"] for row in group)}
        for field in numeric:
            item[f"{field}_mean"], item[f"{field}_std"] = _mean_std([row[field] for row in group if row.get(field) is not None])
        result.append(item)
    return result


def planning_gain(aggregated: list[dict[str, Any]]) -> tuple[dict[str, dict], list[dict[str, Any]]]:
    best: dict[str, dict] = {}
    for item in aggregated:
        if item["family"] == MODEL_FREE and item.get("mean_return_mean") is not None:
            if item["dataset_id"] not in best or item["mean_return_mean"] > best[item["dataset_id"]]["mean_return_mean"]:
                best[item["dataset_id"]] = item
    gains = []
    for item in aggregated:
        baseline = best.get(item["dataset_id"])
        if item["family"] != MODEL_BASED or baseline is None or item.get("mean_return_mean") is None:
            continue
        gains.append(
            {
                "dataset_id": item["dataset_id"],
                "method_id": item["method_id"],
                "best_model_free_method_id": baseline["method_id"],
                "planning_gain_return": item["mean_return_mean"] - baseline["mean_return_mean"],
                "planning_gain_success": (item.get("mean_success_mean") or 0.0) - (baseline.get("mean_success_mean") or 0.0),
                **{key: item.get(key) for key in ("optimization_seconds_mean", "evaluation_seconds_mean", "mean_action_latency_ms_mean", "planner_model_evaluations_mean")},
            }
        )
    return best, gains


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated = aggregate(rows)
    best, gains = planning_gain(aggregated)
    return {"rows": rows, "aggregate": aggregated, "best_model_free": {key: value["method_id"] for key, value in best.items()}, "planning_gain": gains}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["dataset_id", "method_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(root: str | Path | None = None, *, output_dir: str | Path | None = None, dataset_ids: list[str] | None = None) -> dict[str, Any]:
    """Summarize every completed run under ``root`` and write CSV/JSON/Markdown reports."""

    rows = [row for run_dir, manifest in discover_runs(root) if (row := run_row(run_dir, manifest)) is not None]
    if dataset_ids:
        rows = [row for row in rows if row["dataset_id"] in set(dataset_ids)]
    rows.sort(key=lambda row: (str(row["dataset_id"]), str(row["family"]), str(row["method_id"]), row["seed"]))
    summary = summarize(rows)
    destination = ensure_dir(output_dir or artifact_root(root) / "reports" / "summary")
    _write_csv(destination / "runs.csv", summary["rows"])
    _write_csv(destination / "aggregate.csv", summary["aggregate"])
    _write_csv(destination / "planning_gain.csv", summary["planning_gain"])
    payload = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(), "output_dir": str(destination), **summary}
    save_json(payload, destination / "summary.json")
    lines = ["# BenchTWM Summary", "", f"Runs: {len(rows)} | dataset x method groups: {len(summary['aggregate'])}", "",
             "| Dataset | Method | Family | Seeds | Return (mean ± std) | Success |", "|---|---|---|---:|---:|---:|"]
    for item in summary["aggregate"]:
        lines.append(
            f"| `{item['dataset_id']}` | `{item['method_id']}` | {item['family']} | {item['num_seeds']} | "
            f"{item.get('mean_return_mean')} ± {item.get('mean_return_std')} | {item.get('mean_success_mean')} |"
        )
    lines += ["", "Planning gains against the best model-free method per dataset are in `planning_gain.csv`.",
              "OPE estimates are reported per run and never averaged into simulator returns."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
