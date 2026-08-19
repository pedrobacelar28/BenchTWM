"""Local subprocess queue for reproducible benchmark suites."""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.operations.artifacts import artifact_root, is_complete, write_manifest
from src.utils.io import ensure_dir, load_json, save_json
from src.utils.queue_progress import read_progress, update_progress

try:  # Rich is optional for library callers and diagnostics.
    from rich.console import Console as RichConsole
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
except ImportError:  # pragma: no cover - requirements install Rich
    RichConsole = None
    Progress = None


@dataclass
class QueueJob:
    name: str
    params: dict[str, Any]
    output_dir: str
    manifest: dict[str, Any]
    device: str = "auto"
    force: bool = False
    progress_path: str | None = None


def _queue_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_queue_manifest(queue_dir: Path, payload: dict[str, Any]) -> None:
    save_json(payload, queue_dir / "queue_manifest.json")


def _write_suite_results(queue_dir: Path, manifest: dict[str, Any]) -> Path:
    """Persist a central, queue-level view of all jobs and their results."""

    def optional_json(path: Path) -> dict[str, Any]:
        try:
            value = load_json(path)
        except (FileNotFoundError, OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    jobs: list[dict[str, Any]] = []
    for record in manifest.get("jobs", []):
        output_dir = Path(record["output_dir"])
        run_manifest = optional_json(output_dir / "run_manifest.json")
        config = optional_json(output_dir / "config.json")
        metrics = optional_json(output_dir / "metrics_summary.json")
        direct_evaluation = optional_json(output_dir / "eval_summary.json") or metrics.get("eval_summary", {})
        ope = optional_json(output_dir / "ope_summary.json")
        jobs.append(
            {
                "name": record.get("name"),
                "status": record.get("status"),
                "returncode": record.get("returncode"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "gpu": record.get("gpu"),
                "run_id": record.get("run_id") or run_manifest.get("run_id"),
                "dataset_id": record.get("dataset_id") or run_manifest.get("dataset_id") or config.get("dataset_id"),
                "task_id": record.get("task_id") or run_manifest.get("task_id") or config.get("task_id"),
                "method_id": record.get("method_id") or run_manifest.get("method_id") or config.get("method_id"),
                "track": record.get("track") or run_manifest.get("track") or config.get("track"),
                "environment": record.get("environment") or run_manifest.get("environment") or config.get("env_id"),
                "mode": record.get("mode") or run_manifest.get("train_mode") or config.get("train_mode"),
                "seed": record.get("seed", run_manifest.get("seed", config.get("seed"))),
                "observation_backend": record.get("observation_backend") or config.get("obs_backend"),
                "action_type": record.get("action_type") or config.get("action_type"),
                "output_dir": str(output_dir),
                "log_path": record.get("log_path"),
                "progress_path": record.get("progress_path"),
                "progress": read_progress(record.get("progress_path")),
                "run_manifest": run_manifest,
                "config": config,
                "metrics_summary": metrics,
                "optimization": metrics.get("train_summary", {}),
                "direct_evaluation": direct_evaluation,
                "timing": metrics.get("timing_summary", {}),
                "ope": ope,
            }
        )

    def numeric(values: list[Any]) -> list[float]:
        return [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]

    def aggregate(values: list[Any]) -> tuple[float | None, float | None]:
        numbers = numeric(values)
        if not numbers:
            return None, None
        return float(statistics.fmean(numbers)), float(statistics.stdev(numbers)) if len(numbers) > 1 else 0.0

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for job in jobs:
        evaluation = job.get("direct_evaluation") or {}
        if job.get("status") not in {"completed", "skipped"} or not isinstance(evaluation.get("mean_return"), (int, float)):
            continue
        key = (str(job.get("task_id") or ""), str(job.get("method_id") or ""), str(job.get("track") or ""))
        grouped.setdefault(key, []).append(job)

    method_summary: list[dict[str, Any]] = []
    for (task_id, method_id, track), group in sorted(grouped.items()):
        returns = aggregate([(job.get("direct_evaluation") or {}).get("mean_return") for job in group])
        successes = aggregate([(job.get("direct_evaluation") or {}).get("mean_success") for job in group])
        lengths = aggregate([(job.get("direct_evaluation") or {}).get("mean_length") for job in group])
        latencies = aggregate([(job.get("direct_evaluation") or {}).get("mean_action_latency_ms") for job in group])
        optimization_times = aggregate([(job.get("timing") or {}).get("optimization_seconds") for job in group])
        evaluation_times = aggregate([(job.get("timing") or {}).get("evaluation_seconds") for job in group])
        planner_calls = aggregate([(job.get("direct_evaluation") or {}).get("planner_calls") for job in group])
        planner_evaluations = aggregate([(job.get("direct_evaluation") or {}).get("planner_model_evaluations") for job in group])
        method_summary.append(
            {
                "task_id": task_id,
                "method_id": method_id,
                "track": track,
                "seeds": sorted({int(job["seed"]) for job in group if isinstance(job.get("seed"), int)}),
                "num_runs": len(group),
                "mean_return_mean": returns[0],
                "mean_return_std": returns[1],
                "mean_success_mean": successes[0],
                "mean_success_std": successes[1],
                "mean_length_mean": lengths[0],
                "mean_length_std": lengths[1],
                "mean_action_latency_ms_mean": latencies[0],
                "mean_action_latency_ms_std": latencies[1],
                "optimization_seconds_mean": optimization_times[0],
                "optimization_seconds_std": optimization_times[1],
                "evaluation_seconds_mean": evaluation_times[0],
                "evaluation_seconds_std": evaluation_times[1],
                "planner_calls_mean": planner_calls[0],
                "planner_calls_std": planner_calls[1],
                "planner_model_evaluations_mean": planner_evaluations[0],
                "planner_model_evaluations_std": planner_evaluations[1],
            }
        )

    best_direct: dict[str, dict[str, Any]] = {}
    for item in method_summary:
        if item["track"] != "direct_rl" or item["mean_return_mean"] is None:
            continue
        current = best_direct.get(item["task_id"])
        if current is None or item["mean_return_mean"] > current["mean_return_mean"]:
            best_direct[item["task_id"]] = {
                "method_id": item["method_id"],
                "track": item["track"],
                "mean_return_mean": item["mean_return_mean"],
                "mean_success_mean": item["mean_success_mean"],
                "num_runs": item["num_runs"],
            }

    planning_gain: list[dict[str, Any]] = []
    for item in method_summary:
        baseline = best_direct.get(item["task_id"])
        if item["track"] != "provisional_planning" or baseline is None:
            continue
        planning_gain.append(
            {
                "task_id": item["task_id"],
                "method_id": item["method_id"],
                "track": item["track"],
                "best_direct_method_id": baseline["method_id"],
                "planning_gain_return": item["mean_return_mean"] - baseline["mean_return_mean"],
                "planning_gain_success": item["mean_success_mean"] - baseline["mean_success_mean"],
            }
        )

    statuses = [str(job.get("status")) for job in jobs]
    if manifest.get("finished_at"):
        suite_status = "completed" if int(manifest.get("failures", 0)) == 0 else "failed"
    elif any(status in {"running", "pending"} for status in statuses):
        suite_status = "running"
    else:
        suite_status = "planned"
    payload = {
        "schema_version": 1,
        "suite": {
            "queue_id": manifest.get("queue_id"),
            "status": suite_status,
            "created_at": manifest.get("created_at"),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "dataset_ids": sorted({str(job.get("dataset_id")) for job in jobs if job.get("dataset_id")}),
            "task_ids": sorted({str(job.get("task_id")) for job in jobs if job.get("task_id")}),
            "modes": sorted({str(job.get("mode")) for job in jobs if job.get("mode")}),
            "max_workers": manifest.get("max_workers"),
            "gpus": manifest.get("gpus", []),
        },
        "counts": {
            "total": len(jobs),
            "pending": statuses.count("pending"),
            "running": statuses.count("running"),
            "completed": statuses.count("completed"),
            "skipped": statuses.count("skipped"),
            "failed": statuses.count("failed"),
        },
        "jobs": jobs,
        "method_summary": method_summary,
        "best_direct_rl": best_direct,
        "planning_gain": planning_gain,
        "artifacts": {
            "queue_manifest": str(queue_dir / "queue_manifest.json"),
            "suite_results": str(queue_dir / "suite_results.json"),
        },
    }
    destination = queue_dir / "suite_results.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    save_json(payload, temporary)
    temporary.replace(destination)
    return destination


class _FollowDisplay:
    """Render queue progress without mixing worker stdout streams."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._rich_progress = None
        self._tasks: dict[int, int] = {}
        self._fallback_seen: dict[int, tuple[Any, ...]] = {}

    def start(self, records: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        if Progress is not None and RichConsole is not None and sys.stderr.isatty():
            self._rich_progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=RichConsole(stderr=True),
                transient=False,
            )
            self._rich_progress.start()
            for index, record in enumerate(records):
                self._tasks[index] = self._rich_progress.add_task(record["name"], total=None)

    def update(self, records: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        for index, record in enumerate(records):
            snapshot = read_progress(record.get("progress_path"))
            status = str(record.get("status", "pending"))
            phase = str(snapshot.get("phase", "queued"))
            step = snapshot.get("step")
            total = snapshot.get("total_steps")
            if self._rich_progress is not None:
                task_id = self._tasks[index]
                description = f"{record['name']} [{status}/{phase}]"
                kwargs: dict[str, Any] = {"description": description}
                if isinstance(total, (int, float)) and total > 0:
                    kwargs["total"] = int(total)
                    kwargs["completed"] = min(int(step or 0), int(total))
                elif status in {"completed", "skipped", "failed"}:
                    kwargs["total"] = 1
                    kwargs["completed"] = 1
                self._rich_progress.update(task_id, **kwargs)
                continue
            signature = (status, phase, step, total)
            if self._fallback_seen.get(index) == signature:
                continue
            self._fallback_seen[index] = signature
            progress_text = ""
            if isinstance(total, (int, float)) and total > 0:
                progress_text = f" step={int(step or 0)}/{int(total)}"
            elif isinstance(step, (int, float)):
                progress_text = f" step={int(step)}"
            print(f"[suite] {record['name']}: {status} phase={phase}{progress_text}", file=sys.stderr, flush=True)

    def stop(self) -> None:
        if self._rich_progress is not None:
            self._rich_progress.stop()
            self._rich_progress = None


def _parse_gpus(raw: str | None) -> list[str]:
    if not raw:
        return []
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--gpus must be a comma-separated list such as 0,1")
    if any(not value.isdigit() for value in values):
        raise ValueError("--gpus accepts numeric CUDA indices only")
    return values


def execute_queue(
    jobs: list[QueueJob],
    *,
    root: str | Path | None = None,
    max_workers: int = 1,
    gpus: str | None = None,
    resume: bool = True,
    follow: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Execute jobs in isolated workers and persist status after every transition.

    ``follow`` renders a parent-owned progress display. Worker stdout remains
    redirected to per-job logs, so multiple jobs never corrupt each other's
    terminal output.
    """

    if max_workers < 1:
        raise ValueError("--jobs must be at least 1")
    gpu_slots = _parse_gpus(gpus)
    if gpu_slots:
        max_workers = min(max_workers, len(gpu_slots))
    queue_dir = ensure_dir(artifact_root(root) / "queues" / _queue_id())
    jobs_dir = ensure_dir(queue_dir / "jobs")
    logs_dir = ensure_dir(queue_dir / "logs")
    progress_dir = ensure_dir(queue_dir / "progress")
    records: list[dict[str, Any]] = []
    pending: list[tuple[int, QueueJob, Path]] = []
    for index, job in enumerate(jobs):
        output_dir = Path(job.output_dir)
        status = "pending"
        if resume and not job.force and is_complete(output_dir):
            status = "skipped"
        progress_path = progress_dir / f"{index:03d}.json"
        record = {
            "index": index,
            "name": job.name,
            "output_dir": str(output_dir),
            "status": status,
            "log_path": str(logs_dir / f"{index:03d}.log"),
            "progress_path": str(progress_path),
            "run_id": job.manifest.get("run_id"),
            "dataset_id": job.manifest.get("dataset_id") or job.params.get("dataset_id"),
            "task_id": job.manifest.get("task_id") or job.params.get("task_id"),
            "method_id": job.manifest.get("method_id") or job.params.get("method_id"),
            "track": job.manifest.get("track") or job.params.get("track"),
            "environment": job.manifest.get("environment") or job.params.get("env_id"),
            "mode": job.manifest.get("train_mode") or job.params.get("train_mode"),
            "seed": job.manifest.get("seed", job.params.get("seed")),
            "observation_backend": job.params.get("obs_backend"),
            "action_type": job.params.get("action_type"),
        }
        records.append(record)
        job_path = jobs_dir / f"{index:03d}.json"
        payload = asdict(job)
        payload["params"] = {**payload["params"], "show_progress": False}
        payload["progress_path"] = str(progress_path)
        save_json(payload, job_path)
        update_progress(
            progress_path,
            status="skipped" if status == "skipped" else "pending",
            phase="resume" if status == "skipped" else "queued",
            method_id=record["method_id"],
            total_steps=job.params.get("offline_train_steps"),
            step=job.params.get("offline_train_steps") if status == "skipped" else 0,
        )
        if status == "pending":
            pending.append((index, job, job_path))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "queue_id": queue_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "max_workers": max_workers,
        "gpus": gpu_slots,
        "suite_results_path": str(queue_dir / "suite_results.json"),
        "jobs": records,
    }
    _write_queue_manifest(queue_dir, manifest)
    _write_suite_results(queue_dir, manifest)

    active: dict[int, tuple[subprocess.Popen, Any, int]] = {}
    failures = 0
    next_pending = 0
    display = _FollowDisplay(follow)
    display.start(records)
    try:
        while next_pending < len(pending) or active:
            while next_pending < len(pending) and len(active) < max_workers:
                index, job, job_path = pending[next_pending]
                next_pending += 1
                gpu = gpu_slots[index % len(gpu_slots)] if gpu_slots else None
                env = os.environ.copy()
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                    env["BENCH_DEVICE"] = "cuda"
                else:
                    env["BENCH_DEVICE"] = job.device
                env["BENCH_PROGRESS_PATH"] = records[index]["progress_path"]
                log_handle = open(records[index]["log_path"], "w", encoding="utf-8")
                write_manifest(job.output_dir, job.manifest, status="queued")
                update_progress(
                    records[index]["progress_path"],
                    status="running",
                    phase="starting",
                    method_id=records[index]["method_id"],
                    total_steps=job.params.get("offline_train_steps"),
                    step=0,
                )
                command = [sys.executable, "-m", "src.operations.worker", str(job_path)]
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], stdout=log_handle, stderr=subprocess.STDOUT, env=env)
                active[index] = (process, log_handle, int(time.monotonic()))
                records[index]["status"] = "running"
                records[index]["gpu"] = gpu
                records[index]["started_at"] = datetime.now(timezone.utc).isoformat()
                if manifest["started_at"] is None:
                    manifest["started_at"] = records[index]["started_at"]
                _write_queue_manifest(queue_dir, manifest)
                _write_suite_results(queue_dir, manifest)
            display.update(records)
            time.sleep(0.05)
            for index, (process, log_handle, _) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log_handle.close()
                records[index]["returncode"] = int(code)
                records[index]["finished_at"] = datetime.now(timezone.utc).isoformat()
                records[index]["status"] = "completed" if code == 0 else "failed"
                previous_progress = read_progress(records[index]["progress_path"])
                preserved_progress = {
                    key: previous_progress[key]
                    for key in ("step", "total_steps", "sampled_transitions", "mean_return", "mean_success", "validation_loss", "early_stop_reason")
                    if key in previous_progress
                }
                update_progress(
                    records[index]["progress_path"],
                    status=records[index]["status"],
                    phase="complete" if code == 0 else "error",
                    method_id=records[index]["method_id"],
                    **preserved_progress,
                )
                if code != 0:
                    failures += 1
                del active[index]
                _write_queue_manifest(queue_dir, manifest)
                _write_suite_results(queue_dir, manifest)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["failures"] = failures
        _write_queue_manifest(queue_dir, manifest)
        _write_suite_results(queue_dir, manifest)
        display.update(records)
    finally:
        display.stop()
    return queue_dir, manifest


def read_queue(path: str | Path) -> dict[str, Any]:
    return load_json(Path(path) / "queue_manifest.json")
