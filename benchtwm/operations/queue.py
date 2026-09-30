"""Local subprocess queue for reproducible benchmark suites."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchtwm.operations.artifacts import artifact_root, is_complete, write_manifest
from benchtwm.operations.summary import run_row, summarize
from benchtwm.utils.io import ensure_dir, load_json, save_json
from benchtwm.utils.queue_progress import read_progress, update_progress

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
    ready: bool = True
    progress_path: str | None = None


def _queue_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_queue_manifest(queue_dir: Path, payload: dict[str, Any]) -> None:
    save_json(payload, queue_dir / "queue_manifest.json")


def _write_suite_results(queue_dir: Path, manifest: dict[str, Any]) -> Path:
    """Persist a queue-level view of every job, its results, and the planning gain."""

    jobs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for record in manifest.get("jobs", []):
        output_dir = Path(record["output_dir"])
        run_manifest = _optional_json(output_dir / "run_manifest.json")
        row = run_row(output_dir, run_manifest) if run_manifest else None
        if row is not None and record.get("status") in {"completed", "skipped"}:
            rows.append(row)
        jobs.append(
            {
                **{key: record.get(key) for key in ("name", "status", "returncode", "started_at", "finished_at", "gpu", "log_path", "progress_path")},
                **{key: record.get(key) or run_manifest.get(key) for key in ("run_id", "dataset_id", "env_id", "method_id", "family", "seed")},
                "output_dir": str(output_dir),
                "progress": read_progress(record.get("progress_path")),
                "direct_evaluation": _optional_json(output_dir / "eval_summary.json"),
                "ope": (_optional_json(output_dir / "metrics_summary.json").get("ope_summary") or {}),
            }
        )
    statuses = [str(job.get("status")) for job in jobs]
    if manifest.get("finished_at"):
        suite_status = "completed" if int(manifest.get("failures", 0)) == 0 else "failed"
    elif any(status in {"running", "pending"} for status in statuses):
        suite_status = "running"
    else:
        suite_status = "planned"
    summary = summarize(rows)
    payload = {
        "schema_version": 2,
        "suite": {
            "queue_id": manifest.get("queue_id"),
            "status": suite_status,
            "created_at": manifest.get("created_at"),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "dataset_ids": sorted({str(job["dataset_id"]) for job in jobs if job.get("dataset_id")}),
            "max_workers": manifest.get("max_workers"),
            "gpus": manifest.get("gpus", []),
        },
        "counts": {"total": len(jobs), **{status: statuses.count(status) for status in ("pending", "running", "completed", "skipped", "failed")}},
        "jobs": jobs,
        "method_summary": summary["aggregate"],
        "best_model_free": summary["best_model_free"],
        "planning_gain": summary["planning_gain"],
    }
    destination = queue_dir / "suite_results.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    save_json(payload, temporary)
    temporary.replace(destination)
    return destination


def _optional_json(path: Path) -> dict[str, Any]:
    try:
        value = load_json(path)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


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
                bar = self._tasks[index]
                description = f"{record['name']} [{status}/{phase}]"
                kwargs: dict[str, Any] = {"description": description}
                if isinstance(total, (int, float)) and total > 0:
                    kwargs["total"] = int(total)
                    kwargs["completed"] = min(int(step or 0), int(total))
                elif status in {"completed", "skipped", "failed"}:
                    kwargs["total"] = 1
                    kwargs["completed"] = 1
                self._rich_progress.update(bar, **kwargs)
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
            **{key: job.manifest.get(key) for key in ("run_id", "dataset_id", "env_id", "method_id", "family", "seed")},
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
                command = [sys.executable, "-m", "benchtwm.operations.worker", str(job_path)]
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
