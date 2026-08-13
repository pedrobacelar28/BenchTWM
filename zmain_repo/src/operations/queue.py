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

from src.operations.artifacts import artifact_root, is_complete, write_manifest
from src.utils.io import ensure_dir, load_json, save_json


@dataclass
class QueueJob:
    name: str
    params: dict[str, Any]
    output_dir: str
    manifest: dict[str, Any]
    device: str = "auto"
    force: bool = False


def _queue_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_queue_manifest(queue_dir: Path, payload: dict[str, Any]) -> None:
    save_json(payload, queue_dir / "queue_manifest.json")


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
) -> tuple[Path, dict[str, Any]]:
    """Execute jobs in isolated workers and persist status after every transition."""

    if max_workers < 1:
        raise ValueError("--jobs must be at least 1")
    gpu_slots = _parse_gpus(gpus)
    if gpu_slots:
        max_workers = min(max_workers, len(gpu_slots))
    queue_dir = ensure_dir(artifact_root(root) / "queues" / _queue_id())
    jobs_dir = ensure_dir(queue_dir / "jobs")
    logs_dir = ensure_dir(queue_dir / "logs")
    records: list[dict[str, Any]] = []
    pending: list[tuple[int, QueueJob, Path]] = []
    for index, job in enumerate(jobs):
        output_dir = Path(job.output_dir)
        status = "pending"
        if resume and not job.force and is_complete(output_dir):
            status = "skipped"
        record = {"index": index, "name": job.name, "output_dir": str(output_dir), "status": status, "log_path": str(logs_dir / f"{index:03d}.log")}
        records.append(record)
        job_path = jobs_dir / f"{index:03d}.json"
        payload = asdict(job)
        payload["params"] = {**payload["params"], "show_progress": False}
        save_json(payload, job_path)
        if status == "pending":
            pending.append((index, job, job_path))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "queue_id": queue_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_workers": max_workers,
        "gpus": gpu_slots,
        "jobs": records,
    }
    _write_queue_manifest(queue_dir, manifest)

    active: dict[int, tuple[subprocess.Popen, Any, int]] = {}
    failures = 0
    next_pending = 0
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
            log_handle = open(records[index]["log_path"], "w", encoding="utf-8")
            write_manifest(job.output_dir, job.manifest, status="queued")
            command = [sys.executable, "-m", "src.operations.worker", str(job_path)]
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            active[index] = (process, log_handle, int(time.monotonic()))
            records[index]["status"] = "running"
            records[index]["gpu"] = gpu
            records[index]["started_at"] = datetime.now(timezone.utc).isoformat()
            _write_queue_manifest(queue_dir, manifest)
        time.sleep(0.05)
        for index, (process, log_handle, _) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            records[index]["returncode"] = int(code)
            records[index]["finished_at"] = datetime.now(timezone.utc).isoformat()
            records[index]["status"] = "completed" if code == 0 else "failed"
            if code != 0:
                failures += 1
            del active[index]
            _write_queue_manifest(queue_dir, manifest)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["failures"] = failures
    _write_queue_manifest(queue_dir, manifest)
    return queue_dir, manifest


def read_queue(path: str | Path) -> dict[str, Any]:
    return load_json(Path(path) / "queue_manifest.json")
