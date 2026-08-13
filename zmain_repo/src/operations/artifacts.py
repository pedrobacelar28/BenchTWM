"""Artifact layout, run manifests, and run discovery for the clean CLI."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.bootstrap import PROJECT_ROOT
from src.utils.formatting import experiment_signature
from src.utils.io import ensure_dir, load_json, save_json
from src.utils.methods import benchmark_track, task_id


def artifact_root(value: str | Path | None = None) -> Path:
    return Path(value or PROJECT_ROOT / "artifacts").resolve()


def _path_component(value: str) -> str:
    return str(value).strip().replace("_", "-").replace(" ", "-")


def run_directory(
    params: dict[str, Any],
    *,
    dataset_id: str,
    method_id: str,
    root: str | Path | None = None,
) -> Path:
    signature = experiment_signature(params)[:12]
    seed = int(params.get("seed", 0))
    dataset_parts = Path(dataset_id).parts
    environment_component = _path_component(str(params["env_id"]))
    if dataset_parts and _path_component(dataset_parts[0]) in {environment_component, environment_component.replace("_", "-")}:
        dataset_parts = dataset_parts[1:]
    return (
        artifact_root(root)
        / "runs"
        / environment_component
        / Path(*dataset_parts)
        / _path_component(str(params.get("train_mode", "online")))
        / _path_component(method_id)
        / f"seed-{seed}--{signature}"
    )


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance(device: str | None = None) -> dict[str, Any]:
    cuda = bool(torch.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda if cuda else None,
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if cuda else [],
        "requested_device": device or "auto",
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }


def initial_manifest(
    params: dict[str, Any],
    *,
    dataset_id: str,
    method_id: str,
    run_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    signature = experiment_signature(params)
    return {
        "schema_version": 1,
        "run_id": f"{params['env_id']}:{dataset_id}:{method_id}:seed-{int(params.get('seed', 0))}:{signature[:12]}",
        "status": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(run_dir),
        "dataset_id": dataset_id,
        "task_id": task_id(params, dataset_id),
        "method_id": method_id,
        "track": benchmark_track(params),
        "recipe_id": params.get("recipe_id"),
        "config_signature": signature,
        "environment": params.get("env_id"),
        "train_mode": params.get("train_mode", "online"),
        "seed": int(params.get("seed", 0)),
        "provenance": _provenance(device),
    }


def write_manifest(run_dir: str | Path, manifest: dict[str, Any], *, status: str | None = None, error: str | None = None) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    current = dict(manifest)
    if status is not None:
        current["status"] = status
    if error is not None:
        current["error"] = error
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    if status in {"completed", "failed"}:
        current["finished_at"] = current["updated_at"]
    save_json(current, run_dir / "run_manifest.json")
    return current


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    return load_json(Path(run_dir) / "run_manifest.json")


def is_complete(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    manifest_path = path / "run_manifest.json"
    if not manifest_path.exists() or not (path / "metrics.pkl").exists():
        return False
    try:
        return load_manifest(path).get("status") == "completed"
    except Exception:
        return False


def discover_runs(root: str | Path | None = None) -> list[tuple[Path, dict[str, Any]]]:
    runs_root = artifact_root(root) / "runs"
    if not runs_root.exists():
        return []
    discovered = []
    for path in runs_root.rglob("run_manifest.json"):
        try:
            discovered.append((path.parent, load_json(path)))
        except Exception:
            continue
    return sorted(discovered, key=lambda item: item[1].get("updated_at", ""), reverse=True)


def resolve_run(reference: str, root: str | Path | None = None) -> Path:
    candidate = Path(reference)
    if candidate.exists() and (candidate / "run_manifest.json").exists():
        return candidate.resolve()
    matches = [path for path, manifest in discover_runs(root) if manifest.get("run_id") == reference or path.name == reference]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No run matches {reference!r} under {artifact_root(root) / 'runs'}")
    raise ValueError(f"Run reference {reference!r} is ambiguous; use its full path or run_id")
