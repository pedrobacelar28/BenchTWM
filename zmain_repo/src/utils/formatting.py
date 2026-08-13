"""Output path helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.bootstrap import PROJECT_ROOT
from src.utils.methods import method_id


IGNORED_KEYS = {"use_tensorboard", "track", "method_id", "show_progress", "offline_console_style"}


def _normalize_value(value):
    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _normalize_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def normalize_experiment(experiment: dict) -> dict:
    return {
        key: _normalize_value(value)
        for key, value in sorted(experiment.items())
        if key not in IGNORED_KEYS and not (key == "train_mode" and str(value).lower() == "online")
    }


def experiment_signature(experiment: dict) -> str:
    normalized = normalize_experiment(experiment)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_file_name(experiment: dict, sub_folder: str = "adhoc") -> tuple[str, str]:
    algo = method_id(normalize_experiment(experiment))
    folder = PROJECT_ROOT / "artifacts" / sub_folder / algo
    return f"{folder}/", experiment_signature(experiment)


def output_dir_for_experiment(experiment: dict, sub_folder: str = "adhoc") -> Path:
    folder, signature = create_file_name(experiment, sub_folder=sub_folder)
    return Path(folder) / signature


def create_folder(folder: str | Path) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
