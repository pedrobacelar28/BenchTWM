"""Sweep helpers."""

from __future__ import annotations

from pathlib import Path

from src.utils.formatting import output_dir_for_experiment


def check_experiment_not_done(experiment: dict) -> bool:
    output_dir = output_dir_for_experiment(experiment)
    return not (output_dir / "metrics.pkl").exists()


def get_list_pending_experiments(experiments: list[dict]) -> list[int]:
    pending = []
    for idx, experiment in enumerate(experiments):
        print(f"Checking [{idx + 1}/{len(experiments)}]\r", end="")
        if check_experiment_not_done(experiment):
            pending.append(idx)
    print()
    return pending
