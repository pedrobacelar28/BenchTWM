"""Run a single indexed config from a JSON sweep."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.run_experiment import run_experiment
from src.utils.json_handling import get_param_iterable, get_sorted_dict


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/run_config.py <json_file> <idx>")
        raise SystemExit(1)
    json_file = sys.argv[1]
    idx = int(sys.argv[2])
    config = get_sorted_dict(json_file)
    experiments = get_param_iterable(config)
    run_experiment(experiments[idx % len(experiments)], force_run=False)
