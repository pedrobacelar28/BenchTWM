"""Aggregate runs across seeds."""

from __future__ import annotations

import os
import sys

sys.path.append(os.getcwd())

from analysis.utils import aggregate_metric_dict, load_run_metrics, save_processed
from src.utils.json_handling import get_param_iterable, get_param_iterable_runs, get_sorted_dict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/process_data.py <json files>")
        raise SystemExit(1)

    for json_file in sys.argv[1:]:
        config = get_sorted_dict(json_file)
        for group in get_param_iterable_runs(config):
            run_metrics = []
            for concrete in get_param_iterable(group):
                try:
                    run_metrics.append(load_run_metrics(concrete))
                except FileNotFoundError:
                    print(f"Missing run for {concrete}")
            if not run_metrics:
                continue
            processed = aggregate_metric_dict(run_metrics)
            path = save_processed(processed, group)
            print(f"Saved {path}")
