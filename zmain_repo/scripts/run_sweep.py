"""Run pending indexed configs from one or more JSON sweep files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.json_handling import get_param_iterable, get_sorted_dict
from src.utils.run_utils import get_list_pending_experiments


def _pending_indices(json_file: str, *, overwrite: bool, start: int, end: int) -> list[int]:
    experiments = get_param_iterable(get_sorted_dict(json_file))
    pending = list(range(len(experiments))) if overwrite else get_list_pending_experiments(experiments)
    return [idx for idx in pending if idx >= start and (end < 0 or idx <= end)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--python-file", default="src/run_config.py", help="Config runner script.")
    parser.add_argument("-j", "--json", nargs="+", required=True, help="JSON sweep file(s).")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Run configs even when outputs already exist.")
    parser.add_argument("-s", "--start", type=int, default=0, help="First sweep index to run.")
    parser.add_argument("-e", "--end", type=int, default=-1, help="Last sweep index to run. Negative means no limit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for json_file in args.json:
        for idx in _pending_indices(json_file, overwrite=args.overwrite, start=args.start, end=args.end):
            command = [sys.executable, args.python_file, json_file, str(idx)]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
