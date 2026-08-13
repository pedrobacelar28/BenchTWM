"""Private subprocess entry point used by the local queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.run_experiment import run_experiment
from src.utils.io import load_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    args = parser.parse_args()
    job = load_json(Path(args.job))
    run_experiment(
        job["params"],
        force_run=bool(job.get("force", False)),
        output_dir=job["output_dir"],
        manifest=job["manifest"],
        quiet=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
