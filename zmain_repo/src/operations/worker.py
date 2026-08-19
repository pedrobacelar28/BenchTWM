"""Private subprocess entry point used by the local queue."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.run_experiment import run_experiment
from src.utils.io import load_json
from src.utils.queue_progress import update_progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    args = parser.parse_args()
    job = load_json(Path(args.job))
    progress_path = job.get("progress_path")
    if progress_path:
        os.environ["BENCH_PROGRESS_PATH"] = str(progress_path)
    try:
        run_experiment(
            job["params"],
            force_run=bool(job.get("force", False)),
            output_dir=job["output_dir"],
            manifest=job["manifest"],
            quiet=True,
        )
    except Exception as exc:
        update_progress(
            status="failed",
            phase="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
