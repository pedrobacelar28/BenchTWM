"""Collect best Optuna trial result folders into a single directory.

Edit DATASET_SIZE below or pass --dataset-size:

    python scripts/collect_best_trials.py
    python scripts/collect_best_trials.py --dataset-size 50000 --overwrite

Use --overwrite if you want to replace an existing collected folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Easy config. Valid current values are 5_000, 50_000, and 100_000.
DATASET_SIZE = 5_000

# Categories map to runs/optuna-expert-<tag>-<category>.
CATEGORIES = ["baselines", "world-models"]


def _dataset_tag(dataset_size: int) -> str:
    if dataset_size % 1_000 != 0:
        raise ValueError(f"dataset_size must be a multiple of 1000, got {dataset_size}")
    return f"{dataset_size // 1_000}k"


def _default_study_dirs(dataset_size: int, categories: list[str]) -> list[Path]:
    tag = _dataset_tag(dataset_size)
    return [PROJECT_ROOT / "runs" / f"optuna-expert-{tag}-{category}" for category in categories]


def _default_output_dir(dataset_size: int) -> Path:
    return PROJECT_ROOT / "runs" / f"optuna-best-results-{_dataset_tag(dataset_size)}"


def _storage_url(sqlite_path: Path) -> str:
    return f"sqlite:///{sqlite_path}"


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)


def _study_name_from_sqlite(sqlite_path: Path) -> str:
    # Current convention is one study per SQLite and the filename matches the
    # study name. Keeping this separate makes failures easier to diagnose.
    return sqlite_path.stem


def _load_best_trial(sqlite_path: Path):
    import optuna

    study_name = _study_name_from_sqlite(sqlite_path)
    study = optuna.load_study(study_name=study_name, storage=_storage_url(sqlite_path))
    return study, study.best_trial


def _best_output_dir(trial) -> Path:
    output_dir = trial.user_attrs.get("output_dir")
    if not output_dir:
        raise ValueError(f"Best trial {trial.number} does not have user_attr 'output_dir'")
    path = Path(str(output_dir))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _copy_result_dir(source: Path, destination: Path, *, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Best trial output_dir does not exist: {source}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination} (use --overwrite)")
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _discover_sqlite_files(study_dirs: list[Path]) -> list[Path]:
    sqlite_files: list[Path] = []
    for study_dir in study_dirs:
        if not study_dir.exists():
            print(f"[skip] missing study_dir={study_dir}", file=sys.stderr, flush=True)
            continue
        sqlite_files.extend(sorted(study_dir.glob("*.sqlite3")))
    return sorted(sqlite_files)


def _write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest_json = output_dir / "manifest.json"
    manifest_csv = output_dir / "manifest.csv"
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")

    fieldnames = [
        "study_name",
        "trial_number",
        "best_value",
        "state",
        "method_id",
        "algo",
        "dataset_path",
        "source_dir",
        "collected_dir",
        "sqlite_path",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def collect_best_results(study_dirs: list[Path], output_dir: Path, *, overwrite: bool) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_files = _discover_sqlite_files(study_dirs)
    if not sqlite_files:
        raise FileNotFoundError(f"No .sqlite3 studies found in {[str(path) for path in study_dirs]}")

    rows: list[dict[str, Any]] = []
    for sqlite_path in sqlite_files:
        study_name = _study_name_from_sqlite(sqlite_path)
        try:
            study, trial = _load_best_trial(sqlite_path)
            source_dir = _best_output_dir(trial)
            destination = output_dir / _safe_name(study_name)
            _copy_result_dir(source_dir, destination, overwrite=overwrite)
        except Exception as exc:
            print(f"[fail] study={study_name} sqlite={sqlite_path} error={exc!r}", file=sys.stderr, flush=True)
            continue

        row = {
            "study_name": study.study_name,
            "trial_number": int(trial.number),
            "best_value": float(study.best_value),
            "state": str(trial.state.name),
            "method_id": "",
            "algo": "",
            "dataset_path": "",
            "source_dir": str(source_dir),
            "collected_dir": str(destination),
            "sqlite_path": str(sqlite_path),
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
        }
        config_path = destination / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            row["method_id"] = str(config.get("method_id", ""))
            row["algo"] = str(config.get("algo", ""))
            row["dataset_path"] = str(config.get("dataset_path", ""))
        rows.append(row)
        print(
            "[collected] "
            f"study={study.study_name} trial={trial.number} value={study.best_value:.6f} "
            f"dest={destination}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No best result folders were collected")
    _write_manifest(output_dir, rows)
    print(f"[done] collected={len(rows)} output_dir={output_dir}", flush=True)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=DATASET_SIZE,
        choices=[5_000, 50_000, 100_000],
        help=f"Expert dataset size to collect. Default: {DATASET_SIZE}",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=["baselines", "world-models"],
        dest="categories",
        help="Study category to include. Can be passed multiple times. Default: baselines and world-models.",
    )
    parser.add_argument(
        "--study-dir",
        action="append",
        type=Path,
        dest="study_dirs",
        help="Explicit Optuna study directory containing .sqlite3 files. Overrides --dataset-size/--category defaults when passed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where best result folders will be copied. Default: runs/optuna-best-results-<dataset-tag>",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already collected result folders.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    categories = args.categories if args.categories else CATEGORIES
    study_dirs = args.study_dirs if args.study_dirs else _default_study_dirs(args.dataset_size, categories)
    output_dir = args.output_dir if args.output_dir else _default_output_dir(args.dataset_size)
    print(f"[config] dataset_size={args.dataset_size} tag={_dataset_tag(args.dataset_size)}", flush=True)
    print(f"[config] categories={categories}", flush=True)
    print(f"[config] study_dirs={[str(path) for path in study_dirs]}", flush=True)
    print(f"[config] output_dir={output_dir}", flush=True)
    collect_best_results(study_dirs, output_dir, overwrite=bool(args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
