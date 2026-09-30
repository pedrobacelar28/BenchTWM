"""`research sweep|matrix|tune|collect|summarize|reevaluate`."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchtwm.algorithms import get_algorithm
from benchtwm.cli import common
from benchtwm.cli.runs import evaluate, queue_jobs, run_queue
from benchtwm.operations.artifacts import discover_runs, effective_status, initial_manifest, run_directory
from benchtwm.operations.summary import summarize_runs
from benchtwm.paths import PROJECT_ROOT
from benchtwm.runner import run_experiment
from benchtwm.utils.io import ensure_dir, load_json


def sweep(args) -> int:
    jobs = queue_jobs(args, dataset_ids=[args.dataset_id], methods=[args.method_id], seeds=common.parse_seeds(args.seeds))
    return run_queue(args, jobs, title=f"Sweep: {args.dataset_id} {args.method_id}")


def matrix(args) -> int:
    path = Path(args.definition)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    definition = load_json(path)
    default_seeds = [int(seed) for seed in definition.get("seeds", [0])]
    blocks = definition.get("jobs") or [{"datasets": definition["datasets"], "methods": definition.get("methods")}]
    jobs = []
    for block in blocks:
        datasets = block.get("datasets") or [block["dataset"]]
        seeds = [int(seed) for seed in block.get("seeds", default_seeds)]
        jobs += queue_jobs(args, dataset_ids=list(datasets), methods=block.get("methods"), seeds=seeds)
    return run_queue(args, jobs, title=f"Matrix: {path.name}")


def tune(args) -> int:
    import optuna

    console = common.console(args)
    run = common.prepare_run(args, dataset_id=args.dataset_id, method_id=args.method_id, seed=args.seed)
    if not run["preflight"]["ready"]:
        raise FileNotFoundError(f"Dataset {args.dataset_id!r} is incomplete; build it before tuning")
    base = run["params"]
    study_dir = ensure_dir(common.root(args) / "studies")
    name = f"{args.dataset_id.replace('/', '__')}__{base['method_id']}"
    storage = study_dir / f"{name}.sqlite3"
    if args.dry_run:
        if args.json:
            console.emit({"study": name, "trials": args.trials, "base_config": base})
        else:
            console.note("Dry run only. No Optuna trials will be launched.")
        return 0
    study = optuna.create_study(study_name=name, direction="minimize", storage=f"sqlite:///{storage}", load_if_exists=True)

    def objective(trial) -> float:
        # Tuning selects on the offline validation loss, never on the simulator.
        params = {**base, **get_algorithm(base["algo"]).tuning_space(trial), "tuning_trial": int(trial.number)}
        output_dir = run_directory(params, root=common.root(args))
        with common.device(args.device):
            run_experiment(params, output_dir=output_dir, manifest=initial_manifest(params, run_dir=output_dir, device=args.device), force=args.force, quiet=True)
        return float(load_json(output_dir / "metrics_summary.json")["train_summary"]["best_validation_loss"])

    study.optimize(objective, n_trials=args.trials)
    payload = {"study": name, "storage": str(storage), "best_value": study.best_value, "best_params": study.best_params}
    if args.json:
        console.emit(payload)
    else:
        console.table("Best trial", ("field", "value"), payload.items())
    return 0


def collect(args) -> int:
    console = common.console(args)
    rows = []
    for run_dir, manifest in discover_runs(common.root(args)):
        row = {"run_dir": str(run_dir), "run_id": manifest.get("run_id"), "status": effective_status(manifest), "dataset_id": manifest.get("dataset_id"),
               "method_id": manifest.get("method_id"), "seed": manifest.get("seed")}
        if (run_dir / "metrics_summary.json").exists():
            metrics = load_json(run_dir / "metrics_summary.json")
            for section in ("train_summary", "eval_summary", "ope_summary", "timing_summary"):
                row.update({f"{section.removesuffix('_summary')}_{key}": value for key, value in (metrics.get(section) or {}).items()})
        rows.append(row)
    destination = ensure_dir(common.root(args) / "reports") / "run_index.csv"
    fields = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.json:
        console.emit({"output": str(destination), "run_count": len(rows)})
    else:
        console.success(f"Collected {len(rows)} runs into {destination}")
    return 0


def summarize(args) -> int:
    console = common.console(args)
    payload = summarize_runs(common.root(args), output_dir=args.output_dir, dataset_ids=args.datasets)
    if args.json:
        console.emit({key: payload[key] for key in ("output_dir", "best_model_free", "planning_gain")} | {"run_count": len(payload["rows"])})
    else:
        console.success(f"Summary of {len(payload['rows'])} runs written to {payload['output_dir']}")
        console.table("Planning gain vs best model-free method", ("dataset", "method", "baseline", "return gain", "success gain"),
                      [(g["dataset_id"], g["method_id"], g["best_model_free_method_id"], g["planning_gain_return"], g["planning_gain_success"]) for g in payload["planning_gain"]])
    return 0


def reevaluate(args) -> int:
    for reference in args.runs:
        evaluate(argparse.Namespace(**{**vars(args), "run": reference}))
    return 0


def register(subparsers) -> None:
    research = subparsers.add_parser("research", help="Sweeps, matrices, tuning, collection, and summaries.").add_subparsers(dest="research_command", required=True)

    parser = research.add_parser("sweep", help="Queue one method on one dataset over seeds.")
    parser.add_argument("dataset_id")
    parser.add_argument("method_id")
    common.add_queue_options(parser, default_seeds="0,1,2")
    common.add_run_options(parser)
    parser.set_defaults(handler=sweep)

    parser = research.add_parser("matrix", help="Queue a declarative datasets x methods x seeds matrix (configs/research/*.json).")
    parser.add_argument("definition")
    common.add_queue_options(parser, default_seeds="0")
    common.add_run_options(parser)
    parser.set_defaults(handler=matrix)

    parser = research.add_parser("tune", help="Optuna search on the offline validation loss.")
    parser.add_argument("dataset_id")
    parser.add_argument("method_id")
    parser.add_argument("--trials", type=int, default=20)
    common.add_run_options(parser)
    parser.set_defaults(handler=tune)

    research.add_parser("collect", help="Index every run into reports/run_index.csv.").set_defaults(handler=collect)

    parser = research.add_parser("summarize", help="Aggregate completed runs and compute planning gains.")
    parser.add_argument("--dataset", dest="datasets", action="append", help="Limit to a dataset; repeatable.")
    parser.add_argument("--output-dir", help="Defaults to <artifact-root>/reports/summary.")
    parser.set_defaults(handler=summarize)

    parser = research.add_parser("reevaluate", help="Fresh simulator evaluation for several runs.")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(handler=reevaluate)
