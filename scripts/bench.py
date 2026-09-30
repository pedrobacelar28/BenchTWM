"""BenchTWM operational CLI.

The public workflow is offline-first:

    doctor -> catalog -> dataset build -> train -> status -> report / ope

All experiment recipes are resolved from ``configs/catalog.json``. Generated
data and run artifacts are written exclusively under ``artifacts/``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchtwm.algorithms import get_algorithm
from benchtwm.envs import get_environment, make_env
from benchtwm.evaluation import evaluate_policy
from benchtwm.data.collect import collect_dataset
from benchtwm.ope import evaluate_ope, export_ope
from benchtwm.operations.artifacts import (
    artifact_root,
    discover_runs,
    effective_status,
    initial_manifest,
    is_complete,
    load_manifest,
    resolve_run,
    run_directory,
)
from benchtwm.operations.benchmark import summarize_pilot
from benchtwm.operations.catalog import Catalog, CatalogError, dataset_artifact_paths, parse_overrides
from benchtwm.operations.console import BenchConsole
from benchtwm.operations.queue import QueueJob, execute_queue
from benchtwm.operations.reporting import build_run_report
from benchtwm.runner import load_run, run_experiment
from benchtwm.utils.io import ensure_dir, load_json, save_json


def _compact_count(value: int | float) -> str:
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}k"
    return str(value)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _catalog(args: argparse.Namespace) -> Catalog:
    catalog = Catalog.load(args.catalog_path)
    errors = catalog.validate()
    if errors:
        raise CatalogError("Catalog validation failed:\n- " + "\n- ".join(errors))
    return catalog


def _root(args: argparse.Namespace) -> Path:
    return artifact_root(args.artifact_root)


def _console(args: argparse.Namespace) -> BenchConsole:
    return BenchConsole(json_mode=bool(args.json), color=args.color)


def _dataset_status(dataset_id: str, root: Path, *, ope_required: bool) -> dict[str, Any]:
    paths = dataset_artifact_paths(dataset_id, root)
    required = {"train": paths["train"], "validation": paths["validation"], "metadata": paths["metadata"]}
    if ope_required:
        required["ope"] = paths["ope"]
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    return {"ready": not missing, "paths": {name: str(path) for name, path in paths.items()}, "missing": missing}


def _dataset_payload(catalog: Catalog, root: Path, dataset_id: str) -> dict[str, Any]:
    dataset = catalog.dataset(dataset_id)
    status = _dataset_status(dataset_id, root, ope_required=dataset.ope_enabled)
    payload: dict[str, Any] = {
        "id": dataset.identifier,
        "environment": dataset.env_id,
        "tier": dataset.tier,
        "behavior_policy": dataset.behavior_policy,
        "train_episodes": dataset.episodes,
        "ope_enabled": dataset.ope_enabled,
        "description": dataset.description,
        "availability": status,
        "compatible_offline_methods": [item.method_id for item in catalog.compatible_methods(dataset.identifier)],
    }
    metadata_path = Path(status["paths"]["metadata"])
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        payload["schema_version"] = metadata.get("schema_version")
        payload["environment_spec"] = metadata.get("environment")
        payload["splits"] = {key: metadata[key] for key in ("train", "validation", "ope_test") if key in metadata}
    return payload


def _offline_preflight(params: dict[str, Any]) -> dict[str, Any]:
    required = {"train": Path(params["dataset_path"]), "validation": Path(params["val_dataset_path"])}
    if bool(params.get("ope_enabled", False)):
        required["ope"] = Path(params["ope_dataset_path"])
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    return {"ready": not missing, "missing": missing, "paths": {name: str(path) for name, path in required.items()}}


def _estimate(params: dict[str, Any]) -> dict[str, Any]:
    steps = int(params.get("offline_train_steps", 0))
    batch = int(params.get("offline_batch_size", 0))
    sequence_length = int(params.get("dreamer_batch_length", 1)) if params.get("algo") == "dreamer" else 1
    return {
        "optimization_steps": steps,
        "batch_size": batch,
        **({"sequence_length": sequence_length} if params.get("algo") == "dreamer" else {}),
        "sampled_transitions_estimate": steps * batch * sequence_length,
        "final_eval_episodes": int(params.get("eval_episodes", 0)),
    }


def _early_stopping_plan(params: dict[str, Any]) -> dict[str, Any]:
    patience = int(params.get("early_stopping_patience", 20))
    return {
        "enabled": patience > 0,
        "monitor": "validation_loss",
        "meaning": get_algorithm(params["algo"]).validation_objective,
        "direction": "minimize",
        "patience_validation_checks": patience,
        "min_delta": float(params.get("early_stopping_min_delta", 0.0)),
        "validation_check_every_steps": int(params.get("offline_log_interval", 0)),
        "checkpoint": "lowest validation_loss",
    }


def _ope_plan(params: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(params.get("ope_enabled", False) and params.get("ope_dataset_path"))
    deterministic = bool(params.get("ope_deterministic", params.get("eval_deterministic", True)))
    return {
        "enabled": enabled,
        "dataset": str(params.get("ope_dataset_path", "")) if enabled else None,
        "target_policy_mode": "deployed_deterministic" if deterministic else "deployed_stochastic",
        "bootstrap_replicates": int(params.get("ope_bootstrap_replicates", 200)) if enabled else 0,
    }


def _ope_target_mode(config: dict[str, Any], requested: str) -> tuple[bool, str]:
    deployed_deterministic = bool(config.get("ope_deterministic", config.get("eval_deterministic", True)))
    if requested == "stochastic":
        return False, "counterfactual_stochastic"
    return deployed_deterministic, "deployed_deterministic" if deployed_deterministic else "deployed_stochastic"


@contextlib.contextmanager
def _device(requested: str) -> Iterator[None]:
    previous = os.environ.get("BENCH_DEVICE")
    os.environ["BENCH_DEVICE"] = requested
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("BENCH_DEVICE", None)
        else:
            os.environ["BENCH_DEVICE"] = previous


def _load_policy(run_dir: Path):
    return load_run(run_dir)


def _cmd_doctor(args: argparse.Namespace) -> int:
    console = _console(args)
    packages = {}
    for name in ("numpy", "pandas", "torch", "rich", "gymnasium", "optuna"):
        try:
            module = importlib.import_module(name)
            packages[name] = getattr(module, "__version__", "available")
        except Exception as exc:
            packages[name] = f"unavailable: {exc}"
    payload = {
        "project_root": str(PROJECT_ROOT),
        "artifact_root": str(_root(args)),
        "python": sys.version.split()[0],
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "vendored_cores": {
            "icu_sepsis": str(PROJECT_ROOT / "third_party" / "icu-sepsis"),
            "epicare": str(PROJECT_ROOT / "third_party" / "epicare"),
        },
    }
    if args.json:
        console.emit(payload)
    else:
        console.title("BenchTWM Doctor", "Runtime, hardware, and local benchmark cores")
        console.table("Runtime", ("item", "value"), [("Python", payload["python"]), ("Artifact root", payload["artifact_root"]), ("CUDA", payload["cuda_available"]), ("GPUs", ", ".join(payload["cuda_devices"]) or "none")])
        console.table("Packages", ("package", "version"), packages.items())
    return 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    console = _console(args)
    datasets = [_dataset_payload(catalog, _root(args), item.identifier) for item in catalog.datasets]
    payload = {"schema_version": catalog.schema_version, "datasets": datasets, "recipes": [item.__dict__ | {"config_path": str(item.config_path)} for item in catalog.recipes]}
    if args.json:
        console.emit(payload)
        return 0
    console.title("BenchTWM Catalog", "Curated datasets and compatible benchmark methods")
    console.table(
        "Datasets",
        ("id", "env", "behavior", "train", "OPE", "state"),
        [
            (item["id"], item["environment"], item["behavior_policy"], _compact_count(item["train_episodes"]), "yes" if item["ope_enabled"] else "no", "ready" if item["availability"]["ready"] else "build required")
            for item in datasets
        ],
    )
    console.table(
        "Offline Methods",
        ("environment", "tier", "method", "track", "algorithm", "description"),
        [(item.env_id, item.tier, item.method_id, item.track, item.algo, item.description) for item in catalog.recipes if item.mode == "offline"],
    )
    return 0


def _cmd_dataset_list(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    console = _console(args)
    payload = {"datasets": [_dataset_payload(catalog, _root(args), item.identifier) for item in catalog.datasets]}
    if args.json:
        console.emit(payload)
    else:
        console.title("Offline Datasets", "Use dataset show <id> for schema and compatible methods.")
        console.table(
            "Datasets",
            ("id", "environment", "behavior", "train", "OPE", "state"),
            [
                (item["id"], item["environment"], item["behavior_policy"], _compact_count(item["train_episodes"]), "yes" if item["ope_enabled"] else "no", "ready" if item["availability"]["ready"] else "build required")
                for item in payload["datasets"]
            ],
        )
    return 0


def _cmd_dataset_show(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    console = _console(args)
    payload = _dataset_payload(catalog, _root(args), args.dataset_id)
    if args.json:
        console.emit(payload)
    else:
        console.title(f"Dataset: {payload['id']}", payload["description"])
        console.table("Identity", ("field", "value"), [(key, value) for key, value in payload.items() if key not in {"availability", "environment_spec", "splits", "compatible_offline_methods"}])
        console.table("Availability", ("split", "path"), payload["availability"]["paths"].items())
        console.table("Compatible Offline Methods", ("method",), [(name,) for name in payload["compatible_offline_methods"]])
        if payload.get("splits"):
            console.note("Dataset metadata is present and describes train, validation, and optional OPE splits.")
    return 0


def _cmd_dataset_build(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    console = _console(args)
    config = catalog.dataset_config(args.dataset_id, artifact_root=_root(args), force=args.force)
    status = _dataset_status(args.dataset_id, _root(args), ope_required=bool(config.get("ope_enabled", False)))
    payload = {"dataset_id": args.dataset_id, "config": config, "availability_before": status}
    if args.dry_run:
        if args.json:
            console.emit(payload)
        else:
            console.title("Dataset Build Plan", args.dataset_id)
            console.table("Configuration", ("field", "value"), config.items())
            console.note("Dry run only. No data will be generated.")
        return 0
    metadata = collect_dataset({**config, "show_progress": not args.no_progress, "quiet": bool(args.json)})
    payload["metadata"] = metadata
    if args.json:
        console.emit(payload)
    else:
        console.success(f"Dataset ready at {config['output_dir']}")
    return 0


def _prepare_run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    catalog = _catalog(args)
    overrides = parse_overrides(args.overrides)
    params, dataset, recipe = catalog.run_config(
        args.dataset_id,
        args.method_id,
        artifact_root=_root(args),
        seed=args.seed,
        overrides=overrides,
    )
    params["show_progress"] = not args.no_progress
    preflight = _offline_preflight(params)
    output_dir = run_directory(params, dataset_id=dataset.identifier, method_id=recipe.method_id, root=_root(args))
    manifest = initial_manifest(params, dataset_id=dataset.identifier, method_id=recipe.method_id, run_dir=output_dir, device=args.device)
    return params, preflight, output_dir, manifest


def _print_run_plan(console: BenchConsole, params: dict[str, Any], preflight: dict[str, Any], output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "run_id": manifest["run_id"],
        "output_dir": str(output_dir),
        "environment": params["env_id"],
        "dataset_id": params["dataset_id"],
        "task_id": params["task_id"],
        "recipe_id": params["recipe_id"],
        "algorithm": params["algo"],
        "track": params.get("track"),
        "mode": params["train_mode"],
        "seed": params["seed"],
        "observation_backend": params["obs_backend"],
        "estimate": _estimate(params),
        "early_stopping": _early_stopping_plan(params),
        "ope": _ope_plan(params),
        "preflight": preflight,
    }
    if console.json_mode:
        return payload
    console.title("Training Plan", "Offline optimization, simulator evaluation, and OPE remain separate evidence streams.")
    console.table(
        "Run",
        ("field", "value"),
        [(key, payload[key]) for key in ("run_id", "output_dir", "environment", "dataset_id", "task_id", "recipe_id", "algorithm", "track", "mode", "seed", "observation_backend")],
    )
    console.table("Execution Budget", ("field", "value"), payload["estimate"].items())
    console.table("Early Stopping", ("field", "value"), payload["early_stopping"].items())
    if payload["ope"]["enabled"]:
        console.table("Held-out OPE", ("field", "value"), payload["ope"].items())
    console.table("Data Preflight", ("field", "value"), [("ready", preflight["ready"]), ("missing", preflight["missing"]), ("paths", preflight["paths"])])
    if not preflight["ready"]:
        console.warning("Dataset is incomplete. Build it before training.")
    return payload


def _cmd_train(args: argparse.Namespace) -> int:
    console = _console(args)
    params, preflight, output_dir, manifest = _prepare_run(args)
    plan = _print_run_plan(console, params, preflight, output_dir, manifest)
    if args.dry_run:
        if args.json:
            console.emit(plan)
        return 0
    if not preflight["ready"]:
        raise FileNotFoundError(
            f"Dataset {args.dataset_id!r} is not ready. Run: python scripts/bench.py dataset build {args.dataset_id}"
        )
    with _device(args.device):
        actual_output = run_experiment(params, output_dir=output_dir, manifest=manifest, force=args.force, quiet=True)
    if args.json:
        console.emit({**plan, "actual_output_dir": str(actual_output), "status": "completed"})
    else:
        console.success(f"Completed {manifest['run_id']}")
        console.note(f"Artifacts: {actual_output}")
    return 0


def _parse_seeds(raw: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must contain comma-separated integers") from exc
    if not values:
        raise ValueError("--seeds must contain at least one seed")
    return values


def _cmd_suite(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    console = _console(args)
    methods = args.methods or [item.method_id for item in catalog.compatible_methods(args.dataset_id)]
    seeds = _parse_seeds(args.seeds)
    jobs: list[QueueJob] = []
    plans = []
    for method_id in methods:
        for seed in seeds:
            copied = argparse.Namespace(**vars(args))
            copied.method_id = method_id
            copied.seed = seed
            params, preflight, output_dir, manifest = _prepare_run(copied)
            plans.append({"method": method_id, "seed": seed, "output_dir": str(output_dir), "preflight": preflight, "estimate": _estimate(params)})
            jobs.append(QueueJob(name=f"{method_id}-seed-{seed}", params=params, output_dir=str(output_dir), manifest=manifest, device=args.device, force=args.force))
    payload = {"dataset_id": args.dataset_id, "jobs": plans}
    if args.dry_run or not args.execute:
        if args.json:
            console.emit(payload)
        else:
            console.title("Suite Plan", "Use --execute after reviewing the resolved jobs.")
            console.table("Jobs", ("method", "seed", "ready", "output"), [(item["method"], item["seed"], item["preflight"]["ready"], item["output_dir"]) for item in plans])
        return 0
    missing = [plan for plan in plans if not plan["preflight"]["ready"]]
    if missing:
        raise FileNotFoundError(f"Dataset {args.dataset_id!r} is incomplete. Run dataset build before executing the suite.")
    queue_dir, manifest = execute_queue(
        jobs,
        root=_root(args),
        max_workers=args.jobs,
        gpus=args.gpus,
        resume=not args.force,
        follow=bool(getattr(args, "follow", False) and not args.json),
    )
    if args.json:
        console.emit({"queue_dir": str(queue_dir), "suite_results": str(queue_dir / "suite_results.json"), "manifest": manifest})
    else:
        console.success(f"Queue completed: {queue_dir}")
        console.table("Queue Jobs", ("name", "status", "log"), [(item["name"], item["status"], item["log_path"]) for item in manifest["jobs"]])
        console.note(f"Suite results: {queue_dir / 'suite_results.json'}")
    return 0 if not manifest["failures"] else 1


def _run_summary(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "status": effective_status(manifest),
        "dataset_id": manifest.get("dataset_id"),
        "task_id": manifest.get("task_id"),
        "method_id": manifest.get("method_id"),
        "track": manifest.get("track"),
        "environment": manifest.get("environment"),
        "seed": manifest.get("seed"),
        "updated_at": manifest.get("updated_at"),
    }


def _cmd_status(args: argparse.Namespace) -> int:
    console = _console(args)
    if args.run:
        run_dir = resolve_run(args.run, _root(args))
        payload = _run_summary(run_dir, load_manifest(run_dir))
        payload["artifacts"] = sorted(path.name for path in run_dir.iterdir() if path.is_file())
        if args.json:
            console.emit(payload)
        else:
            console.title("Run Status", payload["run_id"])
            console.table("Identity", ("field", "value"), payload.items())
        return 0
    runs = [_run_summary(path, manifest) for path, manifest in discover_runs(_root(args))[: args.limit]]
    if args.json:
        console.emit({"runs": runs})
    else:
        console.title("Run Status", f"Most recent {len(runs)} clean-layout runs")
        console.table("Runs", ("status", "dataset", "method", "seed", "updated", "directory"), [(item["status"], item["dataset_id"], item["method_id"], item["seed"], item["updated_at"], item["run_dir"]) for item in runs])
    return 0


def _cmd_inspect_dataset(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    console = _console(args)
    payload = _dataset_payload(catalog, _root(args), args.dataset_id)
    status = payload["availability"]
    if status["ready"]:
        train = np.load(status["paths"]["train"])
        payload["arrays"] = {key: {"shape": list(train[key].shape), "dtype": str(train[key].dtype)} for key in sorted(train.files)}
        payload["transitions"] = int(train["actions"].shape[0])
        payload["episodes"] = int(np.unique(train["episode_ids"]).size)
    if args.json:
        console.emit(payload)
    else:
        console.title("Dataset Inspection", args.dataset_id)
        if not status["ready"]:
            console.warning(f"Dataset is missing: {status['missing']}")
        else:
            console.table("Arrays", ("array", "shape", "dtype"), [(key, value["shape"], value["dtype"]) for key, value in payload["arrays"].items()])
    return 0


def _cmd_inspect_run(args: argparse.Namespace) -> int:
    console = _console(args)
    run_dir = resolve_run(args.run, _root(args))
    payload = _run_summary(run_dir, load_manifest(run_dir))
    payload["optimization"] = load_json(run_dir / "metrics_summary.json").get("train_summary", {}) if (run_dir / "metrics_summary.json").exists() else {}
    payload["direct_evaluation"] = load_json(run_dir / "eval_summary.json") if (run_dir / "eval_summary.json").exists() else {}
    payload["ope"] = load_json(run_dir / "ope_summary.json") if (run_dir / "ope_summary.json").exists() else {}
    if payload["ope"]:
        payload["ope_recompute_recommended"] = int(payload["ope"].get("schema_version", 0)) < 2
    if args.json:
        console.emit(payload)
    else:
        console.title("Run Inspection", payload["run_id"])
        console.table("Optimization", ("metric", "value"), payload["optimization"].items())
        console.table("Direct Evaluation", ("metric", "value"), payload["direct_evaluation"].items())
        if payload["ope"]:
            console.table("OPE Diagnostics", ("metric", "value"), payload["ope"].get("diagnostics", {}).items())
            console.table("OPE Reliability", ("metric", "value"), payload["ope"].get("reliability", {}).items())
            if payload["ope_recompute_recommended"]:
                console.warning("This OPE summary predates the WPDIS correction and reliability diagnostics. Recompute it with: python scripts/bench.py ope <run> --write-canonical")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    console = _console(args)
    run_dir = resolve_run(args.run, _root(args))
    config, policy = _load_policy(run_dir)
    destination = ensure_dir(run_dir / "evaluations" / _timestamp())
    with _device(args.device):
        result = evaluate_policy(
            policy,
            config=config,
            num_episodes=args.episodes or int(config.get("eval_episodes", 100)),
            eval_seed_base=args.seed_base if args.seed_base is not None else int(config.get("eval_seed_base", 100000)),
            output_dir=destination,
            label="best-re-evaluated",
            export_csv=True,
            show_progress=not args.no_progress,
        )
    save_json(result["summary"], destination / "eval_summary.json")
    payload = {"run": str(run_dir), "evaluation_dir": str(destination), "summary": result["summary"]}
    if args.json:
        console.emit(payload)
    else:
        console.success(f"Evaluation written to {destination}")
        console.table("Direct Simulator Evaluation", ("metric", "value"), result["summary"].items())
    return 0


def _cmd_ope(args: argparse.Namespace) -> int:
    console = _console(args)
    if args.write_canonical and args.target_mode != "deployed":
        raise ValueError("--write-canonical is only valid with --target-mode deployed")
    run_dir = resolve_run(args.run, _root(args))
    config, policy = _load_policy(run_dir)
    dataset_path = Path(config.get("ope_dataset_path", ""))
    if not dataset_path.exists():
        raise FileNotFoundError("This run has no available held-out OPE dataset")
    destination = ensure_dir(run_dir / "ope" / _timestamp())
    deterministic, target_mode = _ope_target_mode(config, args.target_mode)
    summary, bootstrap = evaluate_ope(
        policy,
        dataset_path=dataset_path,
        gamma=float(config.get("gamma", 1.0)),
        bootstrap_replicates=args.bootstrap,
        seed=int(config.get("seed", 0)),
        deterministic=deterministic,
        target_policy_mode=target_mode,
        show_progress=not args.no_progress,
    )
    summary["task_id"] = str(config.get("task_id", ""))
    summary["track"] = str(config.get("track", ""))
    export_ope(summary, bootstrap, destination)
    canonical_updated = False
    if args.write_canonical:
        export_ope(summary, bootstrap, run_dir)
        canonical_updated = True
    payload = {"run": str(run_dir), "ope_dir": str(destination), "canonical_updated": canonical_updated, "summary": summary}
    if args.json:
        console.emit(payload)
    else:
        console.success(f"OPE written to {destination}")
        console.table("OPE Target", ("field", "value"), summary["config"].items())
        console.table("OPE Diagnostics", ("metric", "value"), summary["diagnostics"].items())
        console.table("OPE Reliability", ("metric", "value"), summary["reliability"].items())
        console.table("Policy Probability Contract", ("metric", "value"), summary["policy_probability_contract"].items())
        console.table("OPE Estimates", ("estimator", "estimate", "CI 95%"), [(name, values["estimate"], f"[{values['ci95_low']}, {values['ci95_high']}]") for name, values in summary["estimators"].items()])
        if args.target_mode == "stochastic":
            console.warning("This evaluates a stochastic counterfactual policy, not a deterministic deployed policy.")
        if summary["reliability"]["status"] != "usable":
            console.warning(summary["reliability"]["message"])
        if summary["policy_probability_contract"]["status"] != "exact":
            console.warning(summary["policy_probability_contract"]["message"])
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    console = _console(args)
    run_dir = resolve_run(args.run, _root(args))
    destination, summary = build_run_report(run_dir, root=_root(args))
    if args.json:
        console.emit({"report_dir": str(destination), "summary": summary})
    else:
        console.success(f"Report written to {destination}")
        console.note("The report keeps optimization, direct evaluation, and OPE in separate sections.")
    return 0


def _cmd_baselines(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    console = _console(args)
    dataset = catalog.dataset(args.dataset_id)
    config = catalog.dataset_config(dataset.identifier, artifact_root=_root(args))
    destination = ensure_dir(_root(args) / "reports" / "baselines" / dataset.identifier / _timestamp())
    policies = args.policies or list(get_environment(dataset.env_id).reference_policies)
    summaries = {}
    for name in policies:
        env = make_env(config)
        try:
            policy = env.make_policy(name, seed=int(config["seed"]), options=config)
        finally:
            env.close()
        result = evaluate_policy(
            policy,
            config={**config, "algo": "baseline", "method_id": name},
            num_episodes=args.episodes,
            eval_seed_base=100000,
            output_dir=destination / name,
            label=name,
            export_csv=True,
            show_progress=not args.no_progress,
        )
        summaries[name] = result["summary"]
    save_json(summaries, destination / "baseline_summary.json")
    save_json(
        {
            "schema_version": 1,
            "dataset_id": dataset.identifier,
            "task_id": config.get("task_id"),
            "track": "reference_baseline",
            "environment": dataset.env_id,
            "policies": list(summaries),
        },
        destination / "baseline_manifest.json",
    )
    if args.json:
        console.emit({"output_dir": str(destination), "summaries": summaries})
    else:
        console.success(f"Baseline evaluation written to {destination}")
        console.table("Direct Baselines", ("policy", "return", "success", "adverse events", "length"), [(name, values.get("mean_return"), values.get("mean_success"), values.get("outcome_adverse_event", "-"), values.get("mean_length")) for name, values in summaries.items()])
    return 0


def _research_jobs(args: argparse.Namespace, *, dataset_ids: list[str], methods: list[str], seeds: list[int]) -> list[QueueJob]:
    jobs: list[QueueJob] = []
    for dataset_id in dataset_ids:
        for method_id in methods:
            for seed in seeds:
                copied = argparse.Namespace(**vars(args))
                copied.dataset_id = dataset_id
                copied.method_id = method_id
                copied.seed = seed
                params, preflight, output_dir, manifest = _prepare_run(copied)
                if not preflight["ready"]:
                    raise FileNotFoundError(f"Dataset {dataset_id!r} is incomplete; build it before research execution")
                jobs.append(QueueJob(name=f"{dataset_id}:{method_id}:seed-{seed}", params=params, output_dir=str(output_dir), manifest=manifest, device=args.device, force=args.force))
    return jobs


def _cmd_research_sweep(args: argparse.Namespace) -> int:
    copied = argparse.Namespace(**vars(args))
    copied.methods = [args.method_id]
    copied.seeds = args.seeds
    copied.execute = args.execute
    return _cmd_suite(copied)


def _cmd_research_matrix(args: argparse.Namespace) -> int:
    console = _console(args)
    definition_path = Path(args.definition)
    if not definition_path.is_absolute():
        definition_path = PROJECT_ROOT / definition_path
    definition = load_json(definition_path)
    copied = argparse.Namespace(**vars(args))
    if "jobs" in definition:
        jobs = []
        for item in definition["jobs"]:
            jobs.extend(
                _research_jobs(
                    copied,
                    dataset_ids=[str(item["dataset"])],
                    methods=[str(method) for method in item["methods"]],
                    seeds=[int(seed) for seed in item.get("seeds", definition.get("seeds", [0]))],
                )
            )
    else:
        dataset_ids = list(definition["datasets"])
        methods = list(definition["methods"])
        seeds = [int(value) for value in definition.get("seeds", [0])]
        jobs = _research_jobs(copied, dataset_ids=dataset_ids, methods=methods, seeds=seeds)
    if args.dry_run or not args.execute:
        payload = {"definition": str(definition_path), "jobs": [{"name": job.name, "output_dir": job.output_dir} for job in jobs]}
        if args.json:
            console.emit(payload)
        else:
            console.title("Research Matrix Plan", str(definition_path))
            console.table("Jobs", ("name", "output"), [(item["name"], item["output_dir"]) for item in payload["jobs"]])
        return 0
    queue_dir, manifest = execute_queue(
        jobs,
        root=_root(args),
        max_workers=args.jobs,
        gpus=args.gpus,
        resume=not args.force,
        follow=bool(getattr(args, "follow", False) and not args.json),
    )
    if args.json:
        console.emit({"queue_dir": str(queue_dir), "suite_results": str(queue_dir / "suite_results.json"), "manifest": manifest})
    else:
        console.success(f"Research matrix completed: {queue_dir}")
        console.note(f"Suite results: {queue_dir / 'suite_results.json'}")
    return 0 if not manifest["failures"] else 1


def _cmd_research_tune(args: argparse.Namespace) -> int:
    import optuna

    catalog = _catalog(args)
    console = _console(args)
    base_params, preflight, _, _ = _prepare_run(args)
    if not preflight["ready"]:
        raise FileNotFoundError(f"Dataset {args.dataset_id!r} is incomplete; build it before tuning")
    study_dir = ensure_dir(_root(args) / "studies")
    study_name = f"{args.dataset_id.replace('/', '__')}__{args.method_id.replace('/', '__')}"
    study = optuna.create_study(study_name=study_name, direction="minimize", storage=f"sqlite:///{study_dir / (study_name + '.sqlite3')}", load_if_exists=True)

    def objective(trial):
        params = dict(base_params)
        params.update(get_algorithm(params["algo"]).tuning_space(trial))
        params["tuning_trial"] = int(trial.number)
        output_dir = run_directory(params, dataset_id=args.dataset_id, method_id=args.method_id, root=_root(args))
        manifest = initial_manifest(params, dataset_id=args.dataset_id, method_id=args.method_id, run_dir=output_dir, device=args.device)
        with _device(args.device):
            run_experiment(params, output_dir=output_dir, manifest=manifest, force=args.force, quiet=True)
        summary = load_json(output_dir / "metrics_summary.json").get("train_summary", {})
        value = summary.get("best_validation_loss")
        if value is None:
            raise RuntimeError("The selected method did not emit best_validation_loss for Optuna")
        return float(value)

    if args.dry_run:
        payload = {"study": study_name, "trials": args.trials, "base_config": base_params}
        if args.json:
            console.emit(payload)
        else:
            console.title("Tuning Plan", study_name)
            console.note("Dry run only. No Optuna trials will be launched.")
        return 0
    study.optimize(objective, n_trials=args.trials)
    payload = {"study": study_name, "storage": str(study_dir / (study_name + '.sqlite3')), "best_value": study.best_value, "best_params": study.best_params}
    if args.json:
        console.emit(payload)
    else:
        console.success(f"Tuning completed: {study_name}")
        console.table("Best Trial", ("field", "value"), payload.items())
    return 0


def _cmd_research_collect(args: argparse.Namespace) -> int:
    console = _console(args)
    rows = []
    for run_dir, manifest in discover_runs(_root(args)):
        row = _run_summary(run_dir, manifest)
        summary_path = run_dir / "metrics_summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            for section in ("train_summary", "eval_summary", "ope_summary", "timing_summary"):
                row.update({f"{section.split('_')[0]}_{key}": value for key, value in (summary.get(section) or {}).items()})
        rows.append(row)
    destination = ensure_dir(_root(args) / "reports") / "run_index.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.json:
        console.emit({"output": str(destination), "run_count": len(rows), "columns": fieldnames})
    else:
        console.success(f"Collected {len(rows)} runs into {destination}")
    return 0


def _cmd_research_summarize(args: argparse.Namespace) -> int:
    console = _console(args)
    payload = summarize_pilot(_root(args), output_dir=Path(args.output_dir) if args.output_dir else None)
    if args.json:
        console.emit(
            {
                "output_dir": payload["output_dir"],
                "run_count": len(payload["rows"]),
                "aggregate_count": len(payload["aggregate"]),
                "planning_gain_count": len(payload["planning_gain"]),
            }
        )
    else:
        console.success(f"v0 pilot summary written to {payload['output_dir']}")
        console.table(
            "Best direct baseline",
            ("task", "method", "return", "success"),
            [
                (task, item["method_id"], item.get("mean_return_mean"), item.get("mean_success_mean"))
                for task, item in sorted(payload["best_direct_baselines"].items())
            ],
        )
    return 0


def _cmd_research_reevaluate(args: argparse.Namespace) -> int:
    console = _console(args)
    outputs = []
    for reference in args.runs:
        copied = argparse.Namespace(**vars(args))
        copied.run = reference
        _cmd_evaluate(copied)
        outputs.append(reference)
    if not args.json:
        console.success(f"Re-evaluated {len(outputs)} run(s)")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    console = _console(args)
    payload: dict[str, Any] = {"catalog": "valid", "datasets": len(catalog.datasets), "recipes": len(catalog.recipes), "smoke": []}
    if args.smoke:
        for env_id in sorted(catalog.payload["environments"]):
            recipe = next(item for item in catalog.recipes if item.env_id == env_id)
            config = Catalog._materialize(load_json(recipe.config_path), seed=0)
            config.update({"env_id": env_id, "environment_seed": 1, "env_kwargs": {}})
            env = make_env(config, seed=0)
            try:
                _, info = env.reset(seed=0)
                action = int(info["admissible_actions"][0])
                _, reward, terminated, truncated, _ = env.step(action)
                payload["smoke"].append({"env_id": env_id, "reward": float(reward), "done": bool(terminated or truncated)})
            finally:
                env.close()
    if args.json:
        console.emit(payload)
    else:
        console.success("Catalog and recipes are valid")
        if payload["smoke"]:
            console.table("Environment Smoke", ("env", "reward", "done"), [(item["env_id"], item["reward"], item["done"]) for item in payload["smoke"]])
    return 0


def _add_common(parser: argparse.ArgumentParser, *, run: bool = False) -> None:
    if run:
        parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=JSON", help="Typed config override applied only to this run.")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        parser.add_argument("--force", action="store_true", help="Run even when a completed artifact already exists.")
        parser.add_argument("--dry-run", action="store_true", help="Resolve and display work without writing artifacts.")
        parser.add_argument("--no-progress", action="store_true")


def _normalize_global_flags(argv: list[str]) -> list[str]:
    """Allow global display/path flags before or after a subcommand."""

    leading: list[str] = []
    remaining: list[str] = []
    index = 0
    value_flags = {"--artifact-root", "--catalog-path", "--color"}
    while index < len(argv):
        token = argv[index]
        if token == "--json":
            leading.append(token)
        elif token in value_flags:
            if index + 1 >= len(argv):
                remaining.append(token)
            else:
                leading.extend((token, argv[index + 1]))
                index += 1
        else:
            remaining.append(token)
        index += 1
    return [*leading, *remaining]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--catalog-path", default=str(PROJECT_ROOT / "configs" / "catalog.json"))
    parser.add_argument("--color", choices=["auto", "never"], default="auto")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON for inspection commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect runtime dependencies, hardware, and vendored cores.")
    doctor.set_defaults(handler=_cmd_doctor)
    catalog = subparsers.add_parser("catalog", help="List curated datasets and method recipes.")
    catalog.set_defaults(handler=_cmd_catalog)

    dataset = subparsers.add_parser("dataset", help="List, inspect, or build offline datasets.")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_list = dataset_sub.add_parser("list", help="List curated datasets.")
    dataset_list.set_defaults(handler=_cmd_dataset_list)
    dataset_show = dataset_sub.add_parser("show", help="Show one dataset recipe and availability.")
    dataset_show.add_argument("dataset_id")
    dataset_show.set_defaults(handler=_cmd_dataset_show)
    dataset_build = dataset_sub.add_parser("build", help="Generate train, validation, and optional OPE splits.")
    dataset_build.add_argument("dataset_id")
    dataset_build.add_argument("--force", action="store_true")
    dataset_build.add_argument("--dry-run", action="store_true")
    dataset_build.add_argument("--no-progress", action="store_true")
    dataset_build.set_defaults(handler=_cmd_dataset_build)

    train = subparsers.add_parser("train", help="Train one named offline recipe.")
    train.add_argument("dataset_id")
    train.add_argument("method_id")
    _add_common(train, run=True)
    train.set_defaults(handler=_cmd_train)

    suite = subparsers.add_parser("suite", help="Plan or execute a local queue over named methods and seeds.")
    suite.add_argument("dataset_id")
    suite.add_argument("--method", dest="methods", action="append", help="Limit to a compatible method; repeatable.")
    suite.add_argument("--seeds", default="0")
    suite.add_argument("--execute", action="store_true")
    suite.add_argument("--jobs", type=int, default=1)
    suite.add_argument("--gpus")
    suite.add_argument("--follow", action="store_true", help="Show live job and optimization progress while the suite runs.")
    _add_common(suite, run=True)
    suite.set_defaults(handler=_cmd_suite)

    status = subparsers.add_parser("status", help="Show clean-layout run status.")
    status.add_argument("run", nargs="?")
    status.add_argument("--limit", type=int, default=20)
    status.set_defaults(handler=_cmd_status)

    inspect = subparsers.add_parser("inspect", help="Inspect a named dataset or clean-layout run.")
    inspect_sub = inspect.add_subparsers(dest="inspect_command", required=True)
    inspect_dataset = inspect_sub.add_parser("dataset")
    inspect_dataset.add_argument("dataset_id")
    inspect_dataset.set_defaults(handler=_cmd_inspect_dataset)
    inspect_run = inspect_sub.add_parser("run")
    inspect_run.add_argument("run")
    inspect_run.set_defaults(handler=_cmd_inspect_run)

    evaluate = subparsers.add_parser("evaluate", help="Run fresh simulator evaluation for a completed checkpoint.")
    evaluate.add_argument("run")
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--seed-base", type=int)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate.add_argument("--no-progress", action="store_true")
    evaluate.set_defaults(handler=_cmd_evaluate)
    ope = subparsers.add_parser("ope", help="Recompute OPE from a run's held-out logged trajectories.")
    ope.add_argument("run")
    ope.add_argument("--bootstrap", type=int, default=200)
    ope.add_argument("--target-mode", choices=["deployed", "stochastic"], default="deployed")
    ope.add_argument("--write-canonical", action="store_true", help="Replace the run-level OPE summary when evaluating the deployed target policy.")
    ope.add_argument("--no-progress", action="store_true")
    ope.set_defaults(handler=_cmd_ope)
    report = subparsers.add_parser("report", help="Write an environment-agnostic report for one run.")
    report.add_argument("run")
    report.set_defaults(handler=_cmd_report)
    baselines = subparsers.add_parser("baselines", help="Evaluate declared environment reference policies.")
    baselines.add_argument("dataset_id")
    baselines.add_argument("--episodes", type=int, default=1000)
    baselines.add_argument("--policies", nargs="+")
    baselines.add_argument("--no-progress", action="store_true")
    baselines.set_defaults(handler=_cmd_baselines)

    research = subparsers.add_parser("research", help="Advanced sweeps, matrices, tuning, collection, and reevaluation.")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    sweep = research_sub.add_parser("sweep", help="Run a method across explicit seeds.")
    sweep.add_argument("dataset_id")
    sweep.add_argument("method_id")
    sweep.add_argument("--seeds", default="0,1,2")
    sweep.add_argument("--execute", action="store_true")
    sweep.add_argument("--jobs", type=int, default=1)
    sweep.add_argument("--gpus")
    sweep.add_argument("--follow", action="store_true", help="Show live job and optimization progress while the sweep runs.")
    _add_common(sweep, run=True)
    sweep.set_defaults(handler=_cmd_research_sweep)
    matrix = research_sub.add_parser("matrix", help="Run a declarative dataset-method matrix.")
    matrix.add_argument("definition", help="Path under configs/research or an absolute JSON path.")
    matrix.add_argument("--execute", action="store_true")
    matrix.add_argument("--jobs", type=int, default=1)
    matrix.add_argument("--gpus")
    matrix.add_argument("--follow", action="store_true", help="Show live job and optimization progress while the matrix runs.")
    _add_common(matrix, run=True)
    matrix.set_defaults(handler=_cmd_research_matrix)
    tune = research_sub.add_parser("tune", help="Run Optuna on one named offline recipe.")
    tune.add_argument("dataset_id")
    tune.add_argument("method_id")
    tune.add_argument("--trials", type=int, default=20)
    _add_common(tune, run=True)
    tune.set_defaults(handler=_cmd_research_tune)
    collect = research_sub.add_parser("collect", help="Collect clean-layout runs into a CSV index.")
    collect.set_defaults(handler=_cmd_research_collect)
    summarize = research_sub.add_parser("summarize", help="Aggregate the v0 pilot by task, method, seed, and track.")
    summarize.add_argument("--output-dir", help="Optional report directory (defaults to artifacts/reports/v0_pilot).")
    summarize.set_defaults(handler=_cmd_research_summarize)
    reevaluate = research_sub.add_parser("reevaluate", help="Run fresh evaluation for one or more completed runs.")
    reevaluate.add_argument("runs", nargs="+")
    reevaluate.add_argument("--episodes", type=int)
    reevaluate.add_argument("--seed-base", type=int)
    reevaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    reevaluate.add_argument("--no-progress", action="store_true")
    reevaluate.set_defaults(handler=_cmd_research_reevaluate)

    validate = subparsers.add_parser("validate", help="Validate catalog and optionally smoke-test each environment.")
    validate.add_argument("--smoke", action="store_true")
    validate.set_defaults(handler=_cmd_validate)
    return parser.parse_args(_normalize_global_flags(sys.argv[1:]))


def main() -> int:
    args = _parse_args()
    try:
        return int(args.handler(args))
    except (CatalogError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        _console(args).error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
