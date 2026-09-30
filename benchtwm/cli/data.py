"""`dataset list|show|build`, `inspect dataset`, and `baselines`."""

from __future__ import annotations

import numpy as np

from benchtwm.cli import common
from benchtwm.data.collect import collect_dataset
from benchtwm.envs import get_environment, make_env
from benchtwm.evaluation import evaluate_policy
from benchtwm.utils.io import ensure_dir, save_json


def _dataset_rows(datasets):
    return [
        (item["id"], item["behavior_policy"], common.compact_count(item["train_episodes"]), "yes" if item["ope_enabled"] else "no",
         "ready" if item["availability"]["ready"] else "build required")
        for item in datasets
    ]


def dataset_list(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    datasets = [common.dataset_payload(bench, common.root(args), entry.identifier) for entry in bench.datasets]
    if args.json:
        console.emit({"datasets": datasets})
    else:
        console.title("Offline Datasets", "Use `dataset show <id>` for splits and compatible methods.")
        console.table("Datasets", ("id", "behavior policy", "train episodes", "OPE", "state"), _dataset_rows(datasets))
    return 0


def dataset_show(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    payload = common.dataset_payload(bench, common.root(args), args.dataset_id)
    if args.json:
        console.emit(payload)
        return 0
    console.title(f"Dataset: {payload['id']}", payload["description"])
    console.table("Identity", ("field", "value"), [(key, value) for key, value in payload.items() if key not in {"availability", "environment_spec", "splits", "methods", "description"}])
    console.table("Files", ("split", "path"), payload["availability"]["paths"].items())
    console.table("Methods", ("method",), [(name,) for name in payload["methods"]])
    return 0


def dataset_build(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    config = bench.dataset_config(args.dataset_id, artifact_root=common.root(args), force=args.force)
    if args.dry_run:
        if args.json:
            console.emit({"dataset_id": args.dataset_id, "config": config})
        else:
            console.title("Dataset Build Plan", args.dataset_id)
            console.table("Configuration", ("field", "value"), config.items())
            console.note("Dry run only. No data will be generated.")
        return 0
    metadata = collect_dataset({**config, "show_progress": not args.no_progress, "quiet": bool(args.json)})
    if args.json:
        console.emit({"dataset_id": args.dataset_id, "config": config, "metadata": metadata})
    else:
        console.success(f"Dataset ready at {config['output_dir']}")
    return 0


def inspect_dataset(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    payload = common.dataset_payload(bench, common.root(args), args.dataset_id)
    status = payload["availability"]
    if status["ready"]:
        train = np.load(status["paths"]["train"])
        payload["arrays"] = {key: {"shape": list(train[key].shape), "dtype": str(train[key].dtype)} for key in sorted(train.files)}
        payload["transitions"] = int(train["actions"].shape[0])
        payload["episodes"] = int(np.unique(train["episode_ids"]).size)
    if args.json:
        console.emit(payload)
    elif not status["ready"]:
        console.warning(f"Dataset is missing: {status['missing']}")
    else:
        console.title("Dataset Inspection", args.dataset_id)
        console.table("Arrays", ("array", "shape", "dtype"), [(key, value["shape"], value["dtype"]) for key, value in payload["arrays"].items()])
    return 0


def baselines(args) -> int:
    bench = common.benchmark(args)
    console = common.console(args)
    entry = bench.dataset(args.dataset_id)
    config = {**bench.dataset_config(entry.identifier, artifact_root=common.root(args)), **bench.environments[entry.env_id].get("run", {})}
    config.setdefault("eval_deterministic", bench.protocol.get("eval_deterministic", True))
    destination = ensure_dir(common.root(args) / "reports" / "baselines" / entry.identifier / common.timestamp())
    names = args.policies or list(get_environment(entry.env_id).reference_policies)
    summaries = {}
    for name in names:
        env = make_env(config)
        try:
            policy = env.make_policy(name, seed=int(config["seed"]), options=config)
        finally:
            env.close()
        summaries[name] = evaluate_policy(
            policy,
            config={**config, "method_id": name},
            num_episodes=args.episodes,
            eval_seed_base=int(bench.protocol.get("eval_seed_base", 100_000)),
            output_dir=destination / name,
            label=name,
            show_progress=not args.no_progress,
        )["summary"]
    save_json(summaries, destination / "baseline_summary.json")
    save_json({"schema_version": 2, "dataset_id": entry.identifier, "environment": entry.env_id, "policies": names}, destination / "baseline_manifest.json")
    if args.json:
        console.emit({"output_dir": str(destination), "summaries": summaries})
    else:
        console.success(f"Baseline evaluation written to {destination}")
        console.table("Reference policies", ("policy", "return", "success", "length"), [(name, s["mean_return"], s["mean_success"], s["mean_length"]) for name, s in summaries.items()])
    return 0


def register(subparsers) -> None:
    dataset = subparsers.add_parser("dataset", help="List, inspect, or build offline datasets.").add_subparsers(dest="dataset_command", required=True)
    dataset.add_parser("list", help="List datasets.").set_defaults(handler=dataset_list)
    show = dataset.add_parser("show", help="Show one dataset and its availability.")
    show.add_argument("dataset_id")
    show.set_defaults(handler=dataset_show)
    build = dataset.add_parser("build", help="Collect train, validation, and OPE splits.")
    build.add_argument("dataset_id")
    build.add_argument("--force", action="store_true")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--no-progress", action="store_true")
    build.set_defaults(handler=dataset_build)

    parser = subparsers.add_parser("baselines", help="Evaluate an environment's reference policies in the simulator.")
    parser.add_argument("dataset_id")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(handler=baselines)
