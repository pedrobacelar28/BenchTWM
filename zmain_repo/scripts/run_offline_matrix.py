"""Launch a matrix of offline experiments across multiple datasets.

This helper keeps the existing experiment JSONs as the source of truth, but
overrides the dataset paths at launch time so the same algorithm configs can be
reused across multiple offline dataset sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.formatting import output_dir_for_experiment
from src.utils.json_handling import get_param_iterable, get_sorted_dict
from src.utils.methods import method_id


DEFAULT_CONFIGS = {
    "dqn": PROJECT_ROOT / "experiments/offline/dev/dqn.json",
    "ppo": PROJECT_ROOT / "experiments/offline/dev/ppo.json",
    "raw_model_discrete_cem": PROJECT_ROOT / "experiments/offline/dev/raw_model_discrete_cem.json",
    "raw_model_continuous_cem": PROJECT_ROOT / "experiments/offline/dev/raw_model_continuous_cem.json",
    "latent_model_discrete_cem": PROJECT_ROOT / "experiments/offline/dev/latent_model_discrete_cem.json",
    "latent_model_continuous_cem": PROJECT_ROOT / "experiments/offline/dev/latent_model_continuous_cem.json",
    "reconstructive_model_discrete_cem": PROJECT_ROOT / "experiments/offline/dev/reconstructive_model_discrete_cem.json",
    "reconstructive_model_continuous_cem": PROJECT_ROOT / "experiments/offline/dev/reconstructive_model_continuous_cem.json",
}

PRESET_CHOICES = ("legacy", "v3", "v4", "v5")
CEM_MODEL_ALGOS = {"raw_model", "latent_model", "reconstructive_model"}
V3_DATASET_OVERRIDES = {
    "5k": {
        "offline_train_steps": 5000,
        "offline_log_interval": 500,
        "offline_eval_interval": 500,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
    "50k": {
        "offline_train_steps": 50000,
        "offline_log_interval": 1000,
        "offline_eval_interval": 5000,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
    "100k": {
        "offline_train_steps": 100000,
        "offline_log_interval": 1000,
        "offline_eval_interval": 10000,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
}
V4_DATASET_OVERRIDES = {
    "5k": {
        "offline_train_steps": 200000,
        "offline_log_interval": 1000,
        "offline_eval_interval": 5000,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
    "50k": {
        "offline_train_steps": 200000,
        "offline_log_interval": 1000,
        "offline_eval_interval": 20000,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
    "100k": {
        "offline_train_steps": 200000,
        "offline_log_interval": 1000,
        "offline_eval_interval": 20000,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
}
V5_DATASET_OVERRIDES = {
    "5k": {
        "offline_train_steps": 500000,
        "offline_log_interval": 1000,
        "offline_eval_interval": 5000,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
    "50k": {
        "offline_train_steps": 500000,
        "offline_log_interval": 1000,
        "offline_eval_interval": 20000,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
    "100k": {
        "offline_train_steps": 500000,
        "offline_log_interval": 1000,
        "offline_eval_interval": 20000,
        "offline_eval_episodes": 1000,
        "eval_episodes": 5000,
    },
}
V3_DATASET_ALIASES = {
    "5k": "5k",
    "5000": "5k",
    "50k": "50k",
    "50000": "50k",
    "100k": "100k",
    "100000": "100k",
}
V4_GLOBAL_OVERRIDES = {
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.0,
}
V5_GLOBAL_OVERRIDES = {
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.0,
}


@dataclass
class DatasetSpec:
    tag: str
    root: Path
    train_path: Path
    val_path: Path


@dataclass
class Job:
    dataset_tag: str
    algorithm: str
    config_path: Path
    log_path: Path
    output_dir: Path
    metrics_path: Path
    command: list[str]
    gpu: int | None


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_dataset(spec: str) -> DatasetSpec:
    if "=" in spec:
        tag, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        tag = Path(spec).name
    root = _resolve_path(raw_path)
    train_path = root / "train.npz"
    val_path = root / "val.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val dataset: {val_path}")
    return DatasetSpec(tag=tag, root=root, train_path=train_path, val_path=val_path)


def _load_single_experiment(config_path: Path) -> dict:
    config = get_sorted_dict(str(config_path))
    experiments = get_param_iterable(config)
    if len(experiments) != 1:
        raise ValueError(
            f"{config_path} expands to {len(experiments)} experiments. "
            "run_offline_matrix.py expects exactly one resolved config per base file."
        )
    experiment = dict(experiments[0])
    if str(experiment.get("train_mode", "online")).lower() != "offline":
        raise ValueError(f"{config_path} is not an offline config")
    return experiment


def _write_resolved_config(destination: Path, config: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _canonical_v3_dataset_key(dataset: DatasetSpec) -> str:
    candidates = [dataset.tag, dataset.root.name]
    for raw in candidates:
        key = str(raw).strip().lower().replace("_", "").replace("-", "")
        canonical = V3_DATASET_ALIASES.get(key)
        if canonical is not None:
            return canonical
    raise ValueError(
        "Preset v3 expects datasets tagged or rooted as one of "
        f"{sorted(V3_DATASET_ALIASES)}; got tag={dataset.tag!r}, root={str(dataset.root)!r}"
    )


def _apply_preset_overrides(experiment: dict, dataset: DatasetSpec, *, preset: str) -> dict:
    resolved = dict(experiment)
    if preset == "legacy":
        return resolved
    if preset not in {"v3", "v4", "v5"}:
        raise ValueError(f"Unsupported preset={preset!r}")

    dataset_key = _canonical_v3_dataset_key(dataset)
    if preset == "v3":
        resolved.update(V3_DATASET_OVERRIDES[dataset_key])
    elif preset == "v4":
        resolved.update(V4_DATASET_OVERRIDES[dataset_key])
        resolved.update(V4_GLOBAL_OVERRIDES)
    else:
        resolved.update(V5_DATASET_OVERRIDES[dataset_key])
        resolved.update(V5_GLOBAL_OVERRIDES)

    if resolved.get("algo") in CEM_MODEL_ALGOS:
        if preset == "v3":
            resolved["gamma"] = 0.99
        resolved["cem_horizon"] = 10
        if resolved["algo"] == "latent_model":
            if preset != "v3":
                resolved["cem_horizon"] = 8
                resolved["reward_loss_weight"] = 5.0
                resolved["continue_loss_weight"] = 2.0
            resolved["target_encoder_decay"] = 0.99
        elif resolved["algo"] == "reconstructive_model":
            if preset == "v3":
                resolved["latent_loss_weight"] = 2.0
            else:
                resolved["cem_horizon"] = 8
                resolved["reward_loss_weight"] = 5.0
                resolved["continue_loss_weight"] = 2.0
        elif preset == "v4":
            resolved["reward_loss_weight"] = 5.0
            resolved["continue_loss_weight"] = 2.0
    elif preset == "v3":
        return resolved
    else:
        if preset == "v4":
            resolved.update(V4_GLOBAL_OVERRIDES)
        else:
            resolved.update(V5_GLOBAL_OVERRIDES)
    return resolved


def _build_jobs(
    datasets: list[DatasetSpec],
    config_names: list[str],
    python_executable: str,
    gpu_slots: list[int | None],
    run_root: Path,
    preset: str,
) -> tuple[list[Job], list[dict]]:
    jobs: list[Job] = []
    manifest: list[dict] = []
    resolved_dir = run_root / "configs"
    logs_dir = run_root / "logs"
    for dataset in datasets:
        for job_index, config_name in enumerate(config_names):
            config_path = DEFAULT_CONFIGS[config_name]
            experiment = _load_single_experiment(config_path)
            experiment["dataset_path"] = str(dataset.train_path)
            experiment["val_dataset_path"] = str(dataset.val_path)
            experiment = _apply_preset_overrides(experiment, dataset, preset=preset)
            resolved_path = resolved_dir / dataset.tag / config_path.name
            _write_resolved_config(resolved_path, experiment)
            output_dir = output_dir_for_experiment(experiment)
            metrics_path = output_dir / "metrics.pkl"
            gpu = gpu_slots[job_index % len(gpu_slots)] if gpu_slots else None
            log_path = logs_dir / f"{dataset.tag}__{config_path.stem}.log"
            command = [python_executable, "src/run_config.py", str(resolved_path), "0"]
            job = Job(
                dataset_tag=dataset.tag,
                algorithm=config_name,
                config_path=resolved_path,
                log_path=log_path,
                output_dir=output_dir,
                metrics_path=metrics_path,
                command=command,
                gpu=gpu,
            )
            jobs.append(job)
            manifest.append(
                {
                    "dataset_tag": dataset.tag,
                    "dataset_root": str(dataset.root),
                    "algorithm": config_name,
                    "resolved_config": str(resolved_path),
                    "output_dir": str(output_dir),
                    "metrics_exists": metrics_path.exists(),
                    "gpu": gpu,
                    "command": command,
                    "method_id": method_id(experiment),
                    "preset": preset,
                    "offline_train_steps": experiment.get("offline_train_steps"),
                    "offline_log_interval": experiment.get("offline_log_interval"),
                    "offline_eval_interval": experiment.get("offline_eval_interval"),
                    "offline_eval_episodes": experiment.get("offline_eval_episodes"),
                    "eval_episodes": experiment.get("eval_episodes"),
                    "early_stopping_patience": experiment.get("early_stopping_patience"),
                    "early_stopping_min_delta": experiment.get("early_stopping_min_delta"),
                    "gamma": experiment.get("gamma"),
                    "cem_horizon": experiment.get("cem_horizon"),
                }
            )
    return jobs, manifest


def _launch_jobs(
    jobs: list[Job],
    max_parallel: int,
    run_root: Path,
    dry_run: bool,
    launch_done: bool,
    gpu_slots: list[int | None],
) -> int:
    pending = [job for job in jobs if launch_done or not job.metrics_path.exists()]
    completed = len(jobs) - len(pending)
    print(f"Prepared {len(jobs)} jobs under {run_root}")
    print(f"Already complete: {completed}")
    print(f"To launch now: {len(pending)}")
    for job in jobs:
        status = "done" if job.metrics_path.exists() else "pending"
        gpu_label = "cpu" if job.gpu is None else f"gpu{job.gpu}"
        print(f"[{status:7s}] {job.dataset_tag:>8s}  {job.algorithm:<24s}  {gpu_label:<5s}  {job.output_dir}")
        if dry_run:
            print("  " + " ".join(job.command))
    if dry_run or not pending:
        return 0

    slot_count = max(1, max_parallel)
    free_slots = list(range(slot_count))
    slot_to_gpu = {slot: gpu_slots[slot % len(gpu_slots)] if gpu_slots else None for slot in range(slot_count)}

    active: list[dict] = []
    failures = 0
    while pending or active:
        while pending and free_slots and len(active) < slot_count:
            slot = free_slots.pop(0)
            job = pending.pop(0)
            gpu = slot_to_gpu.get(slot)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = job.log_path.open("w", encoding="utf-8")
            print(f"Launching {job.dataset_tag}/{job.algorithm} on {'cpu' if gpu is None else f'gpu{gpu}'}")
            process = subprocess.Popen(
                job.command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append({"slot": slot, "job": job, "proc": process, "log": log_handle})

        if not active:
            break

        time.sleep(5.0)
        still_active = []
        for item in active:
            code = item["proc"].poll()
            if code is None:
                still_active.append(item)
                continue
            item["log"].close()
            free_slots.append(item["slot"])
            free_slots.sort()
            job = item["job"]
            if code == 0:
                print(f"Finished {job.dataset_tag}/{job.algorithm} -> {job.output_dir}")
            else:
                failures += 1
                print(f"FAILED {job.dataset_tag}/{job.algorithm} (exit={code}) -> {job.log_path}")
        active = still_active
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the 8 offline dev algorithms across multiple dataset roots.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset root in the form tag=path. The path must contain train.npz and val.npz.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_CONFIGS.keys()),
        choices=list(DEFAULT_CONFIGS.keys()),
        help="Subset of algorithms to launch. Defaults to all 8 requested offline dev configs.",
    )
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=[0, 1],
        help="GPU ids to use. Pass no values to force CPU-only launching.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Maximum concurrent jobs. Defaults to 2.",
    )
    parser.add_argument(
        "--allow-gpu-oversubscription",
        action="store_true",
        help=(
            "Allow more concurrent jobs than listed GPUs. By default the launcher caps "
            "max-parallel to one job per GPU to avoid CUDA OOM from oversubscription."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved jobs and commands without launching them.",
    )
    parser.add_argument(
        "--launch-done",
        action="store_true",
        help="Launch jobs even when metrics.pkl already exists for the resolved config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for the launcher output folder. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default="legacy",
        help="Resolved-config override preset. Use v3 or v4 for the balanced offline matrix setups.",
    )
    args = parser.parse_args()

    datasets = [_parse_dataset(spec) for spec in args.dataset]
    run_name = args.run_name or time.strftime("offline-matrix-%Y%m%d-%H%M%S")
    run_root = PROJECT_ROOT / "runs" / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    gpu_slots: list[int | None]
    if args.gpus:
        gpu_slots = [int(gpu) for gpu in args.gpus]
    else:
        gpu_slots = [None]
    effective_max_parallel = int(args.max_parallel)
    real_gpu_slots = [gpu for gpu in gpu_slots if gpu is not None]
    if real_gpu_slots and not args.allow_gpu_oversubscription:
        safe_parallel = len(real_gpu_slots)
        if effective_max_parallel > safe_parallel:
            print(
                f"Capping --max-parallel from {effective_max_parallel} to {safe_parallel} "
                f"because {safe_parallel} GPU(s) were provided. "
                "Pass --allow-gpu-oversubscription to override."
            )
            effective_max_parallel = safe_parallel
    jobs, manifest = _build_jobs(datasets, args.algorithms, args.python, gpu_slots, run_root, args.preset)
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    failures = _launch_jobs(
        jobs,
        max_parallel=effective_max_parallel,
        run_root=run_root,
        dry_run=bool(args.dry_run),
        launch_done=bool(args.launch_done),
        gpu_slots=gpu_slots,
    )
    if failures:
        print(f"Completed with {failures} failed job(s). Check logs under {run_root / 'logs'}")
        return 1
    print(f"All requested jobs handled. Logs and resolved configs are under {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
