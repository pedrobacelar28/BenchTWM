"""Shared helpers for Optuna tuning scripts."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class StudySpec:
    name: str
    gpu: int
    base_config: Path


SuggestFn = Callable[[Any, dict[str, Any]], None]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def objective_metric(output_dir: Path) -> float:
    eval_summary_path = output_dir / "eval_summary.json"
    metrics_summary_path = output_dir / "metrics_summary.json"
    if eval_summary_path.exists():
        return float(load_json(eval_summary_path)["mean_survival"])
    if metrics_summary_path.exists():
        summary = load_json(metrics_summary_path)
        if "eval_summary" in summary:
            return float(summary["eval_summary"]["mean_survival"])
        return float(summary["train_summary"]["final_survival_rate"])
    raise FileNotFoundError(f"No summary metrics found in {output_dir}")


def trial_fields(params: dict[str, Any]) -> str:
    keys = (
        "gamma",
        "offline_batch_size",
        "learning_rate",
        "policy_lr",
        "q_lr",
        "tau",
        "target_network_frequency",
        "hidden_dim",
        "action_embed_dim",
        "latent_dim",
        "cem_horizon",
        "cem_samples",
        "cem_elites",
        "cem_iterations",
    )
    fields = []
    for key in keys:
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, float):
            fields.append(f"{key}={value:.6g}")
        else:
            fields.append(f"{key}={value}")
    return " ".join(fields)


def _base_params(spec: StudySpec, common_overrides: dict[str, Any], seed_base: int) -> dict[str, Any]:
    params = load_json(spec.base_config)
    params.update(common_overrides)
    params["seed"] = seed_base
    return params


def _storage_url(study_dir: Path, spec: StudySpec) -> str:
    return f"sqlite:///{study_dir / f'{spec.name}.sqlite3'}"


def _make_objective(
    *,
    spec: StudySpec,
    common_overrides: dict[str, Any],
    seed_base: int,
    force_run: bool,
    suggest: SuggestFn,
):
    def objective(trial) -> float:
        from src.run_experiment import run_experiment

        params = _base_params(spec, common_overrides, seed_base)
        suggest(trial, params)
        print(
            "[trial-start] "
            f"study={spec.name} trial={trial.number} gpu={spec.gpu} "
            f"{trial_fields(params)}",
            flush=True,
        )
        try:
            output_dir = Path(run_experiment(params, force_run=force_run))
            survival = objective_metric(output_dir)
        except Exception as exc:
            print(f"[trial-fail] study={spec.name} trial={trial.number} error={exc!r}", flush=True)
            raise
        trial.set_user_attr("output_dir", str(output_dir))
        trial.set_user_attr("algo", params["algo"])
        trial.set_user_attr("obs_backend", params["obs_backend"])
        print(
            "[trial-done] "
            f"study={spec.name} trial={trial.number} survival={survival:.6f} "
            f"output_dir={output_dir}",
            flush=True,
        )
        return survival

    return objective


def _redirect_worker_logs(log_dir: Path, spec: StudySpec):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.name}.log"
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = handle
    sys.stderr = handle
    print(f"[log] writing {log_path}", flush=True)
    return handle


def _run_study(
    spec: StudySpec,
    *,
    study_dir: Path,
    log_dir: Path,
    n_trials: int,
    seed_base: int,
    common_overrides: dict[str, Any],
    force_run: bool,
    suggest: SuggestFn,
) -> None:
    _redirect_worker_logs(log_dir, spec)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)

    import optuna

    study_dir.mkdir(parents=True, exist_ok=True)
    storage_url = _storage_url(study_dir, spec)
    sampler = optuna.samplers.TPESampler(seed=seed_base, multivariate=True)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=spec.name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    print(f"[{spec.name}] gpu={spec.gpu} trials={n_trials} storage={storage_url}", flush=True)
    study.optimize(
        _make_objective(
            spec=spec,
            common_overrides=common_overrides,
            seed_base=seed_base,
            force_run=force_run,
            suggest=suggest,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )
    print(f"[{spec.name}] best_value={study.best_value:.6f}", flush=True)
    print(f"[{spec.name}] best_params={json.dumps(study.best_params, sort_keys=True)}", flush=True)


def selected_specs(
    *,
    selected_models: list[str],
    available_models: dict[str, dict[str, Any]],
    min_models: int,
    max_models: int,
) -> list[StudySpec]:
    if not min_models <= len(selected_models) <= max_models:
        raise ValueError(
            f"selected_models must contain between {min_models} and {max_models} "
            f"model keys, got {selected_models!r}"
        )
    unknown = [key for key in selected_models if key not in available_models]
    if unknown:
        raise KeyError(f"Unknown model keys {unknown}; available={sorted(available_models)}")
    specs = []
    for gpu, key in enumerate(selected_models):
        item = available_models[key]
        specs.append(StudySpec(name=str(item["name"]), gpu=gpu, base_config=Path(item["base_config"])))
    return specs


def run_parallel_studies(
    *,
    specs: list[StudySpec],
    study_dir: Path,
    log_dir: Path,
    n_trials: int,
    seed_base: int,
    common_overrides: dict[str, Any],
    force_run: bool,
    suggest: SuggestFn,
) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Per-model logs will be written to: {log_dir}", flush=True)
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_run_study,
            kwargs={
                "spec": spec,
                "study_dir": study_dir,
                "log_dir": log_dir,
                "n_trials": n_trials,
                "seed_base": seed_base,
                "common_overrides": common_overrides,
                "force_run": force_run,
                "suggest": suggest,
            },
            name=spec.name,
        )
        for spec in specs
    ]
    for process in processes:
        process.start()
    failed = False
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed = True
            print(f"{process.name} exited with code {process.exitcode}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(1)
