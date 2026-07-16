"""Run a fresh evaluation for collected best Optuna results.

Edit the config block below and run:

    python scripts/evaluate_best_trials.py

The script loads each collected result folder, restores checkpoint_best.pt,
evaluates it on a new eval seed range, and writes per-model outputs plus a
summary table with deltas against the original eval_summary.json.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import queue
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.algorithms.registry import get_policy_loader
from src.envs.icu_adapter import ICUSepsisEnvWrapper
from src.utils.io import ensure_dir
from src.utils.metrics import action_stability
from src.utils.methods import method_id


# =========================
# Easy config
# =========================
DATASET_SIZE = 100_000
EVAL_SEED_BASE = 700_000
EVAL_EPISODES = 10_000

# One selected model is evaluated at a time on each listed GPU. If you only want
# one GPU, set GPUS = [0]. If CUDA is unavailable, the script falls back to CPU.
GPUS = [0, 1]
PARALLEL_BY_GPU = True

# Set to None to evaluate every collected model folder.
# Otherwise choose from MODEL_GROUPS keys below, e.g.:
#SELECT_MODELS = ["dqn", "ppo", "raw_model-d", "reconstructive_model-c"]
#SELECT_MODELS = ["baselines"]
#SELECT_MODELS = ["world_models"]
SELECT_MODELS: list[str] | None = None

MODEL_GROUPS = {
    "baselines": ["dqn", "ppo"],
    "world_models": ["raw_model-d", "raw_model-c", "reconstructive_model-d", "reconstructive_model-c", "latent_model-d", "latent_model-c"],
    "dqn": ["dqn_offline_expert_{tag}"],
    "ppo": ["ppo_offline_expert_{tag}"],
    "raw_model-d": ["raw_model_discrete_cem_expert_{tag}"],
    "raw_model-c": ["raw_model_continuous_cem_expert_{tag}"],
    "reconstructive_model-d": ["reconstructive_model_discrete_cem_expert_{tag}"],
    "reconstructive_model-c": ["reconstructive_model_continuous_cem_expert_{tag}"],
    "latent_model-d": ["latent_model_discrete_cem_expert_{tag}"],
    "latent_model-c": ["latent_model_continuous_cem_expert_{tag}"],
}

EXPORT_TRAJECTORIES = False
SHOW_PROGRESS = True
PROGRESS_INTERVAL = 1_000
OVERWRITE = False


def _dataset_tag(dataset_size: int) -> str:
    if dataset_size % 1_000 != 0:
        raise ValueError(f"DATASET_SIZE must be a multiple of 1000, got {dataset_size}")
    return f"{dataset_size // 1_000}k"


DATASET_TAG = _dataset_tag(DATASET_SIZE)
COLLECTED_DIR = PROJECT_ROOT / "runs" / f"optuna-best-results-{DATASET_TAG}"
OUTPUT_DIR = PROJECT_ROOT / "runs" / f"optuna-best-results-{DATASET_TAG}-eval-seed{EVAL_SEED_BASE}-n{EVAL_EPISODES}"


SUMMARY_FIELDS = [
    "mean_return",
    "mean_survival",
    "mean_length",
    "mean_action_stability_l1",
    "mean_abrupt_jump_rate",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _expand_selected_models(selected_models: list[str] | None) -> set[str] | None:
    if selected_models is None:
        return None

    expanded: list[str] = []
    unknown: list[str] = []
    stack = list(selected_models)
    while stack:
        key = stack.pop(0)
        if key in MODEL_GROUPS:
            stack = list(MODEL_GROUPS[key]) + stack
        elif "{tag}" in key:
            expanded.append(key.format(tag=DATASET_TAG))
        elif key.endswith(f"_{DATASET_TAG}"):
            expanded.append(key)
        else:
            unknown.append(key)

    if unknown:
        raise KeyError(f"Unknown SELECT_MODELS entries {unknown}; available={sorted(MODEL_GROUPS)}")
    return set(expanded)


def _discover_model_dirs(collected_dir: Path) -> list[Path]:
    if not collected_dir.exists():
        raise FileNotFoundError(f"Collected directory not found: {collected_dir}")
    model_dirs = sorted(path for path in collected_dir.iterdir() if path.is_dir())
    selected = _expand_selected_models(SELECT_MODELS)
    if selected is not None:
        model_dirs = [path for path in model_dirs if path.name in selected]
        missing = sorted(selected.difference(path.name for path in model_dirs))
        if missing:
            raise FileNotFoundError(f"Selected model folders not found under {collected_dir}: {missing}")
    if not model_dirs:
        raise FileNotFoundError(f"No model folders found under {collected_dir}")
    return model_dirs


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu")


def _original_summary(model_dir: Path) -> dict[str, float]:
    eval_summary_path = model_dir / "eval_summary.json"
    if eval_summary_path.exists():
        return {key: float(value) for key, value in _load_json(eval_summary_path).items() if key in SUMMARY_FIELDS}
    metrics_summary_path = model_dir / "metrics_summary.json"
    if metrics_summary_path.exists():
        metrics = _load_json(metrics_summary_path)
        summary = metrics.get("eval_summary", metrics.get("train_summary", {}))
        return {key: float(value) for key, value in summary.items() if key in SUMMARY_FIELDS}
    return {}


def _delta_row(*, model_dir: Path, config: dict[str, Any], original: dict[str, float], fresh: dict[str, float], output_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_folder": model_dir.name,
        "method_id": str(config.get("method_id", "")),
        "algo": str(config.get("algo", "")),
        "obs_backend": str(config.get("obs_backend", "")),
        "dataset_path": str(config.get("dataset_path", "")),
        "original_eval_seed_base": int(config.get("eval_seed_base", -1)),
        "original_eval_episodes": int(config.get("eval_episodes", -1)),
        "fresh_eval_seed_base": EVAL_SEED_BASE,
        "fresh_eval_episodes": EVAL_EPISODES,
        "source_dir": str(model_dir),
        "fresh_output_dir": str(output_dir),
    }
    for field in SUMMARY_FIELDS:
        old_value = original.get(field)
        new_value = fresh.get(field)
        row[f"original_{field}"] = "" if old_value is None else old_value
        row[f"fresh_{field}"] = "" if new_value is None else new_value
        row[f"delta_{field}"] = "" if old_value is None or new_value is None else new_value - old_value
    return row


def _read_existing_summary(output_dir: Path) -> list[dict[str, Any]]:
    json_path = output_dir / "summary.json"
    if json_path.exists():
        payload = _load_json(json_path)
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    csv_path = output_dir / "summary.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def _merge_summary_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_model = {
        str(row.get("model_folder", "")): row
        for row in existing_rows
        if row.get("model_folder")
    }
    order = [str(row.get("model_folder", "")) for row in existing_rows if row.get("model_folder")]
    for row in new_rows:
        model_folder = str(row.get("model_folder", ""))
        if not model_folder:
            continue
        if model_folder not in rows_by_model:
            order.append(model_folder)
        rows_by_model[model_folder] = row
    return [rows_by_model[model_folder] for model_folder in order]


def _write_summary(rows: list[dict[str, Any]], output_dir: Path, *, merge_existing: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    if merge_existing:
        rows = _merge_summary_rows(_read_existing_summary(output_dir), rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(json_path, {"rows": rows})


def _evaluate_policy_with_progress(
    policy,
    *,
    config: dict[str, Any],
    num_episodes: int,
    eval_seed_base: int,
    output_dir: Path,
    checkpoint_name: str,
    export_csv: bool,
    progress_desc: str,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    env = ICUSepsisEnvWrapper(obs_backend=config["obs_backend"])
    method = method_id(config)
    trajectories: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    progress_path = output_dir / "progress.json"

    iterator = tqdm(
        range(num_episodes),
        desc=progress_desc,
        unit="ep",
        dynamic_ncols=True,
        disable=not SHOW_PROGRESS,
        file=sys.stdout,
    )
    survived_count = 0
    for episode_id in iterator:
        obs, info = env.reset(seed=eval_seed_base + episode_id)
        done = False
        total_reward = 0.0
        step_idx = 0
        actions = []
        sofas = [float(info["sofa_score"])]
        initial_state = int(info["state_id"])
        initial_sofa = float(info["sofa_score"])
        episode_start_idx = len(trajectories)

        while not done:
            with torch.inference_mode():
                action = int(policy.act(obs, info, deterministic=True))
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            actions.append(action)
            trajectories.append(
                {
                    "method": method,
                    "algo": config["algo"],
                    "method_id": method,
                    "seed": int(config["seed"]),
                    "eval_seed": int(eval_seed_base + episode_id),
                    "checkpoint": checkpoint_name,
                    "episode_id": int(episode_id),
                    "t": int(step_idx),
                    "state_id": int(info["state_id"]),
                    "sofa_score": float(info["sofa_score"]),
                    "action_idx": int(action),
                    "fluid_bin": int(env.decode_action(action)[0]),
                    "vaso_bin": int(env.decode_action(action)[1]),
                    "reward": float(reward),
                    "done": bool(done),
                    "survived": False,
                    **(
                        {f"state_vector_{i}": float(value) for i, value in enumerate(info["state_vector"])}
                        if config["obs_backend"] == "state_vector"
                        else {}
                    ),
                }
            )
            total_reward += float(reward)
            obs = next_obs
            info = next_info
            sofas.append(float(info["sofa_score"]))
            step_idx += 1

        survived = bool(total_reward > 0)
        survived_count += int(survived)
        stability_l1, abrupt_rate = action_stability(actions)
        episodes.append(
            {
                "method": method,
                "algo": config["algo"],
                "method_id": method,
                "seed": int(config["seed"]),
                "eval_seed": int(eval_seed_base + episode_id),
                "checkpoint": checkpoint_name,
                "episode_id": int(episode_id),
                "return": float(total_reward),
                "survived": survived,
                "length": int(step_idx),
                "initial_state": initial_state,
                "initial_sofa": initial_sofa,
                "mean_sofa": float(sum(sofas) / len(sofas)),
                "action_stability_l1": stability_l1,
                "abrupt_jump_rate": abrupt_rate,
            }
        )
        for row in trajectories[episode_start_idx:]:
            row["survived"] = survived

        completed = episode_id + 1
        running_survival = survived_count / completed
        if SHOW_PROGRESS:
            iterator.set_postfix(surv=f"{running_survival:.4f}", len=step_idx, rows=len(trajectories))
        if completed == 1 or completed % PROGRESS_INTERVAL == 0 or completed == num_episodes:
            _write_json(
                progress_path,
                {
                    "status": "running" if completed < num_episodes else "complete",
                    "completed_episodes": completed,
                    "total_episodes": num_episodes,
                    "eval_seed_base": eval_seed_base,
                    "last_eval_seed": int(eval_seed_base + episode_id),
                    "running_mean_survival": running_survival,
                    "trajectory_rows": len(trajectories),
                },
            )

    env.close()
    episodes_df = pd.DataFrame(episodes)
    trajectories_df = pd.DataFrame(trajectories)
    if export_csv:
        episodes_df.to_csv(output_dir / "episodes_summary.csv", index=False)
        trajectories_df.to_csv(output_dir / "trajectories.csv", index=False)
    return {
        "episodes": episodes_df,
        "trajectories": trajectories_df,
        "summary": {
            "mean_return": float(episodes_df["return"].mean()) if not episodes_df.empty else 0.0,
            "mean_survival": float(episodes_df["survived"].mean()) if not episodes_df.empty else 0.0,
            "mean_length": float(episodes_df["length"].mean()) if not episodes_df.empty else 0.0,
            "mean_action_stability_l1": float(episodes_df["action_stability_l1"].mean()) if not episodes_df.empty else 0.0,
            "mean_abrupt_jump_rate": float(episodes_df["abrupt_jump_rate"].mean()) if not episodes_df.empty else 0.0,
        },
    }


def evaluate_model_dir(model_dir: Path, output_root: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    config = _load_json(config_path)
    checkpoint = _load_checkpoint(model_dir / "checkpoint_best.pt")
    policy_loader = get_policy_loader(str(config["algo"]))
    policy = policy_loader(config, checkpoint)

    model_output_dir = output_root / model_dir.name
    if model_output_dir.exists() and not OVERWRITE:
        existing_summary = model_output_dir / "eval_summary.json"
        if existing_summary.exists():
            fresh_summary = _load_json(existing_summary)
            original = _original_summary(model_dir)
            print(f"[skip-existing] {model_dir.name} output_dir={model_output_dir}", flush=True)
            return _delta_row(
                model_dir=model_dir,
                config=config,
                original=original,
                fresh={key: float(value) for key, value in fresh_summary.items() if key in SUMMARY_FIELDS},
                output_dir=model_output_dir,
            )
        raise FileExistsError(f"Output exists without eval_summary.json: {model_output_dir}")

    model_output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        model_output_dir / "progress.json",
        {
            "status": "starting",
            "completed_episodes": 0,
            "total_episodes": EVAL_EPISODES,
            "eval_seed_base": EVAL_SEED_BASE,
            "model": model_dir.name,
            "algo": str(config["algo"]),
        },
    )
    print(
        "[eval-start] "
        f"model={model_dir.name} algo={config['algo']} episodes={EVAL_EPISODES} seed_base={EVAL_SEED_BASE}",
        flush=True,
    )
    result = _evaluate_policy_with_progress(
        policy,
        config=config,
        num_episodes=EVAL_EPISODES,
        eval_seed_base=EVAL_SEED_BASE,
        output_dir=model_output_dir,
        checkpoint_name=f"best_seed{EVAL_SEED_BASE}_n{EVAL_EPISODES}",
        export_csv=EXPORT_TRAJECTORIES,
        progress_desc=f"eval:{model_dir.name}",
    )
    fresh_summary = result["summary"]
    _write_json(model_output_dir / "eval_summary.json", fresh_summary)
    _write_json(
        model_output_dir / "eval_metadata.json",
        {
            "source_model_dir": str(model_dir),
            "eval_seed_base": EVAL_SEED_BASE,
            "eval_episodes": EVAL_EPISODES,
            "checkpoint": "checkpoint_best.pt",
            "config": config,
        },
    )
    original = _original_summary(model_dir)
    row = _delta_row(model_dir=model_dir, config=config, original=original, fresh=fresh_summary, output_dir=model_output_dir)
    print(
        "[eval-done] "
        f"model={model_dir.name} survival={fresh_summary['mean_survival']:.6f} "
        f"delta={row['delta_mean_survival']}",
        flush=True,
    )
    return row


def _gpu_chunks(model_dirs: list[Path]) -> list[tuple[int | None, list[Path]]]:
    gpu_ids = GPUS or [None]
    chunks: list[list[Path]] = [[] for _ in gpu_ids]
    for index, model_dir in enumerate(model_dirs):
        chunks[index % len(chunks)].append(model_dir)
    return [(gpu_id, chunk) for gpu_id, chunk in zip(gpu_ids, chunks) if chunk]


def _gpu_worker(gpu_id: int | None, model_dirs: list[Path], output_dir: Path, result_queue) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    label = "cpu" if gpu_id is None else f"gpu{gpu_id}"
    print(f"[worker-start] {label} models={len(model_dirs)}", flush=True)
    for model_dir in model_dirs:
        try:
            row = evaluate_model_dir(model_dir, output_dir)
            result_queue.put({"type": "row", "model_folder": model_dir.name, "row": row})
        except Exception as exc:  # noqa: BLE001 - send failures back to the parent process.
            result_queue.put(
                {
                    "type": "error",
                    "model_folder": model_dir.name,
                    "gpu": gpu_id,
                    "error": repr(exc),
                }
            )
    print(f"[worker-done] {label}", flush=True)


def _evaluate_parallel_by_gpu(model_dirs: list[Path]) -> list[dict[str, Any]]:
    chunks = _gpu_chunks(model_dirs)
    if len(chunks) <= 1:
        rows = []
        gpu_id = chunks[0][0] if chunks else None
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        for model_dir in model_dirs:
            rows.append(evaluate_model_dir(model_dir, OUTPUT_DIR))
            _write_summary(rows, OUTPUT_DIR)
        return rows

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_gpu_worker, args=(gpu_id, chunk, OUTPUT_DIR, result_queue))
        for gpu_id, chunk in chunks
    ]
    for process in processes:
        process.start()

    expected_messages = len(model_dirs)
    rows_by_model: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    received_messages = 0
    while received_messages < expected_messages:
        try:
            message = result_queue.get(timeout=5.0)
        except queue.Empty:
            if any(process.is_alive() for process in processes):
                continue
            break
        received_messages += 1
        if message["type"] == "row":
            rows_by_model[message["model_folder"]] = message["row"]
            ordered_rows = [rows_by_model[path.name] for path in model_dirs if path.name in rows_by_model]
            _write_summary(ordered_rows, OUTPUT_DIR)
        else:
            errors.append(message)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            errors.append({"type": "error", "model_folder": "", "gpu": "", "error": f"worker exitcode={process.exitcode}"})

    rows = [rows_by_model[path.name] for path in model_dirs if path.name in rows_by_model]
    if rows:
        _write_summary(rows, OUTPUT_DIR)
    if received_messages < expected_messages:
        errors.append(
            {
                "type": "error",
                "model_folder": "",
                "gpu": "",
                "error": f"received {received_messages}/{expected_messages} worker messages",
            }
        )
    if errors:
        _write_json(OUTPUT_DIR / "errors.json", {"errors": errors})
        raise RuntimeError(f"Parallel evaluation failed for {len(errors)} model(s); see {OUTPUT_DIR / 'errors.json'}")
    return rows


def main() -> int:
    print(f"[config] collected_dir={COLLECTED_DIR}", flush=True)
    print(f"[config] output_dir={OUTPUT_DIR}", flush=True)
    print(f"[config] eval_seed_base={EVAL_SEED_BASE} eval_episodes={EVAL_EPISODES}", flush=True)
    print(f"[config] gpus={GPUS} parallel_by_gpu={PARALLEL_BY_GPU}", flush=True)
    print(f"[config] export_trajectories={EXPORT_TRAJECTORIES}", flush=True)

    model_dirs = _discover_model_dirs(COLLECTED_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PARALLEL_BY_GPU and len(GPUS) > 1 and len(model_dirs) > 1:
        rows = _evaluate_parallel_by_gpu(model_dirs)
    else:
        rows = []
        gpu_id = GPUS[0] if GPUS else None
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        for model_dir in model_dirs:
            rows.append(evaluate_model_dir(model_dir, OUTPUT_DIR))
            _write_summary(rows, OUTPUT_DIR)
    _write_summary(rows, OUTPUT_DIR)
    print(f"[done] evaluated={len(rows)} summary={OUTPUT_DIR / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
