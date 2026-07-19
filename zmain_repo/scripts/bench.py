"""Unified, offline-first command line interface for the benchmark suite.

The historical scripts remain supported. This command discovers their existing
JSON configurations and makes the normal lifecycle explicit:

    logged data -> offline fit -> simulator evaluation -> optional OPE

Examples:
    python scripts/bench.py overview
    python scripts/bench.py list --env epicare --kind offline --tier dev
    python scripts/bench.py dataset --config experiments/datasets/dev/epicare_smart.json
    python scripts/bench.py run --config experiments/offline/dev/epicare_dqn.json
    python scripts/bench.py suite --env epicare --mode offline --tier dev
    python scripts/bench.py inspect-dataset datasets/offline/epicare/dev/smart/1000/train.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.envs.factory import environment_id, make_env
from src.envs.reference import make_reference_policy
from src.evaluation import evaluate_policy
from src.offline.dataset import REQUIRED_ARRAYS
from src.offline.generate_dataset import generate_dataset
from src.run_experiment import run_experiment
from src.utils.formatting import output_dir_for_experiment
from src.utils.io import load_json, save_json
from src.utils.json_handling import get_param_iterable
from src.utils.methods import method_id, train_mode


WORKFLOW_ORDER = ("datasets", "offline", "online")
WORKFLOW_LABELS = {
    "datasets": "1. Logged datasets",
    "offline": "2. Offline training (primary workflow)",
    "online": "3. Online training (simulator baseline)",
}
ENVIRONMENT_LABELS = {
    "icu_sepsis": "ICU-Sepsis",
    "epicare": "EpiCare",
}
ARRAY_ROLES = {
    "observations": "canonical observation",
    "next_observations": "canonical next observation",
    "state_vectors": "continuous agent input",
    "next_state_vectors": "continuous next input",
    "actions": "logged action index",
    "next_actions": "next logged action index",
    "rewards": "logged immediate reward",
    "terminals": "terminal transition flag",
    "state_ids": "diagnostic state id",
    "next_state_ids": "diagnostic next state id",
    "action_masks": "admissible actions at t",
    "next_action_masks": "admissible actions at t+1",
    "behavior_logprobs": "log mu(a_t | o_t)",
    "behavior_policy_ids": "behavior-policy provenance",
    "episode_ids": "trajectory grouping key",
    "timesteps": "within-trajectory index",
    "env_metrics": "adapter metrics at t",
    "next_env_metrics": "adapter metrics at t+1",
    "sofa_scores": "legacy ICU SOFA alias",
    "next_sofa_scores": "legacy ICU next SOFA alias",
}


@dataclass(frozen=True)
class ConfigEntry:
    path: Path
    relative_path: Path
    kind: str
    tier: str
    env_id: str
    algorithm: str
    run_mode: str
    num_variants: int

    @property
    def name(self) -> str:
        return self.path.stem


class Console:
    """Small dependency-free terminal renderer with safe non-TTY output."""

    def __init__(self, color: str = "auto"):
        requested = str(color).lower()
        self.color = requested == "always" or (
            requested == "auto" and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        )
        self.width = 92

    def _paint(self, value: str, code: str) -> str:
        return f"\033[{code}m{value}\033[0m" if self.color else value

    def heading(self, title: str, subtitle: str | None = None) -> None:
        print("=" * self.width)
        print(self._paint(title, "1;36"))
        if subtitle:
            print(subtitle)
        print("=" * self.width)

    def section(self, title: str) -> None:
        print(f"\n{self._paint(title.upper(), '1;34')}")
        print("-" * self.width)

    def note(self, value: str) -> None:
        print(self._paint(f"  {value}", "2"))

    def status(self, label: str, value: str) -> None:
        print(f"{self._paint(label, '1;32')}: {value}")

    def key_values(self, rows: Iterable[tuple[str, Any]]) -> None:
        materialized = [(str(key), _format_value(value)) for key, value in rows]
        if not materialized:
            return
        width = min(30, max(len(key) for key, _ in materialized))
        for key, value in materialized:
            print(f"  {self._paint(key.ljust(width), '1')}  {value}")

    def table(self, headers: Iterable[str], rows: Iterable[Iterable[Any]], *, max_widths: Iterable[int] | None = None) -> None:
        header_values = [str(value) for value in headers]
        row_values = [[str(value) for value in row] for row in rows]
        if not row_values:
            self.note("No entries.")
            return
        widths = [len(value) for value in header_values]
        for row in row_values:
            widths = [max(width, len(value)) for width, value in zip(widths, row)]
        if max_widths is not None:
            widths = [min(width, maximum) for width, maximum in zip(widths, max_widths)]
        clipped_rows = [[_truncate(value, widths[index]) for index, value in enumerate(row)] for row in row_values]
        clipped_headers = [_truncate(value, widths[index]) for index, value in enumerate(header_values)]
        line = "-+-".join("-" * width for width in widths)
        print(" | ".join(self._paint(value.ljust(widths[index]), "1") for index, value in enumerate(clipped_headers)))
        print(line)
        for row in clipped_rows:
            print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[: width - 3]}..."


def _format_value(value: Any) -> str:
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _display_path(path: Path | str) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return str(value)


def _resolved_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_configs(root: Path = PROJECT_ROOT / "experiments") -> list[ConfigEntry]:
    """Return current JSON files as a typed catalog without moving them."""

    entries: list[ConfigEntry] = []
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue
        raw = _load_config(path)
        kind = relative.parts[0]
        tier = relative.parts[1]
        mode = "dataset" if kind == "datasets" else train_mode(raw)
        entries.append(
            ConfigEntry(
                path=path,
                relative_path=relative,
                kind=kind,
                tier=tier,
                env_id=environment_id(raw),
                algorithm=str(raw.get("algo", raw.get("behavior_policy", "dataset"))),
                run_mode=mode,
                num_variants=len(get_param_iterable(raw)),
            )
        )
    return entries


def _filter_entries(
    entries: Iterable[ConfigEntry],
    *,
    env_id: str | None = None,
    kind: str | None = None,
    tier: str | None = None,
    algorithm: str | None = None,
    name: str | None = None,
) -> list[ConfigEntry]:
    selected = list(entries)
    if env_id is not None:
        selected = [entry for entry in selected if entry.env_id == env_id]
    if kind is not None:
        selected = [entry for entry in selected if entry.kind == kind]
    if tier is not None:
        selected = [entry for entry in selected if entry.tier == tier]
    if algorithm is not None:
        selected = [entry for entry in selected if entry.algorithm == algorithm]
    if name is not None:
        selected = [
            entry
            for entry in selected
            if entry.name == name or str(entry.relative_path.with_suffix("")) == name
        ]
    return selected


def _resolve_config(args: argparse.Namespace, *, kind: str | None = None) -> ConfigEntry:
    raw_path = getattr(args, "config", None)
    if raw_path:
        path = _resolved_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        entries = [entry for entry in discover_configs() if entry.path.resolve() == path]
        if entries:
            return entries[0]
        raw = _load_config(path)
        return ConfigEntry(
            path=path,
            relative_path=path,
            kind=kind or "external",
            tier="external",
            env_id=environment_id(raw),
            algorithm=str(raw.get("algo", raw.get("behavior_policy", "dataset"))),
            run_mode="dataset" if kind == "datasets" else train_mode(raw),
            num_variants=len(get_param_iterable(raw)),
        )

    entries = _filter_entries(
        discover_configs(),
        env_id=getattr(args, "env_id", None),
        kind=kind,
        tier=getattr(args, "tier", None),
        algorithm=getattr(args, "algorithm", None),
        name=getattr(args, "name", None),
    )
    if len(entries) == 1:
        return entries[0]
    if not entries:
        raise ValueError("No configuration matched. Use `bench.py list` to inspect available entries.")
    options = "\n".join(f"  - {entry.relative_path}" for entry in entries)
    raise ValueError(f"Configuration selection is ambiguous. Use --config or --name:\n{options}")


def _resolved_variants(entry: ConfigEntry) -> list[dict]:
    return [dict(variant) for variant in get_param_iterable(_load_config(entry.path))]


def _environment_details(config: dict) -> dict[str, Any]:
    """Read the adapter contract for human-facing preflight output."""

    obs_backend = config.get("obs_backend")
    env = make_env(config, obs_backend=obs_backend, seed=int(config.get("seed", 0)))
    try:
        spec = env.spec.to_dict()
    finally:
        env.close()
    return spec


def _print_workflow_hint(console: Console) -> None:
    console.section("Offline-first workflow")
    console.key_values(
        [
            ("1. Build logged data", "bench.py dataset --config experiments/datasets/<tier>/<dataset>.json"),
            ("2. Inspect schema", "bench.py inspect-dataset <dataset>/train.npz"),
            ("3. Preview runs", "bench.py suite --env <env> --mode offline --tier <tier>"),
            ("4. Train", "bench.py run --config experiments/offline/<tier>/<algorithm>.json"),
            ("5. Inspect evidence", "bench.py inspect-run <results-directory>"),
        ]
    )
    console.note("Offline optimization consumes train.npz and val.npz. Direct evaluation uses the simulator; OPE uses ope_test.npz only.")


def _print_catalog(entries: list[ConfigEntry], console: Console) -> None:
    console.heading("BenchTWM configuration catalog", "Discovered experiment configurations; offline entries are shown first.")
    if not entries:
        console.note("No configurations matched.")
        return
    variants = sum(entry.num_variants for entry in entries)
    by_environment = Counter(entry.env_id for entry in entries)
    console.key_values(
        [
            ("Configurations", len(entries)),
            ("Resolved variants", variants),
            ("Environments", ", ".join(f"{ENVIRONMENT_LABELS.get(key, key)} ({value})" for key, value in sorted(by_environment.items()))),
        ]
    )
    for kind in WORKFLOW_ORDER:
        subset = [entry for entry in entries if entry.kind == kind]
        if not subset:
            continue
        console.section(f"{WORKFLOW_LABELS[kind]} | {len(subset)} configurations")
        rows = [
            (
                ENVIRONMENT_LABELS.get(entry.env_id, entry.env_id),
                entry.tier,
                entry.algorithm,
                entry.num_variants,
                entry.relative_path,
            )
            for entry in subset
        ]
        console.table(
            ("environment", "tier", "algorithm / behavior", "variants", "configuration"),
            rows,
            max_widths=(16, 8, 26, 8, 46),
        )


def _print_dataset_plan(console: Console, entry: ConfigEntry, config: dict) -> None:
    spec = _environment_details(config)
    output_dir = _resolved_path(config.get("output_dir", "datasets/offline"))
    console.heading("Dataset build", "A logged dataset is the input to every offline algorithm.")
    console.section("Collection plan")
    console.key_values(
        [
            ("Configuration", entry.relative_path),
            ("Dataset", config.get("dataset_name", output_dir.name)),
            ("Environment", ENVIRONMENT_LABELS.get(spec["env_id"], spec["env_id"])),
            ("Environment seed", config.get("environment_seed", "legacy/default")),
            ("Observation backend", spec["obs_backend"]),
            ("Agent input", f"{spec['obs_dim']} values"),
            ("Action space", f"{spec['num_actions']} actions; {spec['action_geometry']['type']} geometry"),
            ("Behavior policy", config.get("behavior_policy", "unspecified")),
            ("Train / validation episodes", f"{config.get('num_train_episodes', 0)} / {config.get('num_val_episodes', 0)}"),
            ("OPE holdout", f"{config.get('num_ope_episodes', 0)} episodes" if config.get("ope_enabled") else "disabled"),
            ("Output", output_dir),
        ]
    )
    console.note("The generator records behavior_logprobs for each transition. Do not reuse validation data as an OPE holdout.")


def _print_dataset_metadata(console: Console, metadata: dict) -> None:
    environment = metadata["environment"]
    console.section("Created dataset")
    console.key_values(
        [
            ("Schema version", metadata.get("schema_version", "unknown")),
            ("Environment", ENVIRONMENT_LABELS.get(environment["env_id"], environment["env_id"])),
            ("Observation backend", environment["obs_backend"]),
            ("Features", ", ".join(environment.get("feature_names", []))),
            ("Metrics", ", ".join(environment.get("metric_names", []))),
        ]
    )
    rows = []
    for name, label in (("train", "train.npz"), ("validation", "val.npz"), ("ope_test", "ope_test.npz")):
        split = metadata.get(name)
        if not split:
            continue
        rows.append(
            (
                label,
                split.get("num_transitions", 0),
                split.get("num_episodes", 0),
                split.get("mean_episode_length", 0.0),
                split.get("return_mean", 0.0),
                split.get("success_rate", 0.0),
                _display_path(split.get("path", "")),
            )
        )
    console.table(
        ("split", "transitions", "episodes", "mean length", "mean return", "success", "path"),
        rows,
        max_widths=(12, 14, 10, 13, 13, 10, 35),
    )
    console.note("Next: inspect the train split, then preview an offline suite with `bench.py suite ... --mode offline`.")


def _print_algorithm_plan(
    console: Console,
    entry: ConfigEntry,
    params: dict,
    output_dir: Path,
    *,
    offline_checks: dict[str, str] | None = None,
) -> None:
    spec = _environment_details(params)
    offline = train_mode(params) == "offline"
    console.heading(
        "Offline training plan" if offline else "Online training plan",
        "The resolved configuration is shown before any training state is changed.",
    )
    console.section("Identity")
    console.key_values(
        [
            ("Configuration", entry.relative_path),
            ("Environment", ENVIRONMENT_LABELS.get(spec["env_id"], spec["env_id"])),
            ("Algorithm", method_id(params)),
            ("Mode", train_mode(params)),
            ("Seed", params.get("seed", 0)),
            ("Observation backend", spec["obs_backend"]),
            ("Model input", f"{spec['obs_dim']} values"),
            ("Actions", f"{spec['num_actions']} ({spec['action_geometry']['type']})"),
            ("Output", output_dir),
        ]
    )
    if offline:
        console.section("Offline evidence")
        steps = int(params.get("offline_train_steps", 0))
        batch_size = int(params.get("offline_batch_size", 0))
        console.key_values(
            [
                ("Training split", _resolved_path(params["dataset_path"])),
                ("Validation split", _resolved_path(params["val_dataset_path"])),
                ("Dataset preflight", "; ".join(offline_checks.values()) if offline_checks else "not run (suite preview)"),
                ("Optimization budget", f"{steps} gradient steps x batch {batch_size} = {steps * batch_size} sampled transitions"),
                ("Train log interval", f"every {params.get('offline_log_interval', 0)} steps"),
                ("Direct evaluation", f"every {params.get('offline_eval_interval', 0)} steps on {params.get('offline_eval_episodes', 0)} simulator episodes"),
                ("Best checkpoint", "lowest validation loss"),
                ("Final direct evaluation", f"{params.get('eval_episodes', 0)} fresh simulator episodes"),
                ("OPE", _ope_plan_text(params)),
            ]
        )
        console.note("Validation loss selects the checkpoint. OPE is a post-training diagnostic and never selects a model.")
    else:
        console.section("Online budget")
        console.key_values(
            [
                ("Episodes", params.get("max_episodes", 0)),
                ("Final direct evaluation", f"{params.get('eval_episodes', 0)} fresh simulator episodes"),
                ("OPE", "not run for online configs unless an OPE dataset is explicitly configured"),
            ]
        )


def _ope_plan_text(params: dict) -> str:
    if not params.get("ope_enabled") or not params.get("ope_dataset_path"):
        return "disabled"
    return f"enabled on {_display_path(_resolved_path(params['ope_dataset_path']))}; {params.get('ope_bootstrap_replicates', 200)} bootstrap replicates"


def _validate_offline_inputs(params: dict) -> dict[str, str]:
    """Check immutable offline inputs before allocating a trainer or optimizer."""

    checks: dict[str, str] = {}
    configured: tuple[tuple[str, str], ...] = (("train", "dataset_path"), ("validation", "val_dataset_path"))
    if params.get("ope_enabled"):
        if not params.get("ope_dataset_path"):
            raise ValueError("ope_enabled requires ope_dataset_path")
        configured += (("OPE", "ope_dataset_path"),)
    expected_env = environment_id(params)
    expected_backend = str(params["obs_backend"])
    resolved = [(label, _resolved_path(params[key])) for label, key in configured]
    if len({path for _, path in resolved}) != len(resolved):
        raise ValueError("train, validation, and OPE dataset paths must be distinct")
    for label, path in resolved:
        if not path.exists():
            raise FileNotFoundError(f"{label} dataset does not exist: {path}")
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(REQUIRED_ARRAYS - set(payload.files))
            if missing:
                raise ValueError(f"{label} dataset is missing required arrays: {missing}")
            if expected_backend == "state_vector" and not {"state_vectors", "next_state_vectors"}.issubset(payload.files):
                raise ValueError(f"{label} dataset does not contain continuous state vectors required by obs_backend=state_vector")
        metadata_path = path.parent / "metadata.json"
        if metadata_path.exists():
            metadata = load_json(metadata_path)
            environment = metadata.get("environment", {})
            recorded_env = environment.get("env_id")
            recorded_backend = environment.get("obs_backend")
            if recorded_env and recorded_env != expected_env:
                raise ValueError(f"{label} dataset environment={recorded_env!r} does not match config environment={expected_env!r}")
            if recorded_backend and recorded_backend != expected_backend:
                raise ValueError(f"{label} dataset obs_backend={recorded_backend!r} does not match config obs_backend={expected_backend!r}")
            checks[label] = f"{label}: ready ({_display_path(path)})"
        else:
            checks[label] = f"{label}: schema ready; metadata.json absent"
    return checks


def _numeric_stats(values: np.ndarray) -> str:
    if values.size == 0:
        return "empty"
    numeric = np.asarray(values, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return "no finite values"
    return f"mean={finite.mean():.4g}; min={finite.min():.4g}; max={finite.max():.4g}"


def _dataset_profile(path: Path, metadata: dict | None) -> dict[str, Any]:
    """Build a compact, data-derived profile without changing the NPZ schema."""

    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    actions = np.asarray(arrays["actions"], dtype=np.int64)
    episode_ids = np.asarray(arrays["episode_ids"], dtype=np.int64)
    masks = np.asarray(arrays["action_masks"], dtype=np.float32)
    logprobs = np.asarray(arrays["behavior_logprobs"], dtype=np.float64)
    behavior_ids = np.asarray(arrays["behavior_policy_ids"], dtype=np.int64)
    policy_names = {}
    if metadata:
        policy_names = {int(value): key for key, value in metadata.get("policy_ids", {}).items()}
    behavior_counts = Counter(policy_names.get(int(key), str(int(key))) for key in behavior_ids)
    array_rows = [
        (name, "x".join(str(dim) for dim in array.shape) or "scalar", str(array.dtype), ARRAY_ROLES.get(name, "environment extension"))
        for name, array in sorted(arrays.items())
    ]
    primary_input = "state_vectors" if metadata and metadata.get("environment", {}).get("obs_backend") == "state_vector" else "observations"
    metric_names = metadata.get("environment", {}).get("metric_names", []) if metadata else []
    metrics_shape = arrays.get("env_metrics", np.empty((0, 0))).shape
    metric_shape_text = ", ".join(metric_names) if metric_names else "not declared"
    return {
        "num_transitions": int(actions.size),
        "num_episodes": int(np.unique(episode_ids).size),
        "mean_episode_length": float(actions.size / max(1, np.unique(episode_ids).size)),
        "primary_input": primary_input,
        "input_shape": tuple(arrays[primary_input].shape),
        "canonical_obs_shape": tuple(arrays["observations"].shape),
        "num_actions": int(masks.shape[1]),
        "actions_observed": int(np.unique(actions).size),
        "mean_admissible_actions": float(masks.sum(axis=1).mean()),
        "all_rows_have_action": bool(np.all(masks[np.arange(actions.size), actions] > 0.0)),
        "reward_stats": _numeric_stats(np.asarray(arrays["rewards"])),
        "terminal_rate": float(np.asarray(arrays["terminals"], dtype=np.float64).mean()),
        "finite_behavior_logprobs": bool(np.isfinite(logprobs).all()),
        "behavior_logprob_stats": _numeric_stats(logprobs),
        "behavior_counts": behavior_counts,
        "metric_shape": metrics_shape,
        "metric_names": metric_shape_text,
        "array_rows": array_rows,
    }


def _print_dataset_inspection(console: Console, path: Path, metadata: dict | None, profile: dict[str, Any]) -> None:
    environment = (metadata or {}).get("environment", {})
    config = (metadata or {}).get("config", {})
    console.heading("Dataset inspection", "Schema, provenance, and OPE-readiness of one immutable logged split.")
    console.section("Identity")
    console.key_values(
        [
            ("File", path),
            ("Dataset name", config.get("dataset_name", path.parent.name)),
            ("Schema version", (metadata or {}).get("schema_version", "metadata not found")),
            ("Environment", ENVIRONMENT_LABELS.get(environment.get("env_id", "unknown"), environment.get("env_id", "unknown"))),
            ("Observation backend", environment.get("obs_backend", "unknown")),
            ("Behavior policy", config.get("behavior_policy", "recorded per transition")),
        ]
    )
    console.section("Offline model interface")
    console.key_values(
        [
            ("Agent input", f"{profile['primary_input']} with shape {profile['input_shape']}"),
            ("Canonical observation", f"observations with shape {profile['canonical_obs_shape']}"),
            ("Action vector", f"integer action index in [0, {profile['num_actions'] - 1}]"),
            ("Action coverage", f"{profile['actions_observed']} / {profile['num_actions']} actions observed"),
            ("Action-mask check", "all logged actions admissible" if profile["all_rows_have_action"] else "FAILED: logged inadmissible action found"),
            ("Metrics vector", f"env_metrics shape {profile['metric_shape']}; {profile['metric_names']}"),
        ]
    )
    console.section("Split statistics")
    console.key_values(
        [
            ("Transitions / episodes", f"{profile['num_transitions']} / {profile['num_episodes']}"),
            ("Mean episode length", profile["mean_episode_length"]),
            ("Reward", profile["reward_stats"]),
            ("Terminal transition rate", profile["terminal_rate"]),
            ("Mean admissible actions", profile["mean_admissible_actions"]),
            ("Behavior log probabilities", profile["behavior_logprob_stats"]),
            ("OPE precondition", "ready: all behavior_logprobs finite" if profile["finite_behavior_logprobs"] else "FAILED: non-finite behavior_logprobs"),
        ]
    )
    console.section("Behavior provenance")
    console.table(("behavior policy", "transitions"), sorted(profile["behavior_counts"].items()), max_widths=(36, 16))
    console.section("Arrays")
    console.table(("array", "shape", "dtype", "role"), profile["array_rows"], max_widths=(28, 18, 12, 35))


def _json_or_empty(path: Path) -> dict:
    return load_json(path) if path.exists() else {}


def _print_run_inspection(console: Console, run_dir: Path, *, raw: bool) -> None:
    if raw:
        print(f"Run: {run_dir}")
        for filename in ("config.json", "environment_spec.json", "metrics_summary.json", "eval_summary.json", "ope_summary.json"):
            path = run_dir / filename
            if path.exists():
                print(f"\n{filename}:")
                print(path.read_text(encoding="utf-8"))
        return

    config = _json_or_empty(run_dir / "config.json")
    metrics = _json_or_empty(run_dir / "metrics_summary.json")
    evaluation = _json_or_empty(run_dir / "eval_summary.json") or metrics.get("eval_summary", {})
    ope = _json_or_empty(run_dir / "ope_summary.json")
    spec = _json_or_empty(run_dir / "environment_spec.json")
    console.heading("Run inspection", "Training evidence, simulator evaluation, and OPE are reported separately.")
    console.section("Identity")
    console.key_values(
        [
            ("Directory", run_dir),
            ("Environment", ENVIRONMENT_LABELS.get(config.get("env_id", spec.get("env_id", "icu_sepsis")), config.get("env_id", spec.get("env_id", "icu_sepsis")))),
            ("Algorithm", metrics.get("method_id", config.get("algo", "unknown"))),
            ("Mode", metrics.get("train_mode", train_mode(config) if config else "unknown")),
            ("Seed", config.get("seed", metrics.get("seed", "unknown"))),
            ("Observation backend", config.get("obs_backend", spec.get("obs_backend", "unknown"))),
            ("Checkpoints", ", ".join(name for name in ("checkpoint_best.pt", "checkpoint_last.pt") if (run_dir / name).exists()) or "not found"),
        ]
    )
    train_summary = metrics.get("train_summary", {})
    if train_summary:
        console.section("Training summary")
        preferred = [
            "offline_train_steps", "offline_batch_size", "total_sampled_transitions", "effective_epochs",
            "best_validation_loss", "best_validation_step", "final_return", "final_success_rate",
            "mean_episode_length", "mean_action_stability_l1", "mean_abrupt_jump_rate", "early_stop_reason",
        ]
        selected = [(key, train_summary[key]) for key in preferred if key in train_summary]
        selected.extend((key, value) for key, value in train_summary.items() if key not in dict(selected))
        console.key_values(selected)
    if evaluation:
        console.section("Direct simulator evaluation")
        console.key_values(evaluation.items())
        console.note("These values use fresh simulator episodes; they are not offline validation losses and not OPE estimates.")
    if ope:
        console.section("Off-policy evaluation on held-out logs")
        diagnostics = ope.get("diagnostics", {})
        console.key_values(
            [
                ("Dataset", ope.get("config", {}).get("dataset_path", "unknown")),
                ("Deterministic target", ope.get("config", {}).get("deterministic", "unknown")),
                ("Bootstrap replicates", ope.get("config", {}).get("bootstrap_replicates", "unknown")),
                ("Support rate", diagnostics.get("support_rate", "unknown")),
                ("Out-of-support rate", diagnostics.get("out_of_support_rate", "unknown")),
                ("Trajectory ESS", diagnostics.get("trajectory_ess", "unknown")),
                ("Zero-weight trajectories", diagnostics.get("zero_weight_rate", "unknown")),
            ]
        )
        estimator_rows = [
            (name, values.get("estimate"), values.get("ci95_low"), values.get("ci95_high"))
            for name, values in ope.get("estimators", {}).items()
        ]
        console.table(("estimator", "estimate", "95% CI low", "95% CI high"), estimator_rows, max_widths=(14, 16, 16, 16))
        console.note("Low support or low ESS means the OPE numbers are diagnostics, not a reliable policy ranking.")
    console.section("Artifacts")
    artifact_rows = []
    for filename in sorted(path.name for path in run_dir.iterdir() if path.is_file()):
        size_kib = (run_dir / filename).stat().st_size / 1024.0
        artifact_rows.append((filename, f"{size_kib:.1f} KiB"))
    console.table(("file", "size"), artifact_rows, max_widths=(48, 16))


def _cmd_overview(args: argparse.Namespace) -> int:
    console = Console(args.color)
    entries = discover_configs()
    by_kind = Counter(entry.kind for entry in entries)
    by_env = Counter(entry.env_id for entry in entries)
    console.heading("BenchTWM", "Structured offline RL and world-model benchmark operations.")
    console.section("Repository status")
    console.key_values(
        [
            ("Environments", ", ".join(f"{ENVIRONMENT_LABELS.get(env, env)} ({count} configs)" for env, count in sorted(by_env.items()))),
            ("Dataset configurations", by_kind["datasets"]),
            ("Offline configurations", by_kind["offline"]),
            ("Online configurations", by_kind["online"]),
            ("Primary use", "fixed logged datasets, validation-selected checkpoints, direct simulator evaluation, optional OPE"),
        ]
    )
    _print_workflow_hint(console)
    console.section("Start here")
    console.key_values(
        [
            ("Explore", "python scripts/bench.py list --kind offline"),
            ("EpiCare dev dataset", "python scripts/bench.py dataset --config experiments/datasets/dev/epicare_smart.json"),
            ("EpiCare dev offline plan", "python scripts/bench.py suite --env epicare --mode offline --tier dev"),
            ("ICU dev offline plan", "python scripts/bench.py suite --env icu_sepsis --mode offline --tier dev"),
        ]
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    entries = _filter_entries(
        discover_configs(),
        env_id=args.env_id,
        kind=args.kind,
        tier=args.tier,
        algorithm=args.algorithm,
        name=args.name,
    )
    _print_catalog(entries, Console(args.color))
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    entry = _resolve_config(args, kind="datasets")
    raw = _load_config(entry.path)
    console = Console(args.color)
    _print_dataset_plan(console, entry, raw)
    metadata = generate_dataset(raw)
    _print_dataset_metadata(console, metadata)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    entry = _resolve_config(args, kind=args.mode)
    variants = _resolved_variants(entry)
    if args.index < 0 or args.index >= len(variants):
        raise IndexError(f"index={args.index} outside [0, {len(variants) - 1}]")
    params = variants[args.index]
    output_dir = output_dir_for_experiment({**params, "method_id": method_id(params)})
    console = Console(args.color)
    offline_checks = _validate_offline_inputs(params) if train_mode(params) == "offline" else None
    _print_algorithm_plan(console, entry, params, output_dir, offline_checks=offline_checks)
    console.status("Starting", f"variant {args.index + 1}/{len(variants)}")
    actual_output_dir = run_experiment(params, force_run=args.force)
    console.status("Completed", _display_path(actual_output_dir))
    _print_run_inspection(console, Path(actual_output_dir), raw=False)
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    entries = _filter_entries(
        discover_configs(),
        env_id=args.env_id,
        kind=args.mode,
        tier=args.tier,
        algorithm=args.algorithm,
    )
    if not entries:
        raise ValueError("No configurations matched the requested suite.")
    planned: list[tuple[ConfigEntry, int, dict]] = []
    for entry in entries:
        for index, params in enumerate(_resolved_variants(entry)):
            planned.append((entry, index, params))
    console = Console(args.color)
    title = "Offline suite plan" if args.mode == "offline" else "Online suite plan"
    console.heading(title, "Preview is safe. Use --execute only after reviewing the resolved runs below.")
    console.section("Selection")
    console.key_values(
        [
            ("Environment", ENVIRONMENT_LABELS.get(args.env_id, args.env_id)),
            ("Tier", args.tier),
            ("Mode", args.mode),
            ("Algorithm filter", args.algorithm or "all available"),
            ("Resolved runs", len(planned)),
        ]
    )
    rows = []
    for entry, index, params in planned:
        detail = _offline_detail(params) if args.mode == "offline" else f"episodes={params.get('max_episodes', '?')}"
        rows.append((entry.algorithm, f"{index + 1}/{entry.num_variants}", detail, entry.relative_path))
    console.section("Run plan")
    console.table(("algorithm", "variant", "budget", "configuration"), rows, max_widths=(28, 10, 42, 46))
    if not args.execute:
        console.status("Preview only", "Re-run with --execute to start this suite.")
        return 0
    for ordinal, (entry, index, params) in enumerate(planned, start=1):
        console.section(f"Run {ordinal}/{len(planned)} | {method_id(params)}")
        _print_algorithm_plan(console, entry, params, output_dir_for_experiment({**params, "method_id": method_id(params)}))
        run_experiment(params, force_run=args.force)
    console.status("Suite completed", f"{len(planned)} resolved runs")
    return 0


def _offline_detail(params: dict) -> str:
    steps = int(params.get("offline_train_steps", 0))
    batch = int(params.get("offline_batch_size", 0))
    ope = " + OPE" if params.get("ope_enabled") and params.get("ope_dataset_path") else ""
    return f"{_compact_count(steps)} steps x {batch} = {_compact_count(steps * batch)} samples{ope}"


def _compact_count(value: int) -> str:
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}k"
    return str(value)


def _cmd_baselines(args: argparse.Namespace) -> int:
    entry = _resolve_config(args, kind="datasets")
    config = _load_config(entry.path)
    output_root = _resolved_path(args.output_dir)
    console = Console(args.color)
    console.heading("Reference-policy evaluation", "Direct simulator baselines declared by the environment adapter.")
    console.key_values(
        [
            ("Environment", ENVIRONMENT_LABELS.get(environment_id(config), environment_id(config))),
            ("Episodes per policy", args.episodes),
            ("Policies", args.policies),
            ("Output", output_root),
        ]
    )
    summaries = {}
    for name in args.policies:
        console.status("Evaluating", name)
        env = make_env(config, obs_backend=config.get("obs_backend"))
        try:
            policy = make_reference_policy(env, name, seed=int(config.get("seed", 0)))
        finally:
            env.close()
        policy_config = {**config, "algo": "baseline", "method_id": name}
        result = evaluate_policy(
            policy,
            config=policy_config,
            num_episodes=args.episodes,
            eval_seed_base=int(config.get("eval_seed_base", 100000)),
            output_dir=output_root / name,
            checkpoint_name=name,
            export_csv=True,
            show_progress=True,
        )
        summaries[name] = result["summary"]
    save_json(summaries, output_root / "baseline_summary.json")
    console.section("Direct evaluation summary")
    preferred = ("mean_return", "mean_success", "mean_adverse_event_rate", "mean_symptom_burden", "mean_length")
    rows = [(name, *[summary.get(key, "-") for key in preferred]) for name, summary in summaries.items()]
    console.table(("policy", "return", "success", "adverse events", "primary metric", "length"), rows, max_widths=(16, 14, 12, 16, 16, 12))
    console.status("Artifacts", _display_path(output_root))
    return 0


def _cmd_inspect_dataset(args: argparse.Namespace) -> int:
    path = _resolved_path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)
    metadata_path = path.parent / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else None
    profile = _dataset_profile(path, metadata)
    console = Console(args.color)
    _print_dataset_inspection(console, path, metadata, profile)
    if args.full_metadata and metadata is not None:
        console.section("Raw metadata")
        print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _cmd_inspect_run(args: argparse.Namespace) -> int:
    run_dir = _resolved_path(args.path)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    _print_run_inspection(Console(args.color), run_dir, raw=bool(args.raw))
    return 0


def _add_selector_arguments(parser: argparse.ArgumentParser, *, include_algorithm: bool = True) -> None:
    parser.add_argument("--config", help="Explicit JSON config path; takes precedence over catalog filters.")
    parser.add_argument("--env", dest="env_id", choices=["icu_sepsis", "epicare"], help="Environment filter.")
    parser.add_argument("--tier", choices=["dev", "final"], help="Configuration tier filter.")
    if include_algorithm:
        parser.add_argument("--algorithm", help="Algorithm filter, e.g. dqn or raw_model.")
    parser.add_argument("--name", help="Config stem or relative path without .json.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto", help="Terminal color policy (default: auto).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    overview_parser = subparsers.add_parser("overview", help="Show the offline-first workflow and repository catalog counts.")
    overview_parser.set_defaults(handler=_cmd_overview)

    list_parser = subparsers.add_parser("list", help="List configurations grouped by the benchmark workflow.")
    list_parser.add_argument("--env", dest="env_id", choices=["icu_sepsis", "epicare"])
    list_parser.add_argument("--kind", choices=["datasets", "online", "offline"])
    list_parser.add_argument("--tier", choices=["dev", "final"])
    list_parser.add_argument("--algorithm")
    list_parser.add_argument("--name")
    list_parser.set_defaults(handler=_cmd_list)

    dataset_parser = subparsers.add_parser("dataset", help="Generate and summarize one offline dataset.")
    _add_selector_arguments(dataset_parser, include_algorithm=False)
    dataset_parser.set_defaults(handler=_cmd_dataset)

    run_parser = subparsers.add_parser("run", help="Preflight and run one resolved online or offline experiment.")
    _add_selector_arguments(run_parser)
    run_parser.add_argument("--mode", choices=["online", "offline"], default="offline")
    run_parser.add_argument("--index", type=int, default=0, help="Index in the JSON parameter sweep.")
    run_parser.add_argument("--force", action="store_true", help="Re-run even if an output already exists.")
    run_parser.set_defaults(handler=_cmd_run)

    suite_parser = subparsers.add_parser("suite", help="Preview or execute all matching configurations.")
    suite_parser.add_argument("--env", dest="env_id", choices=["icu_sepsis", "epicare"], required=True)
    suite_parser.add_argument("--mode", choices=["online", "offline"], required=True)
    suite_parser.add_argument("--tier", choices=["dev", "final"], default="dev")
    suite_parser.add_argument("--algorithm", help="Limit the suite to one algorithm family.")
    suite_parser.add_argument("--execute", action="store_true", help="Start jobs; otherwise only print the plan.")
    suite_parser.add_argument("--force", action="store_true", help="Re-run completed output directories.")
    suite_parser.set_defaults(handler=_cmd_suite)

    baseline_parser = subparsers.add_parser("baselines", help="Directly evaluate an environment's declared reference policies.")
    _add_selector_arguments(baseline_parser, include_algorithm=False)
    baseline_parser.add_argument("--episodes", type=int, default=1000)
    baseline_parser.add_argument("--output-dir", default="runs/baselines")
    baseline_parser.add_argument("--policies", nargs="+", default=["random", "smart", "soc", "oracle"])
    baseline_parser.set_defaults(handler=_cmd_baselines)

    inspect_dataset_parser = subparsers.add_parser("inspect-dataset", help="Summarize an NPZ schema, provenance, and OPE preconditions.")
    inspect_dataset_parser.add_argument("path")
    inspect_dataset_parser.add_argument("--full-metadata", action="store_true", help="Append the unabridged metadata.json payload.")
    inspect_dataset_parser.set_defaults(handler=_cmd_inspect_dataset)

    inspect_run_parser = subparsers.add_parser("inspect-run", help="Summarize training, direct evaluation, OPE, and run artifacts.")
    inspect_run_parser.add_argument("path")
    inspect_run_parser.add_argument("--raw", action="store_true", help="Print the original JSON artifacts instead of the compact report.")
    inspect_run_parser.set_defaults(handler=_cmd_inspect_run)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return int(args.handler(args))
    except (FileNotFoundError, IndexError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
