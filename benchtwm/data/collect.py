"""Collect offline datasets by rolling out a behavior policy in a simulator.

Every logged action is sampled from the behavior policy's probability vector,
so ``behavior_logprobs`` is the exact propensity needed by importance-sampling
OPE. Train, validation, and OPE splits use disjoint episode ids and seeds.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchtwm.envs import make_env
from benchtwm.policies import Policy, sample_action
from benchtwm.utils.io import ensure_dir, load_json, save_json
from benchtwm.utils.metrics import action_stability
from benchtwm.utils.progress import trange

SCHEMA_VERSION = 3
# Behavior policy id recorded for datasets collected with a trained checkpoint.
CHECKPOINT_POLICY = "checkpoint"
CHECKPOINT_POLICY_ID = 4

FLOAT_ARRAYS = {"observations", "next_observations", "rewards", "action_masks", "next_action_masks", "behavior_logprobs", "env_metrics", "next_env_metrics"}
INT_ARRAYS = {"actions", "next_actions", "state_ids", "next_state_ids", "behavior_policy_ids", "episode_ids", "timesteps"}
ARRAYS = (*sorted(FLOAT_ARRAYS), *sorted(INT_ARRAYS), "terminals")


def _checkpoint_policy(config: dict[str, Any]) -> Policy:
    import torch

    from benchtwm.algorithms import load_policy

    run_dir = Path(config["behavior_checkpoint"])
    run_config = load_json(run_dir / "config.json")
    checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu")
    return load_policy(run_config, checkpoint)


def _behavior(config: dict[str, Any], env) -> tuple[Policy, int]:
    name = str(config["behavior_policy"])
    if name == CHECKPOINT_POLICY:
        return _checkpoint_policy(config), CHECKPOINT_POLICY_ID
    if name not in env.behavior_policy_ids:
        raise ValueError(f"{env.spec.env_id} has no behavior policy {name!r}; available: {sorted(env.behavior_policy_ids)}")
    return env.make_policy(name, seed=int(config["seed"]), options=config), env.behavior_policy_ids[name]


def _collect_split(
    *,
    env,
    policy: Policy,
    policy_id: int,
    policy_name: str,
    rng: np.random.Generator,
    num_episodes: int,
    first_episode_id: int,
    first_seed: int,
    deterministic: bool,
    show_progress: bool,
    label: str,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    metric_names = env.spec.metric_names
    columns: dict[str, list] = {key: [] for key in ARRAYS}
    summaries: list[dict] = []
    for local_episode in trange(num_episodes, desc=f"collect {label}", disable=not show_progress):
        episode_id = first_episode_id + local_episode
        obs, info = env.reset(seed=first_seed + local_episode)
        policy.reset_episode()
        initial_state = int(info["state_id"])
        done = False
        total_reward = 0.0
        actions: list[int] = []
        metric_rows = [env.transition_metrics(info)]
        while not done:
            probabilities = policy.action_probabilities(obs, info, deterministic=deterministic)
            action, logprob = sample_action(rng, probabilities)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            policy.observe(action, reward)
            done = bool(terminated or truncated)
            current_metrics = env.transition_metrics(info)
            next_metrics = env.transition_metrics(next_info)
            row = {
                "observations": np.asarray(obs, dtype=np.float32),
                "next_observations": np.asarray(next_obs, dtype=np.float32),
                "actions": action,
                "rewards": float(reward),
                "terminals": done,
                "state_ids": int(info["state_id"]),
                "next_state_ids": int(next_info["state_id"]),
                "action_masks": np.asarray(info["action_mask"], dtype=np.float32),
                "next_action_masks": np.asarray(next_info["action_mask"], dtype=np.float32),
                "behavior_logprobs": logprob,
                "behavior_policy_ids": policy_id,
                "episode_ids": episode_id,
                "timesteps": len(actions),
                "env_metrics": np.asarray([current_metrics[name] for name in metric_names], dtype=np.float32),
                "next_env_metrics": np.asarray([next_metrics[name] for name in metric_names], dtype=np.float32),
            }
            for key, value in row.items():
                columns[key].append(value)
            actions.append(action)
            total_reward += float(reward)
            obs, info = next_obs, next_info
            metric_rows.append(next_metrics)
        # next_actions links each transition to the following logged action.
        columns["next_actions"].extend([*actions[1:], -1])
        stability_l1, abrupt_rate = action_stability(actions, env.action_distance, abrupt_fn=env.action_is_abrupt)
        outcome = env.episode_metrics(total_reward=total_reward, final_info=info, transition_metrics=metric_rows)
        summaries.append(
            {
                "episode_id": episode_id,
                "behavior_policy": policy_name,
                "return": float(total_reward),
                "length": len(actions),
                "initial_state": initial_state,
                "action_stability_l1": stability_l1,
                "abrupt_jump_rate": abrupt_rate,
                **outcome,
                **{f"mean_{name}": float(np.mean([metrics[name] for metrics in metric_rows])) for name in metric_names},
            }
        )
    num_actions = env.num_actions
    arrays: dict[str, np.ndarray] = {}
    for key, values in columns.items():
        if key == "terminals":
            arrays[key] = np.asarray(values, dtype=bool)
        elif key in INT_ARRAYS:
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif key in {"observations", "next_observations"}:
            arrays[key] = np.asarray(values, dtype=np.float32).reshape(-1, env.obs_dim)
        elif key in {"action_masks", "next_action_masks"}:
            arrays[key] = np.asarray(values, dtype=np.float32).reshape(-1, num_actions)
        elif key in {"env_metrics", "next_env_metrics"}:
            arrays[key] = np.asarray(values, dtype=np.float32).reshape(-1, len(metric_names))
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)
    return arrays, summaries


def _split_stats(path: Path, arrays: dict[str, np.ndarray], summaries: list[dict]) -> dict[str, Any]:
    frame = pd.DataFrame(summaries)
    return {
        "path": str(path),
        "num_transitions": int(len(arrays["actions"])),
        "num_episodes": int(len(frame)),
        "return_mean": float(frame["return"].mean()) if not frame.empty else 0.0,
        "success_rate": float(frame["success"].mean()) if not frame.empty else 0.0,
        "mean_episode_length": float(frame["length"].mean()) if not frame.empty else 0.0,
        "behavior_policy_counts": dict(Counter(frame["behavior_policy"])) if not frame.empty else {},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Write ``train.npz``, ``val.npz``, optional ``ope_test.npz``, and metadata to ``output_dir``."""

    output_dir = Path(config["output_dir"])
    paths = {"train": output_dir / "train.npz", "validation": output_dir / "val.npz", "ope_test": output_dir / "ope_test.npz"}
    metadata_path = output_dir / "metadata.json"
    if paths["train"].exists() and paths["validation"].exists() and metadata_path.exists() and not bool(config.get("overwrite", False)):
        if not bool(config.get("quiet", False)):
            print(f"Dataset already exists: {output_dir}")
        return load_json(metadata_path)

    seed = int(config["seed"])
    num_train = int(config["num_train_episodes"])
    num_val = int(config["num_val_episodes"])
    num_ope = int(config.get("num_ope_episodes", 0)) if bool(config.get("ope_enabled", False)) else 0
    env = make_env(config, seed=seed)
    policy, policy_id = _behavior(config, env)
    rng = np.random.default_rng(seed)
    common = {
        "env": env,
        "policy": policy,
        "policy_id": policy_id,
        "policy_name": str(config["behavior_policy"]),
        "rng": rng,
        "deterministic": bool(config.get("behavior_deterministic", False)),
        "show_progress": bool(config.get("show_progress", True)),
    }
    # Episode ids are contiguous across splits; reset seeds are separated by
    # large offsets so the splits never share an initial state stream.
    splits = {
        "train": _collect_split(**common, num_episodes=num_train, first_episode_id=0, first_seed=seed, label="train"),
        "validation": _collect_split(**common, num_episodes=num_val, first_episode_id=num_train, first_seed=seed + num_train + 100_000, label="val"),
    }
    if num_ope > 0:
        splits["ope_test"] = _collect_split(
            **common,
            num_episodes=num_ope,
            first_episode_id=num_train + num_val,
            first_seed=seed + num_train + num_val + 200_000,
            label="ope",
        )
    spec = env.spec
    env.close()

    ensure_dir(output_dir)
    split_metadata = {}
    summaries: list[dict] = []
    for name, (arrays, split_summaries) in splits.items():
        np.savez_compressed(paths[name], **arrays)
        split_metadata[name] = _split_stats(paths[name], arrays, split_summaries)
        summaries.extend(split_summaries)
    pd.DataFrame(summaries).to_csv(output_dir / "episodes_summary.csv", index=False)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": config.get("dataset_id"),
        "config": config,
        "environment": spec.to_dict(),
        "observation": {"dtype": "float32", "dim": spec.obs_dim, "feature_names": list(spec.feature_names)},
        "action": {"type": "discrete", "dtype": "int64", "num_actions": spec.num_actions, "mask_field": "action_masks"},
        "behavior": {"policy": str(config["behavior_policy"]), "policy_id": policy_id, "deterministic": common["deterministic"]},
        **split_metadata,
        "episodes_summary_path": str(output_dir / "episodes_summary.csv"),
        "file_hashes": {paths[name].name: _sha256(paths[name]) for name in splits},
    }
    save_json(metadata, metadata_path)
    if not bool(config.get("quiet", False)):
        print(f"Saved dataset under {output_dir}")
    return metadata
